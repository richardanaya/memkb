use clap::Parser;
use futures::future::{BoxFuture, FutureExt};
use hyper::{
    Method, Request, Response, StatusCode,
    body::{Bytes, Frame, Incoming},
    service::Service,
};
use hyper_util::{
    rt::{TokioExecutor, TokioIo},
    server::conn::auto::Builder,
    service::TowerToHyperService,
};
// StreamExt imported locally where needed
use http_body_util::{BodyExt, Full, StreamBody};
use openai_api_rs::v1::{
    api::OpenAIClient,
    chat_completion::{self, ChatCompletionMessage, ChatCompletionRequest, MessageRole},
    embedding::{EmbeddingRequest, EncodingFormat},
};
use rmcp::handler::server::tool::Parameters;
use rmcp::transport::streamable_http_server::{
    StreamableHttpService, session::local::LocalSessionManager,
};
use rmcp::{
    ErrorData as McpError, handler::server::router::tool::ToolRouter, model::*, tool, tool_handler,
    tool_router,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use simsimd::SpatialSimilarity;
use std::{fs, io::Write, sync::Arc};
use text_splitter::{ChunkConfig, MarkdownSplitter};
use tokio::sync::Mutex;
use walkdir::WalkDir;

const SYSTEM_PROMPT: &str = "You are a helpful assistant that answers the user's specific question. Your goal is to directly respond to what the user is asking. You must follow these rules:
1. Focus solely on answering the user's specific question using information from the provided context
2. Only use information that is explicitly present in the context provided
3. If the context doesn't contain relevant information to answer the user's question, respond with \"We don't have knowledge on that question\" and DO NOT include any citations or references section
4. Never make up information or provide answers from your general knowledge
5. Always be concise and direct
6. Answer directly without any preambles or references to documents, text, context, or chunks - just give the answer immediately
7. IMPORTANT: You must cite your sources using superscript numbers. When you reference information from the context, use superscript numbers like ¹, ², ³ etc. corresponding to the [Citation X] numbers in the context
8. At the end of your response, include a references section with numbered citations like: 1. filename.md (Line X-Y)
9. Use multiple citations when drawing from different sources, and be specific about which information comes from which citation
10. CRITICAL: Only include citations and references when you actually answer the question using the provided context. If you respond with \"We don't have knowledge on that question\", include NO citations or references";

#[derive(Parser)]
#[command(name = "memkb")]
#[command(about = "Memory knowledge base MCP server")]
struct Args {
    #[arg(short, long, default_value = "8080")]
    port: u16,

    #[arg(short = 'H', long, default_value = "localhost")]
    host: String,

    #[arg(short, long, default_value = ".")]
    directory: String,

    #[arg(short, long)]
    embedding_url: Option<String>,

    #[arg(short, long)]
    generation_url: Option<String>,

    #[arg(short = 'c', long, default_value = "1000")]
    chunk_size: usize,

    #[arg(short = 'o', long, default_value = "200")]
    overlap: usize,

    #[arg(long)]
    http: bool,

    #[arg(short = 'j', long, default_value = "1")]
    parallel_embeddings: usize,
}

#[derive(Serialize, Deserialize, JsonSchema)]
struct AskRequest {
    prompt: String,
}

#[derive(Serialize, Deserialize, Clone)]
struct Message {
    role: String,
    content: String,
}

#[derive(Serialize, Deserialize)]
struct StreamRequest {
    model: Option<String>,
    messages: Vec<Message>,
    stream: bool,
}

#[derive(Clone, Debug)]
struct TextChunk {
    content: String,
    source_file: String,
    start_line: usize,
    end_line: usize,
    embedding: Option<Vec<f32>>,
}

pub struct MemoryKB {
    directory: String,
    chunks: Arc<Mutex<Vec<TextChunk>>>,
    embedding_client: Option<Arc<Mutex<OpenAIClient>>>,
    generation_client: Option<Arc<Mutex<OpenAIClient>>>,
    tool_router: ToolRouter<Self>,
}

#[tool_router]
impl MemoryKB {
    fn new(
        directory: String,
        embedding_client: Option<Arc<Mutex<OpenAIClient>>>,
        generation_client: Option<Arc<Mutex<OpenAIClient>>>,
    ) -> Self {
        Self {
            directory,
            chunks: Arc::new(Mutex::new(Vec::new())),
            embedding_client,
            generation_client,
            tool_router: Self::tool_router(),
        }
    }

    #[tool(
        name = "ask_question",
        description = "Ask a question of the knowledge base"
    )]
    async fn ask(&self, params: Parameters<AskRequest>) -> Result<CallToolResult, McpError> {
        let prompt = &params.0.prompt;

        // If no embedding client, fall back to returning all content
        if self.embedding_client.is_none() {
            let merged_content = match read_and_merge_markdown_files(&self.directory) {
                Ok(content) => content,
                Err(e) => {
                    return Ok(CallToolResult::success(vec![rmcp::model::Content::text(
                        format!("Error reading markdown files: {e}"),
                    )]));
                }
            };

            // Try to generate answer with all content if generation client is available
            if let Some(gen_client) = &self.generation_client {
                match generate_answer(gen_client, prompt, &merged_content).await {
                    Ok(answer) => {
                        return Ok(CallToolResult::success(vec![rmcp::model::Content::text(
                            answer,
                        )]));
                    }
                    Err(e) => {
                        return Ok(CallToolResult::success(vec![rmcp::model::Content::text(
                            format!(
                                "Error generating answer: {e}. Returning raw content:\n\n{merged_content}"
                            ),
                        )]));
                    }
                }
            } else {
                return Ok(CallToolResult::success(vec![rmcp::model::Content::text(
                    merged_content,
                )]));
            }
        }

        // Generate embedding for the query
        let query_embedding = match &self.embedding_client {
            Some(client) => {
                let mut client_guard = client.lock().await;
                match generate_embedding(&mut client_guard, prompt).await {
                    Ok(embedding) => embedding,
                    Err(e) => {
                        return Ok(CallToolResult::success(vec![rmcp::model::Content::text(
                            format!("Error generating query embedding: {e}"),
                        )]));
                    }
                }
            }
            None => {
                return Ok(CallToolResult::success(vec![rmcp::model::Content::text(
                    "No embedding client available".to_string(),
                )]));
            }
        };

        // Find similar chunks
        let chunks = self.chunks.lock().await;
        let similar_chunks = find_similar_chunks(&query_embedding, &chunks, 5).await;

        if similar_chunks.is_empty() {
            return Ok(CallToolResult::success(vec![rmcp::model::Content::text(
                "No relevant content found.".to_string(),
            )]));
        }

        // Collect relevant content for context
        let mut context = String::new();
        for (i, (similarity, chunk)) in similar_chunks.iter().enumerate() {
            context.push_str(&format!(
                "Context {} (Similarity: {:.3}) - From: {} (Line {}-{})\n{}\n\n",
                i + 1,
                similarity,
                chunk.source_file,
                chunk.start_line,
                chunk.end_line,
                chunk.content
            ));
        }

        // Generate answer using the context
        if let Some(gen_client) = &self.generation_client {
            match generate_answer(gen_client, prompt, &context).await {
                Ok(answer) => Ok(CallToolResult::success(vec![rmcp::model::Content::text(
                    answer,
                )])),
                Err(e) => {
                    // Fall back to returning chunks if generation fails
                    let fallback_response = format!(
                        "Error generating answer: {e}. Here are the relevant chunks:\n\n{context}"
                    );
                    Ok(CallToolResult::success(vec![rmcp::model::Content::text(
                        fallback_response,
                    )]))
                }
            }
        } else {
            // No generation client, return formatted chunks
            let response = format!(
                "Top {} relevant chunks for your query:\n\n{}",
                similar_chunks.len(),
                context
            );
            Ok(CallToolResult::success(vec![rmcp::model::Content::text(
                response,
            )]))
        }
    }
}

// Implement the server handler
#[tool_handler]
impl rmcp::ServerHandler for MemoryKB {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            instructions: Some(
                "Memory knowledge base server - ask questions about markdown content".into(),
            ),
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            ..Default::default()
        }
    }
}

async fn test_embedding_server(url: &str) -> anyhow::Result<()> {
    let mut client = OpenAIClient::builder()
        .with_endpoint(url)
        .with_api_key("test")
        .build()
        .unwrap();

    let mut req = EmbeddingRequest::new(
        "text-embedding-3-small".to_string(),
        vec!["Hello, this is a test embedding".to_string()],
    );
    req.encoding_format = Some(EncodingFormat::Float);

    match client.embedding(req).await {
        Ok(_) => {
            println!("✅ Embedding server test: PASSED");
            Ok(())
        }
        Err(e) => {
            let msg = sanitize_error_message(&format!("{e}"));
            println!("❌ Embedding server test: FAILED - {msg}");
            Err(anyhow::anyhow!("Embedding server test failed: {}", msg))
        }
    }
}

async fn test_generation_server(url: &str) -> anyhow::Result<()> {
    let mut client = OpenAIClient::builder()
        .with_endpoint(url)
        .with_api_key("test")
        .build()
        .unwrap();

    let req = ChatCompletionRequest::new(
        "Generate".to_string(),
        vec![ChatCompletionMessage {
            role: MessageRole::user,
            content: chat_completion::Content::Text("hey".to_string()),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }],
    );

    match client.chat_completion(req).await {
        Ok(_) => {
            println!("✅ Generation server test: PASSED");
            Ok(())
        }
        Err(e) => {
            println!("❌ Generation server test: FAILED - {e}");
            Err(anyhow::anyhow!("Generation server test failed: {}", e))
        }
    }
}

fn scan_directory_for_md_files(dir_path: &str) -> anyhow::Result<(Vec<String>, usize)> {
    let mut md_files = Vec::new();

    for entry in WalkDir::new(dir_path) {
        let entry = entry?;
        if entry.file_type().is_file() {
            if let Some(extension) = entry.path().extension() {
                if extension == "md" {
                    if let Some(path_str) = entry.path().to_str() {
                        md_files.push(path_str.to_string());
                    }
                }
            }
        }
    }

    let count = md_files.len();
    Ok((md_files, count))
}

fn read_and_merge_markdown_files(dir_path: &str) -> anyhow::Result<String> {
    let mut merged_content = String::new();

    for entry in WalkDir::new(dir_path) {
        let entry = entry?;
        if entry.file_type().is_file() {
            if let Some(extension) = entry.path().extension() {
                if extension == "md" {
                    let file_path = entry.path();
                    let content = fs::read_to_string(file_path)?;

                    merged_content.push_str(&format!("\n\n# File: {}\n\n", file_path.display()));
                    merged_content.push_str(&content);
                    merged_content.push_str("\n\n---\n");
                }
            }
        }
    }

    Ok(merged_content)
}

fn chunk_text_with_lines(
    text: &str,
    chunk_size: usize,
    overlap: usize,
) -> Vec<(String, usize, usize)> {
    // Create a text splitter with specified chunk size and overlap
    let cfg = ChunkConfig::new(chunk_size).with_overlap(overlap).unwrap();
    let splitter = MarkdownSplitter::new(cfg);

    let mut chunks_with_lines = Vec::new();
    let _lines: Vec<&str> = text.lines().collect();

    // Get chunks from the splitter
    let chunks: Vec<String> = splitter.chunks(text).map(|s| s.to_string()).collect();

    for chunk in chunks {
        // Find the line numbers for this chunk by searching in the original text
        let mut start_line = 1;
        let mut end_line = 1;

        // Find the first occurrence of the chunk in the text
        if let Some(chunk_start) = text.find(&chunk) {
            // Count newlines before this position to get start line
            start_line = text[..chunk_start].matches('\n').count() + 1;

            // Count newlines within the chunk to get end line
            let chunk_newlines = chunk.matches('\n').count();
            end_line = start_line + chunk_newlines;
        }

        chunks_with_lines.push((chunk, start_line, end_line));
    }

    chunks_with_lines
}

// Sanitize noisy error messages to avoid dumping massive embedding arrays
fn sanitize_error_message(raw: &str) -> String {
    let mut msg = raw.to_string();

    // Trim off any raw JSON payload after a known marker
    if let Some(idx) = msg.find(" / response ") {
        msg.truncate(idx);
    }

    // Collapse very large nested arrays like "[[...]]"
    if let Some(start) = msg.find("[[") {
        if let Some(rel_end) = msg[start..].find("]]") {
            let end = start + rel_end + 2; // include the closing "]]"
            msg.replace_range(start..end, "[[...]]");
        } else {
            msg.replace_range(start.., "[[...]]");
        }
    }

    // Final safety cap on length
    const MAX_LEN: usize = 500;
    if msg.len() > MAX_LEN {
        msg.truncate(MAX_LEN);
        msg.push('…');
    }

    msg
}

async fn chunk_and_embed_files(
    dir_path: &str,
    embedding_client: &Option<Arc<Mutex<OpenAIClient>>>,
    chunk_size: usize,
    overlap: usize,
    parallel_embeddings: usize,
) -> anyhow::Result<Vec<TextChunk>> {
    let mut chunks = Vec::new();

    // First pass: collect all chunks without embeddings
    for entry in WalkDir::new(dir_path) {
        let entry = entry?;
        if entry.file_type().is_file() {
            if let Some(extension) = entry.path().extension() {
                if extension == "md" {
                    let file_path = entry.path();
                    let content = fs::read_to_string(file_path)?;
                    let source_file = file_path.to_string_lossy().to_string();

                    // Chunk the content with configurable size and overlap
                    let text_chunks = chunk_text_with_lines(&content, chunk_size, overlap);

                    for (chunk_content, start_line, end_line) in text_chunks {
                        let chunk = TextChunk {
                            content: chunk_content,
                            source_file: source_file.clone(),
                            start_line,
                            end_line,
                            embedding: None,
                        };
                        chunks.push(chunk);
                    }
                }
            }
        }
    }

    // Second pass: generate embeddings with parallel processing
    if let Some(client) = embedding_client {
        let total_chunks = chunks.len();
        println!(
            "📊 Generating embeddings for {total_chunks} chunks with {parallel_embeddings} parallel workers..."
        );

        use tokio::sync::Semaphore;

        let semaphore = Arc::new(Semaphore::new(parallel_embeddings));
        let completed_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));

        let tasks: Vec<_> = chunks
            .iter_mut()
            .enumerate()
            .map(|(i, chunk)| {
                let client = client.clone();
                let semaphore = semaphore.clone();
                let completed_count = completed_count.clone();
                let chunk_content = chunk.content.clone();
                let chunk_source = chunk.source_file.clone();

                async move {
                    let _permit = semaphore.acquire().await.unwrap();

                    let mut client_guard = client.lock().await;
                    let result = generate_embedding(&mut client_guard, &chunk_content).await;
                    drop(client_guard);

                    let current_count =
                        completed_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                    print!(
                        "\r🔄 Embedding chunk {}/{} ({:.1}%)",
                        current_count,
                        total_chunks,
                        (current_count as f32 / total_chunks as f32) * 100.0
                    );
                    std::io::stdout().flush().unwrap();

                    match result {
                        Ok(embedding) => (i, Some(embedding), None),
                        Err(e) => {
                            let msg = sanitize_error_message(&format!("{e}"));
                            (i, None, Some((chunk_source, msg)))
                        }
                    }
                }
            })
            .collect();

        let results = futures::future::join_all(tasks).await;

        // Apply results back to chunks
        for (index, embedding, error) in results {
            if let Some(embedding) = embedding {
                chunks[index].embedding = Some(embedding);
            } else if let Some((source_file, msg)) = error {
                println!("\n⚠️  Failed to embed chunk from {}: {}", source_file, msg);
            }
        }

        println!(); // New line after progress counter
    }

    Ok(chunks)
}

async fn generate_embedding(client: &mut OpenAIClient, text: &str) -> anyhow::Result<Vec<f32>> {
    let mut req =
        EmbeddingRequest::new("text-embedding-3-small".to_string(), vec![text.to_string()]);
    req.encoding_format = Some(EncodingFormat::Float);

    let response = client.embedding(req).await?;

    if let Some(embedding_data) = response.data.first() {
        Ok(embedding_data.embedding.clone())
    } else {
        Err(anyhow::anyhow!("No embedding data returned"))
    }
}

// Generate search questions based on conversation context
async fn generate_search_questions(
    client: &Arc<Mutex<OpenAIClient>>,
    messages: &[Message],
) -> anyhow::Result<Vec<String>> {
    let mut client_guard = client.lock().await;

    // Build conversation context for question generation
    let mut conversation_context = String::new();
    for msg in messages.iter().take(10) {
        // Last 10 messages for context
        conversation_context.push_str(&format!("{}: {}\n", msg.role, msg.content));
    }

    let system_prompt = "You are a search query generator. Given a conversation, generate 2-3 specific search questions that would help provide relevant context from a knowledge base to continue the conversation naturally. Return only the questions, one per line, without numbering or bullets.";

    let user_prompt = format!(
        "Conversation:\n{}\n\nGenerate 2-3 specific search questions that would help provide relevant context:",
        conversation_context.trim()
    );

    let req = ChatCompletionRequest::new(
        "gpt-3.5-turbo".to_string(),
        vec![
            ChatCompletionMessage {
                role: MessageRole::system,
                content: chat_completion::Content::Text(system_prompt.to_string()),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            },
            ChatCompletionMessage {
                role: MessageRole::user,
                content: chat_completion::Content::Text(user_prompt),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            },
        ],
    );

    let response = client_guard.chat_completion(req).await?;

    if let Some(choice) = response.choices.first() {
        if let Some(content) = &choice.message.content {
            // Parse questions from response (one per line)
            let questions: Vec<String> = content
                .lines()
                .map(|line| line.trim().to_string())
                .filter(|line| !line.is_empty())
                .collect();
            Ok(questions)
        } else {
            Err(anyhow::anyhow!(
                "No content in question generation response"
            ))
        }
    } else {
        Err(anyhow::anyhow!(
            "No choices in question generation response"
        ))
    }
}

async fn generate_answer(
    client: &Arc<Mutex<OpenAIClient>>,
    question: &str,
    context: &str,
) -> anyhow::Result<String> {
    let mut client_guard = client.lock().await;

    let system_prompt = SYSTEM_PROMPT;

    let user_prompt = format!(
        "Context:\n{context}\n\nQuestion: {question}\n\nPlease provide a helpful answer based on the context above."
    );

    let req = ChatCompletionRequest::new(
        "gpt-3.5-turbo".to_string(),
        vec![
            ChatCompletionMessage {
                role: MessageRole::system,
                content: chat_completion::Content::Text(system_prompt.to_string()),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            },
            ChatCompletionMessage {
                role: MessageRole::user,
                content: chat_completion::Content::Text(user_prompt),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            },
        ],
    );

    let response = client_guard.chat_completion(req).await?;

    if let Some(choice) = response.choices.first() {
        if let Some(content) = &choice.message.content {
            Ok(content.to_string())
        } else {
            Err(anyhow::anyhow!("No content in response"))
        }
    } else {
        Err(anyhow::anyhow!("No choices in response"))
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    // Use simsimd for optimized SIMD-accelerated cosine similarity
    // Note: simsimd cosine returns distance (0 = identical, 2 = opposite)
    // We convert to similarity (1 = identical, -1 = opposite)
    let distance = f32::cosine(a, b).unwrap_or(2.0);
    (1.0 - (distance / 2.0)) as f32
}

// Search knowledge base for multiple questions and return combined context
async fn search_knowledge_base_for_questions(
    questions: &[String],
    embedding_client: &Option<Arc<Mutex<OpenAIClient>>>,
    chunks: &[TextChunk],
) -> anyhow::Result<String> {
    if questions.is_empty() {
        return Ok(String::new());
    }

    let Some(client) = embedding_client else {
        return Ok(String::new());
    };

    let mut all_relevant_chunks = Vec::new();

    for question in questions.iter() {
        // Generate embedding for this question
        let mut client_guard = client.lock().await;
        match generate_embedding(&mut client_guard, question).await {
            Ok(query_embedding) => {
                drop(client_guard); // Release the lock
                let similar_chunks = find_similar_chunks(&query_embedding, chunks, 2).await; // Top 2 per question
                all_relevant_chunks.extend(similar_chunks);
            }
            Err(_e) => {
                // Silently skip failed embeddings
            }
        }
    }

    // Remove duplicates and sort by similarity
    all_relevant_chunks.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    all_relevant_chunks
        .dedup_by(|a, b| a.1.source_file == b.1.source_file && a.1.content == b.1.content);

    // Take top 5 overall
    let top_chunks: Vec<_> = all_relevant_chunks.into_iter().take(5).collect();

    if top_chunks.is_empty() {
        return Ok(String::new());
    }

    // Format as tool context with citation information
    let mut context = String::new();
    for (i, (_similarity, chunk)) in top_chunks.iter().enumerate() {
        let citation_id = i + 1;
        context.push_str(&format!(
            "[Citation {}] From {} (Line {}-{}):\n{}\n\n",
            citation_id,
            chunk.source_file,
            chunk.start_line,
            chunk.end_line,
            chunk.content.trim()
        ));
    }

    Ok(context.trim().to_string())
}

async fn find_similar_chunks<'a>(
    query_embedding: &[f32],
    chunks: &'a [TextChunk],
    top_k: usize,
) -> Vec<(f32, &'a TextChunk)> {
    let mut similarities: Vec<(f32, &TextChunk)> = chunks
        .iter()
        .filter_map(|chunk| {
            chunk.embedding.as_ref().map(|embedding| {
                let similarity = cosine_similarity(query_embedding, embedding);
                (similarity, chunk)
            })
        })
        .collect();

    // Sort by similarity (highest first)
    similarities.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    // Return top k results
    similarities.into_iter().take(top_k).collect()
}

// Helper function to create a boxed body from a string
fn body_from_string(
    s: String,
) -> http_body_util::combinators::BoxBody<Bytes, Box<dyn std::error::Error + Send + Sync>> {
    Full::from(s)
        .map_err(|never| -> Box<dyn std::error::Error + Send + Sync> { match never {} })
        .boxed()
}

// Helper function to create a boxed body from bytes
fn body_from_bytes(
    b: Bytes,
) -> http_body_util::combinators::BoxBody<Bytes, Box<dyn std::error::Error + Send + Sync>> {
    Full::from(b)
        .map_err(|never| -> Box<dyn std::error::Error + Send + Sync> { match never {} })
        .boxed()
}

#[derive(Clone)]
struct StreamServer {
    chunks: Arc<Mutex<Vec<TextChunk>>>,
    embedding_client: Option<Arc<Mutex<OpenAIClient>>>,
    generation_client: Option<Arc<Mutex<OpenAIClient>>>,
    directory: String,
    generation_url: Option<String>,
}

impl Service<Request<Incoming>> for StreamServer {
    type Response = Response<
        http_body_util::combinators::BoxBody<Bytes, Box<dyn std::error::Error + Send + Sync>>,
    >;
    type Error = hyper::Error;
    type Future = BoxFuture<'static, Result<Self::Response, Self::Error>>;

    fn call(&self, req: Request<Incoming>) -> Self::Future {
        let chunks = self.chunks.clone();
        let embedding_client = self.embedding_client.clone();
        let generation_client = self.generation_client.clone();
        let _directory = self.directory.clone();
        let generation_url = self.generation_url.clone();

        async move {
            match (req.method(), req.uri().path()) {
                (&Method::GET, "/") => {
                    let html = r#"
<!DOCTYPE html>
<html>
<head>
    <title>MemKB Chat Interface</title>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; max-width: 1000px; margin: 0 auto; padding: 20px; }
        .panel { border: 1px solid #ddd; border-radius: 8px; padding: 20px; }
        .chat-form { margin-bottom: 20px; }
        textarea { width: 100%; padding: 10px; box-sizing: border-box; min-height: 100px; resize: vertical; }
        button { padding: 10px 20px; margin: 5px 0; background: #007cba; color: white; border: none; border-radius: 4px; cursor: pointer; }
        button:hover { background: #005a87; }
        button:disabled { background: #ccc; cursor: not-allowed; }
        .chat-messages { max-height: 400px; overflow-y: auto; border: 1px solid #eee; padding: 15px; border-radius: 5px; margin-bottom: 15px; }
        .message { margin-bottom: 15px; }
        .message.user { text-align: right; }
        .message.assistant { text-align: left; }
        .message.tool { background: #e8f4fd; padding: 10px; border-radius: 5px; border-left: 4px solid #007cba; margin: 10px 0; }
        .message-content { display: inline-block; padding: 10px; border-radius: 10px; max-width: 80%; word-wrap: break-word; }
        .user .message-content { background: #007cba; color: white; }
        .assistant .message-content { background: #f0f0f0; }
        .message-role { font-size: 0.8em; color: #666; margin-bottom: 5px; }
        .streaming-indicator { color: #007cba; font-style: italic; }
        /* Markdown styles for assistant messages */
        .message-content h1, .message-content h2, .message-content h3 { margin-top: 0; margin-bottom: 10px; }
        .message-content h1 { font-size: 1.2em; }
        .message-content h2 { font-size: 1.1em; }
        .message-content h3 { font-size: 1em; font-weight: bold; }
        .message-content p { margin: 8px 0; }
        .message-content ul, .message-content ol { margin: 8px 0; padding-left: 20px; }
        .message-content li { margin: 4px 0; }
        .message-content code { background: #f4f4f4; padding: 2px 4px; border-radius: 3px; font-family: monospace; font-size: 0.9em; }
        .message-content pre { background: #f4f4f4; padding: 10px; border-radius: 5px; overflow-x: auto; margin: 10px 0; }
        .message-content pre code { background: none; padding: 0; }
        .message-content blockquote { border-left: 4px solid #ddd; margin: 10px 0; padding-left: 15px; color: #666; }
        .message-content strong { font-weight: bold; }
        .message-content em { font-style: italic; }
    </style>
</head>
<body>
    <h1>MemKB</h1>
    
    <div class="panel">
        <div id="chat-messages" class="chat-messages"></div>
        <div class="chat-form">
            <textarea id="chat-input" placeholder="Ask a question about the knowledge base..."></textarea>
            <button onclick="sendMessage()" id="send-btn">Send</button>
            <button onclick="clearChat()">Clear Chat</button>
            <input type="hidden" id="stream-checkbox" checked>
        </div>
    </div>

    <script>
        let chatMessages = [];

        async function sendMessage() {
            const input = document.getElementById('chat-input');
            const sendBtn = document.getElementById('send-btn');
            const message = input.value.trim();
            const isStreaming = document.getElementById('stream-checkbox').checked;
            
            if (!message) return;
            
            // Add user message to chat
            chatMessages.push({ role: 'user', content: message });
            updateChatDisplay();
            
            input.value = '';
            sendBtn.disabled = true;
            
            try {
                const payload = {
                    model: 'gpt-3.5-turbo',
                    messages: chatMessages,
                    stream: isStreaming
                };
                
                const response = await fetch('/v1/chat/completions', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${await response.text()}`);
                }
                
                if (isStreaming) {
                    await handleStreamingResponse(response);
                } else {
                    const data = await response.json();
                    if (data.choices && data.choices[0] && data.choices[0].message) {
                        chatMessages.push({
                            role: 'assistant',
                            content: data.choices[0].message.content
                        });
                        updateChatDisplay();
                    }
                }
            } catch (error) {
                console.error('Chat error:', error);
                chatMessages.push({
                    role: 'assistant',
                    content: `Error: ${error.message}`
                });
                updateChatDisplay();
            } finally {
                sendBtn.disabled = false;
            }
        }
        
        async function handleStreamingResponse(response) {
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let assistantMessage = { role: 'assistant', content: '' };
            chatMessages.push(assistantMessage);
            
            // Add streaming indicator
            const messagesContainer = document.getElementById('chat-messages');
            const streamingDiv = document.createElement('div');
            streamingDiv.className = 'streaming-indicator';
            streamingDiv.textContent = 'Streaming response...';
            messagesContainer.appendChild(streamingDiv);
            
            try {
                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    
                    const chunk = decoder.decode(value);
                    const lines = chunk.split('\n');
                    
                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            const data = line.slice(6);
                            if (data === '[DONE]') continue;
                            
                            try {
                                const parsed = JSON.parse(data);
                                if (parsed.choices && parsed.choices[0] && parsed.choices[0].delta && parsed.choices[0].delta.content) {
                                    assistantMessage.content += parsed.choices[0].delta.content;
                                    updateChatDisplay();
                                }
                            } catch (e) {
                                // Skip invalid JSON chunks
                            }
                        }
                    }
                }
            } finally {
                // Remove streaming indicator
                if (streamingDiv.parentNode) {
                    streamingDiv.parentNode.removeChild(streamingDiv);
                }
                updateChatDisplay();
            }
        }
        
        function updateChatDisplay() {
            const messagesContainer = document.getElementById('chat-messages');
            messagesContainer.innerHTML = '';
            
            chatMessages.forEach(msg => {
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${msg.role}`;
                
                if (msg.role === 'tool') {
                    messageDiv.innerHTML = `
                        <div class="message-role">Knowledge Base Context:</div>
                        <div class="message-content" style="white-space: pre-wrap;">${msg.content}</div>
                    `;
                } else {
                    const roleLabel = msg.role === 'user' ? 'You' : 'Assistant';
                    const content = msg.role === 'assistant' ? marked.parse(msg.content) : msg.content;
                    messageDiv.innerHTML = `
                        <div class="message-role">${roleLabel}:</div>
                        <div class="message-content">${content}</div>
                    `;
                }
                
                messagesContainer.appendChild(messageDiv);
            });
            
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }
        
        function clearChat() {
            chatMessages = [];
            updateChatDisplay();
        }
        
        // Enter key support
        document.getElementById('chat-input').addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
    </script>
</body>
</html>
"#;
                    Ok(Response::builder()
                        .header("content-type", "text/html")
                        .body(body_from_string(html.to_string())).unwrap())
                },
                (&Method::POST, "/v1/chat/completions") => {
                    let body_bytes = match req.into_body().collect().await {
                        Ok(collected) => collected.to_bytes(),
                        Err(e) => {
                            println!("❌ Failed to read request body: {e:?}");
                            return Ok(Response::builder()
                                .status(StatusCode::BAD_REQUEST)
                                .body(body_from_string("Invalid request body".to_string())).unwrap());
                        }
                    };
                    
                    let stream_req: StreamRequest = match serde_json::from_slice::<StreamRequest>(&body_bytes) {
                        Ok(req) => req,
                        Err(e) => {
                            println!("❌ Failed to parse JSON: {e:?}");
                            return Ok(Response::builder()
                                .status(StatusCode::BAD_REQUEST)
                                .body(body_from_string("Invalid JSON".to_string())).unwrap());
                        }
                    };
                    
                    // Extract the user message (latest message from user)
                    let user_message = stream_req.messages
                        .iter()
                        .rev()
                        .find(|msg| msg.role == "user")
                        .map(|msg| msg.content.as_str())
                        .unwrap_or("");
                    
                    if user_message.is_empty() {
                        return Ok(Response::builder()
                            .status(StatusCode::BAD_REQUEST)
                            .body(body_from_string("No user message found".to_string())).unwrap());
                    }
                    
                    // Two-phase approach: Generate search questions, then search knowledge base
                    let mut augmented_messages = stream_req.messages.clone();
                    
                    // Add system prompt at the beginning if not already present
                    let has_system_message = augmented_messages.iter().any(|msg| msg.role == "system");
                    if !has_system_message {
                        let system_prompt = SYSTEM_PROMPT;
                        
                        augmented_messages.insert(0, Message {
                            role: "system".to_string(),
                            content: system_prompt.to_string(),
                        });
                    }
                    
                    // Phase 1: Generate search questions if we have a generation client
                    if let Some(gen_client) = &generation_client {
                        match generate_search_questions(gen_client, &stream_req.messages).await {
                            Ok(questions) => {
                                
                                // Phase 2: Search knowledge base for these questions
                                if !questions.is_empty() {
                                    let chunks_guard = chunks.lock().await;
                                    match search_knowledge_base_for_questions(&questions, &embedding_client, &chunks_guard).await {
                                        Ok(context) => {
                                            if !context.is_empty() {
                                                // Add tool role message with the retrieved context
                                                augmented_messages.push(Message {
                                                    role: "tool".to_string(),
                                                    content: context,
                                                });
                                            }
                                        }
                                        Err(e) => {
                                            println!("❌ Failed to search knowledge base: {e:?}");
                                        }
                                    }
                                }
                            }
                            Err(e) => {
                                println!("❌ Failed to generate search questions: {e:?}");
                            }
                        }
                    }
                    
                    // Use the generation URL from CLI args and ensure it ends with /v1/chat/completions
                    let target_url = match &generation_url {
                        Some(url) => {
                            let mut final_url = url.clone();
                            // Ensure the URL ends with the correct endpoint
                            if !final_url.ends_with("/v1/chat/completions") {
                                if final_url.ends_with("/") {
                                    final_url.push_str("v1/chat/completions");
                                } else if final_url.ends_with("/v1") {
                                    final_url.push_str("/chat/completions");
                                } else {
                                    final_url.push_str("/v1/chat/completions");
                                }
                            }
                            final_url
                        },
                        None => {
                            println!("❌ No generation URL configured");
                            return Ok(Response::builder()
                                .status(StatusCode::BAD_REQUEST)
                                .body(body_from_string("No generation URL configured".to_string())).unwrap());
                        }
                    };
                    
                    // Prepare the request payload
                    let forward_payload = serde_json::json!({
                        "model": stream_req.model.unwrap_or_else(|| "gpt-3.5-turbo".to_string()),
                        "messages": augmented_messages,
                        "stream": stream_req.stream
                    });
                    
                    // Make the HTTP request
                    let client = reqwest::Client::new();
                    match client
                        .post(&target_url)
                        .header("Content-Type", "application/json")
                        .header("Accept", if stream_req.stream { "text/event-stream" } else { "application/json" })
                        .json(&forward_payload)
                        .send()
                        .await
                    {
                        Ok(response) => {
                            let status = response.status();
                            
                            // Convert reqwest status to hyper status
                            let hyper_status = StatusCode::from_u16(status.as_u16())
                                .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                            
                            if stream_req.stream {
                                
                                // Create a stream from the response body that forwards chunks as they arrive
                                use futures::stream::TryStreamExt;
                                let stream = response.bytes_stream()
                                    .map_ok(|bytes| {
                                        Frame::data(bytes)
                                    })
                                    .map_err(|e| {
                                        eprintln!("Stream error: {e:?}");
                                        // Create a hyper error - hyper::Error doesn't have From<io::Error>
                                        // so we need to find another way
                                        Box::new(std::io::Error::other(format!("Stream error: {e}"))) as Box<dyn std::error::Error + Send + Sync>
                                    });
                                
                                let body = BodyExt::boxed(StreamBody::new(stream));
                                
                                Ok(Response::builder()
                                    .status(hyper_status)
                                    .header("content-type", "text/event-stream")
                                    .header("cache-control", "no-cache")
                                    .header("connection", "keep-alive")
                                    .header("access-control-allow-origin", "*")
                                    .body(body)
                                    .unwrap())
                            } else {
                                // Non-streaming: get the full response body
                                let body_bytes = response.bytes().await.unwrap_or_default();
                                
                                Ok(Response::builder()
                                    .status(hyper_status)
                                    .header("content-type", "application/json")
                                    .body(body_from_bytes(body_bytes))
                                    .unwrap())
                            }
                        }
                        Err(e) => {
                            println!("❌ Failed to connect to generation endpoint: {e:?}");
                            Ok(Response::builder()
                                .status(StatusCode::BAD_GATEWAY)
                                .body(body_from_string(format!("Failed to connect to generation endpoint: {e}")))
                                .unwrap())
                        }
                    }
                },
                _ => {
                    Ok(Response::builder()
                        .status(StatusCode::NOT_FOUND)
                        .body(body_from_string("Not Found".to_string())).unwrap())
                }
            }
        }.boxed()
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    println!("🚀 Starting memkb MCP server");
    println!("📍 Server running on: http://{}:{}", args.host, args.port);
    println!("📁 Target directory: {}", args.directory);

    println!("🔍 Scanning directory for .md files...");
    match scan_directory_for_md_files(&args.directory) {
        Ok((md_files, count)) => {
            println!("📄 Found {count} .md files:");
            for (index, file) in md_files.iter().enumerate().take(10) {
                println!("   {}. {}", index + 1, file);
            }
            if count > 10 {
                println!("   ... and {} more files", count - 10);
            }
        }
        Err(e) => {
            println!("❌ Error scanning directory: {e}");
        }
    }

    if let Some(embedding_url) = &args.embedding_url {
        println!("🧠 Embedding server: {embedding_url}");
        println!("🔍 Testing embedding server...");
        if let Err(e) = test_embedding_server(embedding_url).await {
            println!("⚠️  Warning: Embedding server test failed: {e}");
        }
    } else {
        println!("🧠 Embedding server: Not configured");
    }

    if let Some(generation_url) = &args.generation_url {
        println!("✨ Generation server: {generation_url}");
        println!("🔍 Testing generation server...");
        if let Err(e) = test_generation_server(generation_url).await {
            println!("⚠️  Warning: Generation server test failed: {e}");
        }
    } else {
        println!("✨ Generation server: Not configured");
    }

    // Initialize embedding client if URL provided
    let embedding_client = args.embedding_url.as_ref().map(|embedding_url| {
        Arc::new(Mutex::new(
            OpenAIClient::builder()
                .with_endpoint(embedding_url)
                .with_api_key("test")
                .build()
                .unwrap(),
        ))
    });

    // Initialize generation client if URL provided
    let generation_client = args.generation_url.as_ref().map(|generation_url| {
        Arc::new(Mutex::new(
            OpenAIClient::builder()
                .with_endpoint(generation_url)
                .with_api_key("test")
                .build()
                .unwrap(),
        ))
    });

    // Process and embed all markdown files
    println!("🧠 Processing and embedding markdown files...");
    println!(
        "📏 Chunk size: {} characters, Overlap: {} characters",
        args.chunk_size, args.overlap
    );
    let chunks = match chunk_and_embed_files(
        &args.directory,
        &embedding_client,
        args.chunk_size,
        args.overlap,
        args.parallel_embeddings,
    )
    .await
    {
        Ok(chunks) => {
            let embedded_count = chunks
                .iter()
                .filter(|chunk| chunk.embedding.is_some())
                .count();
            println!(
                "✅ Successfully processed {} chunks ({} embedded)",
                chunks.len(),
                embedded_count
            );
            chunks
        }
        Err(e) => {
            println!("❌ Error processing files: {e}");
            Vec::new()
        }
    };

    println!("Press Ctrl+C to stop the server");
    println!();

    let bind_addr = format!("{}:{}", args.host, args.port);
    let listener = tokio::net::TcpListener::bind(&bind_addr).await?;

    // Prepare shared chunks
    let chunks_arc = Arc::new(Mutex::new(chunks));

    // Start MCP server
    let directory = args.directory.clone();
    let mcp_chunks_arc = chunks_arc.clone();
    let embedding_client_arc = embedding_client.clone();
    let generation_client_arc = generation_client.clone();
    let mcp_service = TowerToHyperService::new(StreamableHttpService::new(
        move || {
            let mut kb = MemoryKB::new(
                directory.clone(),
                embedding_client_arc.clone(),
                generation_client_arc.clone(),
            );
            kb.chunks = mcp_chunks_arc.clone();
            Ok(kb)
        },
        LocalSessionManager::default().into(),
        Default::default(),
    ));

    if args.http {
        // Start additional HTTP server on port + 1
        let http_port = args.port + 1;
        let http_bind_addr = format!("{}:{}", args.host, http_port);
        let http_listener = tokio::net::TcpListener::bind(&http_bind_addr).await?;

        println!("🌐 Starting HTTP server at http://{http_bind_addr}");
        println!("   Available endpoints:");
        println!("     GET  / - Web chat interface");
        println!("     POST /v1/chat/completions - Chat completions with knowledge augmentation");

        let stream_service = StreamServer {
            chunks: chunks_arc.clone(),
            embedding_client: embedding_client.clone(),
            generation_client: generation_client.clone(),
            directory: args.directory.clone(),
            generation_url: args.generation_url.clone(),
        };

        // Spawn HTTP server task
        tokio::spawn(async move {
            loop {
                let io = tokio::select! {
                    accept = http_listener.accept() => {
                        TokioIo::new(accept.unwrap().0)
                    }
                };
                let service = stream_service.clone();
                tokio::spawn(async move {
                    let _result = Builder::new(TokioExecutor::default())
                        .serve_connection_with_upgrades(
                            io,
                            hyper::service::service_fn(move |req| service.call(req)),
                        )
                        .await;
                });
            }
        });
    }

    // Run main MCP server
    loop {
        let io = tokio::select! {
            _ = tokio::signal::ctrl_c() => break,
            accept = listener.accept() => {
                TokioIo::new(accept?.0)
            }
        };
        let service = mcp_service.clone();
        tokio::spawn(async move {
            let _result = Builder::new(TokioExecutor::default())
                .serve_connection(io, service)
                .await;
        });
    }
    Ok(())
}
