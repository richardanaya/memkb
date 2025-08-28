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
                        format!("Error reading markdown files: {}", e),
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
                                "Error generating answer: {}. Returning raw content:\n\n{}",
                                e, merged_content
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
                match generate_embedding(&mut *client_guard, prompt).await {
                    Ok(embedding) => embedding,
                    Err(e) => {
                        return Ok(CallToolResult::success(vec![rmcp::model::Content::text(
                            format!("Error generating query embedding: {}", e),
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
                "Context {} (Similarity: {:.3}) - From: {}\n{}\n\n",
                i + 1,
                similarity,
                chunk.source_file,
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
                        "Error generating answer: {}. Here are the relevant chunks:\n\n{}",
                        e, context
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
            println!("❌ Embedding server test: FAILED - {}", e);
            Err(anyhow::anyhow!("Embedding server test failed: {}", e))
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
            println!("❌ Generation server test: FAILED - {}", e);
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

fn chunk_text(text: &str, chunk_size: usize, overlap: usize) -> Vec<String> {
    // Create a text splitter with specified chunk size and overlap
    let cfg = ChunkConfig::new(chunk_size).with_overlap(overlap).unwrap();
    let splitter = MarkdownSplitter::new(cfg);

    // For now, just use basic splitting. We can enhance with overlap later
    splitter.chunks(text).map(|s| s.to_string()).collect()
}

async fn chunk_and_embed_files(
    dir_path: &str,
    embedding_client: &Option<Arc<Mutex<OpenAIClient>>>,
    chunk_size: usize,
    overlap: usize,
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
                    let text_chunks = chunk_text(&content, chunk_size, overlap);

                    for chunk_content in text_chunks {
                        let chunk = TextChunk {
                            content: chunk_content,
                            source_file: source_file.clone(),
                            embedding: None,
                        };
                        chunks.push(chunk);
                    }
                }
            }
        }
    }

    // Second pass: generate embeddings with progress counter
    if let Some(client) = embedding_client {
        let total_chunks = chunks.len();
        println!("📊 Generating embeddings for {} chunks...", total_chunks);

        for (i, chunk) in chunks.iter_mut().enumerate() {
            let progress = i + 1;
            print!(
                "\r🔄 Embedding chunk {}/{} ({:.1}%)",
                progress,
                total_chunks,
                (progress as f32 / total_chunks as f32) * 100.0
            );
            std::io::stdout().flush().unwrap();

            let mut client_guard = client.lock().await;
            match generate_embedding(&mut *client_guard, &chunk.content).await {
                Ok(embedding) => {
                    chunk.embedding = Some(embedding);
                }
                Err(e) => {
                    println!(
                        "\n⚠️  Failed to embed chunk from {}: {}",
                        chunk.source_file, e
                    );
                }
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

async fn generate_answer(
    client: &Arc<Mutex<OpenAIClient>>,
    question: &str,
    context: &str,
) -> anyhow::Result<String> {
    let mut client_guard = client.lock().await;

    let system_prompt = "You are a helpful assistant that answers questions based on the provided context. If the context doesn't contain relevant information to answer the question, respond with \"We don't have any information on that question\". Always base your answer strictly on the provided context and be concise.";

    let user_prompt = format!(
        "Context:\n{}\n\nQuestion: {}\n\nPlease provide a helpful answer based on the context above.",
        context, question
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
        let _generation_client = self.generation_client.clone();
        let _directory = self.directory.clone();
        let generation_url = self.generation_url.clone();

        async move {
            match (req.method(), req.uri().path()) {
                (&Method::GET, "/") => {
                    let html = r#"
<!DOCTYPE html>
<html>
<head>
    <title>MemKB Test Interface</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .search-form { margin-bottom: 20px; }
        input[type="text"] { width: 70%; padding: 10px; }
        button { padding: 10px 20px; }
        .results { background: #f5f5f5; padding: 15px; border-radius: 5px; }
        .chunk { margin-bottom: 15px; padding: 10px; background: white; border-radius: 3px; }
        .chunk-meta { color: #666; font-size: 0.9em; margin-bottom: 5px; }
    </style>
</head>
<body>
    <h1>MemKB Knowledge Base Search</h1>
    <div class="search-form">
        <input type="text" id="query" placeholder="Enter your search query..." />
        <button onclick="search()">Search</button>
    </div>
    <div id="results"></div>

    <script>
        async function search() {
            const query = document.getElementById('query').value;
            const resultsDiv = document.getElementById('results');
            
            if (!query.trim()) {
                resultsDiv.innerHTML = '<p>Please enter a search query.</p>';
                return;
            }
            
            resultsDiv.innerHTML = '<p>Searching...</p>';
            
            try {
                const response = await fetch('/search', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query: query })
                });
                
                const data = await response.json();
                
                if (data.results && data.results.length > 0) {
                    let html = '<div class="results">';
                    data.results.forEach((result, i) => {
                        html += `<div class="chunk">
                            <div class="chunk-meta">Chunk ${i + 1} - Similarity: ${result.similarity.toFixed(3)} - From: ${result.source_file}</div>
                            <div>${result.content}</div>
                        </div>`;
                    });
                    html += '</div>';
                    resultsDiv.innerHTML = html;
                } else {
                    resultsDiv.innerHTML = '<div class="results"><p>No results found.</p></div>';
                }
            } catch (error) {
                resultsDiv.innerHTML = `<div class="results"><p>Error: ${error.message}</p></div>`;
            }
        }
        
        document.getElementById('query').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                search();
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
                (&Method::POST, "/search") => {
                    let body_bytes = match req.into_body().collect().await {
                        Ok(collected) => collected.to_bytes(),
                        Err(_) => {
                            return Ok(Response::builder()
                                .status(StatusCode::BAD_REQUEST)
                                .body(body_from_string("Invalid request body".to_string())).unwrap());
                        }
                    };
                    
                    let search_req: serde_json::Value = match serde_json::from_slice(&body_bytes) {
                        Ok(req) => req,
                        Err(_) => {
                            return Ok(Response::builder()
                                .status(StatusCode::BAD_REQUEST)
                                .body(body_from_string("Invalid JSON".to_string())).unwrap());
                        }
                    };
                    
                    let query = match search_req.get("query").and_then(|q| q.as_str()) {
                        Some(q) => q,
                        None => {
                            return Ok(Response::builder()
                                .status(StatusCode::BAD_REQUEST)
                                .body(body_from_string("Missing query field".to_string())).unwrap());
                        }
                    };
                    
                    let results = if let Some(client) = &embedding_client {
                        // Use embedding search
                        match {
                            let mut client_guard = client.lock().await;
                            generate_embedding(&mut *client_guard, query).await
                        } {
                            Ok(query_embedding) => {
                                let chunks_guard = chunks.lock().await;
                                let similar_chunks = find_similar_chunks(&query_embedding, &chunks_guard, 5).await;
                                
                                let mut results = Vec::new();
                                for (similarity, chunk) in similar_chunks {
                                    results.push(serde_json::json!({
                                        "similarity": similarity,
                                        "content": chunk.content,
                                        "source_file": chunk.source_file
                                    }));
                                }
                                results
                            }
                            Err(_) => Vec::new(),
                        }
                    } else {
                        // Fallback to simple text matching
                        let chunks_guard = chunks.lock().await;
                        let mut results = Vec::new();
                        let query_lower = query.to_lowercase();
                        
                        for chunk in chunks_guard.iter() {
                            if chunk.content.to_lowercase().contains(&query_lower) {
                                results.push(serde_json::json!({
                                    "similarity": 1.0,
                                    "content": chunk.content,
                                    "source_file": chunk.source_file
                                }));
                                if results.len() >= 5 { break; }
                            }
                        }
                        results
                    };
                    
                    let response_json = serde_json::json!({ "results": results });
                    
                    Ok(Response::builder()
                        .header("content-type", "application/json")
                        .body(body_from_string(response_json.to_string())).unwrap())
                },
                (&Method::POST, "/v1/chat/completions") => {
                    println!("📡 Received POST request to /api/stream");
                    
                    let body_bytes = match req.into_body().collect().await {
                        Ok(collected) => collected.to_bytes(),
                        Err(e) => {
                            println!("❌ Failed to read request body: {:?}", e);
                            return Ok(Response::builder()
                                .status(StatusCode::BAD_REQUEST)
                                .body(body_from_string("Invalid request body".to_string())).unwrap());
                        }
                    };
                    
                    println!("📄 Request body size: {} bytes", body_bytes.len());
                    
                    let stream_req: StreamRequest = match serde_json::from_slice::<StreamRequest>(&body_bytes) {
                        Ok(req) => {
                            println!("✅ Successfully parsed StreamRequest");
                            println!("💬 Message count: {}", req.messages.len());
                            req
                        },
                        Err(e) => {
                            println!("❌ Failed to parse JSON: {:?}", e);
                            println!("📄 Raw body: {}", String::from_utf8_lossy(&body_bytes));
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
                    
                    println!("🔍 Extracted user message: '{}'", user_message);
                    
                    if user_message.is_empty() {
                        println!("❌ No user message found in request");
                        return Ok(Response::builder()
                            .status(StatusCode::BAD_REQUEST)
                            .body(body_from_string("No user message found".to_string())).unwrap());
                    }
                    
                    // Get relevant context from knowledge base
                    println!("🧠 Searching knowledge base for context...");
                    let context = if let Some(client) = &embedding_client {
                        println!("🔤 Generating embedding for user message...");
                        match {
                            let mut client_guard = client.lock().await;
                            generate_embedding(&mut *client_guard, user_message).await
                        } {
                            Ok(query_embedding) => {
                                println!("✅ Embedding generated, searching for similar chunks...");
                                let chunks_guard = chunks.lock().await;
                                let similar_chunks = find_similar_chunks(&query_embedding, &chunks_guard, 3).await;
                                println!("📚 Found {} similar chunks", similar_chunks.len());
                                
                                let mut context_str = String::new();
                                for (i, (similarity, chunk)) in similar_chunks.iter().enumerate() {
                                    println!("  {}. Similarity: {:.3}, Source: {}", i + 1, similarity, chunk.source_file);
                                    context_str.push_str(&format!(
                                        "Context (Similarity: {:.3}) - From: {}\n{}\n\n",
                                        similarity,
                                        chunk.source_file,
                                        chunk.content
                                    ));
                                }
                                context_str
                            }
                            Err(e) => {
                                println!("❌ Failed to generate embedding: {:?}", e);
                                String::new()
                            },
                        }
                    } else {
                        println!("⚠️  No embedding client available, skipping context search");
                        String::new()
                    };
                    
                    // Augment the system message or last message with context if available
                    let mut augmented_messages = stream_req.messages.clone();
                    if !context.is_empty() {
                        println!("📝 Augmenting messages with knowledge base context");
                        // Find system message and augment it, or create one
                        let system_augmentation = format!("\n\nRelevant context from knowledge base:\n{}", context);
                        
                        if let Some(system_msg) = augmented_messages.iter_mut().find(|msg| msg.role == "system") {
                            println!("🔧 Augmenting existing system message");
                            system_msg.content.push_str(&system_augmentation);
                        } else {
                            println!("➕ Creating new system message with context");
                            // Insert system message at the beginning
                            augmented_messages.insert(0, Message {
                                role: "system".to_string(),
                                content: format!("You are a helpful assistant. Use the following context to help answer questions:{}", system_augmentation),
                            });
                        }
                    } else {
                        println!("⚠️  No context found, proceeding without augmentation");
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
                    
                    println!("🚀 Proxying to generation endpoint: {}", target_url);
                    
                    // Prepare the request payload
                    let forward_payload = serde_json::json!({
                        "model": stream_req.model.unwrap_or_else(|| "gpt-3.5-turbo".to_string()),
                        "messages": augmented_messages,
                        "stream": stream_req.stream
                    });
                    
                    println!("📤 Sending {} request with {} messages", 
                             if stream_req.stream { "streaming" } else { "non-streaming" },
                             augmented_messages.len());
                    
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
                            println!("📥 Received response with status: {}", status);
                            
                            // Convert reqwest status to hyper status
                            let hyper_status = StatusCode::from_u16(status.as_u16())
                                .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                            
                            if stream_req.stream {
                                println!("🌊 Setting up TRUE streaming response");
                                
                                // Create a stream from the response body that forwards chunks as they arrive
                                use futures::stream::TryStreamExt;
                                let stream = response.bytes_stream()
                                    .map_ok(|bytes| {
                                        Frame::data(bytes)
                                    })
                                    .map_err(|e| {
                                        eprintln!("Stream error: {:?}", e);
                                        // Create a hyper error - hyper::Error doesn't have From<io::Error>
                                        // so we need to find another way
                                        Box::new(std::io::Error::new(std::io::ErrorKind::Other, format!("Stream error: {}", e))) as Box<dyn std::error::Error + Send + Sync>
                                    });
                                
                                let body = BodyExt::boxed(StreamBody::new(stream));
                                
                                println!("✅ Starting TRUE streaming response");
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
                                println!("📄 Response body size: {} bytes", body_bytes.len());
                                
                                println!("✅ Forwarding non-streaming response");
                                Ok(Response::builder()
                                    .status(hyper_status)
                                    .header("content-type", "application/json")
                                    .body(body_from_bytes(body_bytes))
                                    .unwrap())
                            }
                        }
                        Err(e) => {
                            println!("❌ Failed to connect to generation endpoint: {:?}", e);
                            Ok(Response::builder()
                                .status(StatusCode::BAD_GATEWAY)
                                .body(body_from_string(format!("Failed to connect to generation endpoint: {}", e)))
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
            println!("📄 Found {} .md files:", count);
            for (index, file) in md_files.iter().enumerate().take(10) {
                println!("   {}. {}", index + 1, file);
            }
            if count > 10 {
                println!("   ... and {} more files", count - 10);
            }
        }
        Err(e) => {
            println!("❌ Error scanning directory: {}", e);
        }
    }

    if let Some(embedding_url) = &args.embedding_url {
        println!("🧠 Embedding server: {}", embedding_url);
        println!("🔍 Testing embedding server...");
        if let Err(e) = test_embedding_server(embedding_url).await {
            println!("⚠️  Warning: Embedding server test failed: {}", e);
        }
    } else {
        println!("🧠 Embedding server: Not configured");
    }

    if let Some(generation_url) = &args.generation_url {
        println!("✨ Generation server: {}", generation_url);
        println!("🔍 Testing generation server...");
        if let Err(e) = test_generation_server(generation_url).await {
            println!("⚠️  Warning: Generation server test failed: {}", e);
        }
    } else {
        println!("✨ Generation server: Not configured");
    }

    // Initialize embedding client if URL provided
    let embedding_client = if let Some(embedding_url) = &args.embedding_url {
        Some(Arc::new(Mutex::new(
            OpenAIClient::builder()
                .with_endpoint(embedding_url)
                .with_api_key("test")
                .build()
                .unwrap(),
        )))
    } else {
        None
    };

    // Initialize generation client if URL provided
    let generation_client = if let Some(generation_url) = &args.generation_url {
        Some(Arc::new(Mutex::new(
            OpenAIClient::builder()
                .with_endpoint(generation_url)
                .with_api_key("test")
                .build()
                .unwrap(),
        )))
    } else {
        None
    };

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
            println!("❌ Error processing files: {}", e);
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

        println!("🌐 Starting HTTP server at http://{}", http_bind_addr);
        println!("   Available endpoints:");
        println!("     GET  / - Search interface");
        println!("     POST /search - Search knowledge base");
        println!("     POST /api/stream - Streaming endpoint with knowledge augmentation");

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
