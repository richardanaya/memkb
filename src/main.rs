
use clap::Parser;
use hyper_util::{
    rt::{TokioExecutor, TokioIo},
    server::conn::auto::Builder,
    service::TowerToHyperService,
};
use rmcp::transport::streamable_http_server::{
    StreamableHttpService, session::local::LocalSessionManager,
};
use openai_api_rs::v1::{api::OpenAIClient, chat_completion::{self, ChatCompletionMessage, ChatCompletionRequest, MessageRole}, embedding::{EmbeddingRequest, EncodingFormat}};
use rmcp::{ErrorData as McpError, model::*, tool, tool_router,tool_handler, handler::server::router::tool::ToolRouter};
use serde::{Deserialize, Serialize};
use schemars::JsonSchema;
use simsimd::SpatialSimilarity;
use std::{fs, io::Write, sync::Arc};
use text_splitter::{ChunkConfig, MarkdownSplitter};
use tokio::sync::Mutex;
use walkdir::WalkDir;
use rmcp::handler::server::tool::Parameters;


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
}

#[derive(Serialize, Deserialize, JsonSchema)]
struct AskRequest {
    prompt: String,
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
    tool_router: ToolRouter<Self>,
}

#[tool_router]
impl MemoryKB {
    fn new(directory: String, embedding_client: Option<Arc<Mutex<OpenAIClient>>>) -> Self {
        Self {
            directory,
            chunks: Arc::new(Mutex::new(Vec::new())),
            embedding_client,
            tool_router: Self::tool_router(),
        }
    }

    #[tool(name = "ask", description = "Ask a question and get relevant content from all markdown files")]
    async fn ask(&self, params: Parameters<AskRequest>) -> Result<CallToolResult, McpError> {
        let prompt = &params.0.prompt;
        
        // If no embedding client, fall back to returning all content
        if self.embedding_client.is_none() {
            match read_and_merge_markdown_files(&self.directory) {
                Ok(merged_content) => {
                    return Ok(CallToolResult::success(vec![rmcp::model::Content::text(
                        merged_content,
                    )]));
                }
                Err(e) => {
                    return Ok(CallToolResult::success(vec![rmcp::model::Content::text(
                        format!("Error reading markdown files: {}", e),
                    )]));
                }
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
            },
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
        
        // Format response with top similar chunks
        let mut response = format!("Top {} relevant chunks for your query:\n\n", similar_chunks.len());
        
        for (i, (similarity, chunk)) in similar_chunks.iter().enumerate() {
            response.push_str(&format!(
                "**Chunk {} (Similarity: {:.3})** - From: {}\n{}\n\n---\n\n",
                i + 1,
                similarity,
                chunk.source_file,
                chunk.content
            ));
        }
        
        Ok(CallToolResult::success(vec![rmcp::model::Content::text(
            response,
        )]))
    }
}

// Implement the server handler
#[tool_handler]
impl rmcp::ServerHandler for MemoryKB {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            instructions: Some("Memory knowledge base server - ask questions about markdown content".into()),
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            ..Default::default()
        }
    }
}

async fn test_embedding_server(url: &str) -> anyhow::Result<()> {
    let mut client = OpenAIClient::builder()
    .with_endpoint(url)
    .with_api_key("test")
    .build().unwrap();
    
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
    .build().unwrap();
    
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
            print!("\r🔄 Embedding chunk {}/{} ({:.1}%)", 
                   progress, total_chunks, 
                   (progress as f32 / total_chunks as f32) * 100.0);
            std::io::stdout().flush().unwrap();
            
            let mut client_guard = client.lock().await;
            match generate_embedding(&mut *client_guard, &chunk.content).await {
                Ok(embedding) => {
                    chunk.embedding = Some(embedding);
                }
                Err(e) => {
                    println!("\n⚠️  Failed to embed chunk from {}: {}", chunk.source_file, e);
                }
            }
        }
        println!(); // New line after progress counter
    }
    
    Ok(chunks)
}

async fn generate_embedding(client: &mut OpenAIClient, text: &str) -> anyhow::Result<Vec<f32>> {
    let mut req = EmbeddingRequest::new(
        "text-embedding-3-small".to_string(),
        vec![text.to_string()],
    );
    req.encoding_format = Some(EncodingFormat::Float);
    
    let response = client.embedding(req).await?;
    
    if let Some(embedding_data) = response.data.first() {
        Ok(embedding_data.embedding.clone())
    } else {
        Err(anyhow::anyhow!("No embedding data returned"))
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
        Some(Arc::new(Mutex::new(OpenAIClient::builder()
            .with_endpoint(embedding_url)
            .with_api_key("test")
            .build().unwrap())))
    } else {
        None
    };
    
    // Process and embed all markdown files
    println!("🧠 Processing and embedding markdown files...");
    println!("📏 Chunk size: {} characters, Overlap: {} characters", args.chunk_size, args.overlap);
    let chunks = match chunk_and_embed_files(&args.directory, &embedding_client, args.chunk_size, args.overlap).await {
        Ok(chunks) => {
            let embedded_count = chunks.iter().filter(|chunk| chunk.embedding.is_some()).count();
            println!("✅ Successfully processed {} chunks ({} embedded)", chunks.len(), embedded_count);
            chunks
        }
        Err(e) => {
            println!("❌ Error processing files: {}", e);
            Vec::new()
        }
    };
    
    println!("Press Ctrl+C to stop the server");
    println!();
    
    let directory = args.directory.clone();
    let chunks_arc = Arc::new(Mutex::new(chunks));
    let embedding_client_arc = embedding_client.clone();
    let service = TowerToHyperService::new(StreamableHttpService::new(
        move || {
            let mut kb = MemoryKB::new(directory.clone(), embedding_client_arc.clone());
            kb.chunks = chunks_arc.clone();
            Ok(kb)
        },
        LocalSessionManager::default().into(),
        Default::default(),
    ));
    let bind_addr = format!("{}:{}", args.host, args.port);
    let listener = tokio::net::TcpListener::bind(&bind_addr).await?;
    loop {
        let io = tokio::select! {
            _ = tokio::signal::ctrl_c() => break,
            accept = listener.accept() => {
                TokioIo::new(accept?.0)
            }
        };
        let service = service.clone();
        tokio::spawn(async move {
            let _result = Builder::new(TokioExecutor::default())
                .serve_connection(io, service)
                .await;
        });
    }
    Ok(())
}