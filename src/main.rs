
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
use std::fs;
use walkdir::WalkDir;
use rmcp::handler::server::tool::Parameters;


#[derive(Parser)]
#[command(name = "memkb")]
#[command(about = "Memory knowledge base MCP server")]
struct Args {
    #[arg(short, long, default_value = "8080")]
    port: u16,
    
    #[arg(short, long, default_value = ".")]
    directory: String,
    
    #[arg(short, long)]
    embedding_url: Option<String>,
    
    #[arg(short, long)]
    generation_url: Option<String>,
}

#[derive(Serialize, Deserialize, JsonSchema)]
struct AskRequest {
    prompt: String,
}

#[derive(Clone)]
pub struct MemoryKB {
    directory: String,
    tool_router: ToolRouter<Self>,
}

#[tool_router]
impl MemoryKB {
    fn new(directory: String) -> Self {
        Self {
            directory,
            tool_router: Self::tool_router(),
        }
    }

    #[tool(name = "ask", description = "Ask a question and get relevant content from all markdown files")]
    async fn ask(&self, params: Parameters<AskRequest>) -> Result<CallToolResult, McpError> {
        let _prompt = &params.0.prompt; // Access the prompt parameter
        
        match read_and_merge_markdown_files(&self.directory) {
            Ok(merged_content) => {
                Ok(CallToolResult::success(vec![rmcp::model::Content::text(
                    merged_content,
                )]))
            }
            Err(e) => {
                Ok(CallToolResult::success(vec![rmcp::model::Content::text(
                    format!("Error reading markdown files: {}", e),
                )]))
            }
        }
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

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    
    println!("🚀 Starting memkb MCP server");
    println!("📍 Server running on: http://[::1]:{}", args.port);
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
    
    println!("Press Ctrl+C to stop the server");
    println!();
    
    let directory = args.directory.clone();
    let service = TowerToHyperService::new(StreamableHttpService::new(
        move || Ok(MemoryKB::new(directory.clone())),
        LocalSessionManager::default().into(),
        Default::default(),
    ));
    let bind_addr = format!("[::1]:{}", args.port);
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