
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
use std::sync::Arc;
use tokio::sync::Mutex;

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

#[derive(Clone)]
pub struct Counter {
    counter: Arc<Mutex<i32>>,
    tool_router: ToolRouter<Self>,
}

#[tool_router]
impl Counter {
    fn new() -> Self {
        Self {
            counter: Arc::new(Mutex::new(0)),
            tool_router: Self::tool_router(),
        }
    }

    #[tool(description = "Increment the counter by 1")]
    async fn increment(&self) -> Result<CallToolResult, McpError> {
        let mut counter = self.counter.lock().await;
        *counter += 1;
        Ok(CallToolResult::success(vec![rmcp::model::Content::text(
            counter.to_string(),
        )]))
    }

    #[tool(description = "Get the current counter value")]
    async fn get(&self) -> Result<CallToolResult, McpError> {
        let counter = self.counter.lock().await;
        Ok(CallToolResult::success(vec![rmcp::model::Content::text(
            counter.to_string(),
        )]))
    }
}

// Implement the server handler
#[tool_handler]
impl rmcp::ServerHandler for Counter {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            instructions: Some("A simple calculator".into()),
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

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    
    println!("🚀 Starting memkb MCP server");
    println!("📍 Server running on: http://[::1]:{}", args.port);
    println!("📁 Target directory: {}", args.directory);
    
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
    
    let service = TowerToHyperService::new(StreamableHttpService::new(
        || Ok(Counter::new()),
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