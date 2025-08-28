# MemKB - Memory Knowledge Base

A Rust-based MCP (Model Context Protocol) server that provides AI-powered semantic search over markdown documentation using local LLMs all in memory!

## Features

- **Semantic Search**: Uses embeddings for intelligent content retrieval
- **AI-Generated Answers**: Synthesizes information from multiple sources into coherent responses

## Quick Start

### Basic Usage 
```bash
# With embedding and generation servers
memkb --directory ./docs -e http://127.0.0.1:9095/v1 -g http://127.0.0.1:9091/v1
```

## Command Line Options

```
-p, --port <PORT>                      MCP server port [default: 8080]
-H, --host <HOST>                      Server host [default: localhost]
-d, --directory <DIRECTORY>            Directory with .md files [default: .]
-e, --embedding-url <URL>              Embedding server endpoint
-g, --generation-url <URL>             Generation server endpoint  
-c, --chunk-size <SIZE>                Chunk size in characters [default: 1000]
-o, --overlap <OVERLAP>                Chunk overlap in characters [default: 200]
    --test                             Enable web test interface on port+1
```

## How It Works

1. **Indexing**: Scans directory for `.md` files and chunks them
2. **Embedding**: Generates embeddings for each chunk (if embedding server available)
3. **Query**: When asked a question:
   - Generates query embedding
   - Finds most similar chunks
   - Uses generation server to create contextual answer
4. **Response**: Returns AI-generated answer or falls back to raw chunks

## Web Test Interface

When using `--test`, a web interface is available at `http://localhost:8081` (port+1) for easy testing.

## Key Libraries

- **rmcp**: Rust MCP (Model Context Protocol) framework for building AI tool servers
- **simsimd**: SIMD-accelerated similarity calculations for fast embedding comparisons
- **text-splitter**: Intelligent markdown-aware text chunking with configurable overlap

## MCP Integration

The server exposes an `ask` tool that can be used by MCP-compatible clients like Claude Desktop.
