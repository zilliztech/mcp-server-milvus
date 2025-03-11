# MCP Server for Milvus

> The Model Context Protocol (MCP) is an open protocol that enables seamless integration between LLM applications and external data sources and tools. Whether you're building an AI-powered IDE, enhancing a chat interface, or creating custom AI workflows, MCP provides a standardized way to connect LLMs with the context they need.

This repository contains a MCP server that provides access to [Milvus](https://milvus.io/) vector database functionality.

## Prerequisites

Before using this MCP server, ensure you have:

- Python 3.10 or higher
- A running [Milvus](https://milvus.io/) instance (local or remote)
- [uv](https://github.com/astral-sh/uv) installed (recommended for running the server)

## Usage

The recommended way to use this MCP server is to run it directly with `uv` without installation. This is how both Claude Desktop and Cursor are configured to use it in the examples below.

If you want to clone the repository:

```bash
git clone https://github.com/stephen37/mcp-server-milvus.git
cd mcp-server-milvus
```

Then you can run the server directly:

```bash
uv run src/mcp_server_milvus/server.py --milvus-uri http://localhost:19530
```

## Supported Applications

This MCP server can be used with various LLM applications that support the Model Context Protocol:

- **Claude Desktop**: Anthropic's desktop application for Claude
- **Cursor**: AI-powered code editor with MCP support in its Composer feature
- **Custom MCP clients**: Any application implementing the MCP client specification

## Usage with Claude Desktop

1. Install Claude Desktop from https://claude.ai/download
2. Open your Claude Desktop configuration:
   - macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`

3. Add the following configuration:
```json
{
  "mcpServers": {
    "milvus": {
      "command": "/PATH/TO/uv",
      "args": [
        "--directory",
        "/path/to/mcp-server-milvus/src/mcp_server_milvus",
        "run",
        "server.py",
        "--milvus-uri",
        "http://localhost:19530"
      ]
    }
  }
}
```

4. Restart Claude Desktop

## Usage with Cursor

[Cursor also supports MCP](https://docs.cursor.com/context/model-context-protocol) tools through its Agent feature in Composer. You can add the Milvus MCP server to Cursor in two ways:

### Option 1: Using Cursor Settings UI

1. Go to `Cursor Settings` > `Features` > `MCP`
2. Click on the `+ Add New MCP Server` button
3. Fill out the form:
   - **Type**: Select `stdio` (since you're running a command)
   - **Name**: `milvus`
   - **Command**: `/PATH/TO/uv --directory /path/to/mcp-server-milvus/src/mcp_server_milvus run server.py --milvus-uri http://127.0.0.1:19530`
   
   > ⚠️ Note: Use `127.0.0.1` instead of `localhost` to avoid potential DNS resolution issues.

### Option 2: Using Project-specific Configuration (Recommended)

Create a `.cursor/mcp.json` file in your project root:

1. Create the `.cursor` directory in your project root:
   ```bash
   mkdir -p /path/to/your/project/.cursor
   ```

2. Create a `mcp.json` file with the following content:
   ```json
   {
     "mcpServers": {
       "milvus": {
         "command": "/PATH/TO/uv",
         "args": [
           "--directory",
           "/path/to/mcp-server-milvus/src/mcp_server_milvus",
           "run",
           "server.py",
           "--milvus-uri",
           "http://127.0.0.1:19530"
         ]
       }
     }
   }
   ```

3. Restart Cursor or reload the window

After adding the server, you may need to press the refresh button in the MCP settings to populate the tool list. The Composer Agent will automatically use the Milvus tools when relevant to your queries.

### Verifying the Integration

To verify that Cursor has successfully integrated with your Milvus MCP server:

1. Open Cursor Settings > Features > MCP
2. Check that "Milvus" appears in the list of MCP servers
3. Verify that the tools are listed (e.g., milvus_list_collections, milvus_vector_search, etc.)
4. If the server is enabled but shows an error, check the Troubleshooting section below

## Available Tools

The server provides the following tools:

### Search and Query Operations

- `milvus-text-search`: Search for documents using full text search
  - Parameters:
    - `collection_name`: Name of collection to search
    - `query_text`: Text to search for
    - `limit`: Maximum results (default: 5)
    - `output_fields`: Fields to include in results
    - `drop_ratio`: Proportion of low-frequency terms to ignore (0.0-1.0)

- `milvus-vector-search`: Perform vector similarity search on a collection
  - Parameters:
    - `collection_name`: Name of collection to search
    - `vector`: Query vector
    - `vector_field`: Field containing vectors to search (default: "vector")
    - `limit`: Maximum results (default: 5)
    - `output_fields`: Fields to include in results
    - `metric_type`: Distance metric (COSINE, L2, IP) (default: "COSINE")
    - `filter_expr`: Optional filter expression

- `milvus-hybrid-search`: Perform hybrid search combining vector similarity and attribute filtering
  - Parameters:
    - `collection_name`: Name of collection to search
    - `vector`: Query vector
    - `vector_field`: Field containing vectors to search (default: "vector")
    - `filter_expr`: Filter expression for metadata
    - `limit`: Maximum results (default: 5)
    - `output_fields`: Fields to include in results
    - `metric_type`: Distance metric (COSINE, L2, IP) (default: "COSINE")

- `milvus-multi-vector-search`: Perform vector similarity search with multiple query vectors
  - Parameters:
    - `collection_name`: Name of collection to search
    - `vectors`: List of query vectors
    - `vector_field`: Field containing vectors to search (default: "vector")
    - `limit`: Maximum results per query (default: 5)
    - `output_fields`: Fields to include in results
    - `metric_type`: Distance metric (COSINE, L2, IP) (default: "COSINE")
    - `filter_expr`: Optional filter expression

- `milvus-query`: Query collection using filter expressions
  - Parameters:
    - `collection_name`: Name of collection to query
    - `filter_expr`: Filter expression (e.g. 'age > 20')
    - `output_fields`: Fields to include in results
    - `limit`: Maximum results (default: 10)

- `milvus-count`: Count entities in a collection
  - Parameters:
    - `collection_name`: Name of the collection
    - `filter_expr`: Optional filter expression

### Collection Management

- `milvus-list-collections`: List all collections in the database

- `milvus-collection-info`: Get detailed information about a collection
  - Parameters:
    - `collection_name`: Name of the collection

- `milvus-get-collection-stats`: Get statistics about a collection
  - Parameters:
    - `collection_name`: Name of collection

- `milvus-create-collection`: Create a new collection with specified schema
  - Parameters:
    - `collection_name`: Name for the new collection
    - `schema`: Collection schema definition
    - `index_params`: Optional index parameters

- `milvus-load-collection`: Load a collection into memory for search and query
  - Parameters:
    - `collection_name`: Name of collection to load
    - `replica_number`: Number of replicas (default: 1)

- `milvus-release-collection`: Release a collection from memory
  - Parameters:
    - `collection_name`: Name of collection to release

- `milvus-get-query-segment-info`: Get information about query segments
  - Parameters:
    - `collection_name`: Name of collection

- `milvus-get-collection-loading-progress`: Get the loading progress of a collection
  - Parameters:
    - `collection_name`: Name of collection

### Data Operations

- `milvus-insert-data`: Insert data into a collection
  - Parameters:
    - `collection_name`: Name of collection
    - `data`: Dictionary mapping field names to lists of values

- `milvus-bulk-insert`: Insert data in batches for better performance
  - Parameters:
    - `collection_name`: Name of collection
    - `data`: Dictionary mapping field names to lists of values
    - `batch_size`: Number of records per batch (default: 1000)

- `milvus-upsert-data`: Upsert data into a collection (insert or update if exists)
  - Parameters:
    - `collection_name`: Name of collection
    - `data`: Dictionary mapping field names to lists of values

- `milvus-delete-entities`: Delete entities from a collection based on filter expression
  - Parameters:
    - `collection_name`: Name of collection
    - `filter_expr`: Filter expression to select entities to delete

- `milvus-create-dynamic-field`: Add a dynamic field to an existing collection
  - Parameters:
    - `collection_name`: Name of collection
    - `field_name`: Name of the new field
    - `data_type`: Data type of the field
    - `description`: Optional description

### Index Management

- `milvus-create-index`: Create an index on a vector field
  - Parameters:
    - `collection_name`: Name of collection
    - `field_name`: Field to index
    - `index_type`: Type of index (IVF_FLAT, HNSW, etc.) (default: "IVF_FLAT")
    - `metric_type`: Distance metric (COSINE, L2, IP) (default: "COSINE")
    - `params`: Additional index parameters

- `milvus-get-index-info`: Get information about indexes in a collection
  - Parameters:
    - `collection_name`: Name of collection
    - `field_name`: Optional specific field to get index info for

## Environment Variables

- `MILVUS_URI`: Milvus server URI (can be set instead of --milvus-uri)
- `MILVUS_TOKEN`: Optional authentication token
- `MILVUS_DB`: Database name (defaults to "default")

## Development

To run the server directly:

```bash
uv run server.py --milvus-uri http://localhost:19530
```

## Examples

### Using Claude Desktop 

#### Example 1: Listing Collections

```
What are the collections I have in my Milvus DB?
```
Claude will then use MCP to check this information on our Milvus DB. 
```
I'll check what collections are available in your Milvus database.

> View result from milvus-list-collections from milvus (local)

Here are the collections in your Milvus database:

1. rag_demo
2. test
3. chat_messages
4. text_collection
5. image_collection
6. customized_setup
7. streaming_rag_demo
```

#### Example 2: Searching for Documents

```
Find documents in my text_collection that mention "machine learning"
```

Claude will use the full-text search capabilities of Milvus to find relevant documents:

```
I'll search for documents about machine learning in your text_collection.

> View result from milvus-text-search from milvus (local)

Here are the documents I found that mention machine learning:
[Results will appear here based on your actual data]
```

### Using Cursor

#### Example: Creating a Collection

In Cursor's Composer, you can ask:

```
Create a new collection called 'articles' in Milvus with fields for title (string), content (string), and a vector field (128 dimensions)
```

Cursor will use the MCP server to execute this operation:

```
I'll create a new collection called 'articles' with the specified fields.

> View result from milvus-create-collection from milvus (local)

Collection 'articles' has been created successfully with the following schema:
- title: string
- content: string
- vector: float vector[128]
```

## Troubleshooting

### Common Issues

#### Connection Errors

If you see errors like "Failed to connect to Milvus server":

1. Verify your Milvus instance is running: `docker ps` (if using Docker)
2. Check the URI is correct in your configuration
3. Ensure there are no firewall rules blocking the connection
4. Try using `127.0.0.1` instead of `localhost` in the URI

#### Authentication Issues

If you see authentication errors:

1. Verify your `MILVUS_TOKEN` is correct
2. Check if your Milvus instance requires authentication
3. Ensure you have the correct permissions for the operations you're trying to perform

#### Tool Not Found

If the MCP tools don't appear in Claude Desktop or Cursor:

1. Restart the application
2. Check the server logs for any errors
3. Verify the MCP server is running correctly
4. Press the refresh button in the MCP settings (for Cursor)

### Getting Help

If you continue to experience issues:

1. Check the [GitHub Issues](https://github.com/stephen37/mcp-server-milvus/issues) for similar problems
2. Join the [Zilliz Community Discord](https://discord.gg/zilliz) for support
3. File a new issue with detailed information about your problem
