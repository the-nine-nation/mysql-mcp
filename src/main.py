import os
import sys
import logging
import json
import asyncio
import traceback
from typing import Dict, Any, List, Optional, TypedDict, Union
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
from typing import AsyncIterator

import mcp.types as types
from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions

# Import database drivers
try:
    # Use aiomysql for async operations
    import aiomysql
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False
    aiomysql = None # Define aiomysql as None if import fails

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 从环境变量读取数据库连接配置
DB_CONFIG = {
    "enabled": os.environ.get("MYSQL_ENABLED", "false").lower() == "true",
    "host": os.environ.get("MYSQL_HOST", "localhost"),
    "port": int(os.environ.get("MYSQL_PORT", "3306")),
    "database": os.environ.get("MYSQL_DATABASE", ""),
    "username": os.environ.get("MYSQL_USERNAME", "root"),
    "password": os.environ.get("MYSQL_PASSWORD", ""),
    "pool_minsize": int(os.environ.get("MYSQL_POOL_MINSIZE", "1")),
    "pool_maxsize": int(os.environ.get("MYSQL_POOL_MAXSIZE", "10")),
    "resource_desc_file": os.environ.get("MYSQL_RESOURCE_DESC_FILE", ""),  # 资源描述文件路径（必需）
    "max_rows": int(os.environ.get("MAX_ROWS", "20"))
}


@dataclass
class DatabaseConnection:
    connection: Any  # The actual database connection object
    database: str
    last_used: float = field(default_factory=lambda: 0)
    
@dataclass
class AppContext:
    # Store the connection pool instead of a single connection
    # Use Any to avoid issues with conditional import and type checkers
    pool: Optional[DatabaseConnection] = None

# --- MCP Server Setup ---
app_context = AppContext()
server = Server("mysql-mcp")

# 从文件加载资源描述
def load_resource_description():
    """从文件加载资源描述，文件路径必须指定"""
    file_path = DB_CONFIG["resource_desc_file"]
    if not file_path:
        logger.error("Resource description file path (MYSQL_RESOURCE_DESC_FILE) is required but not specified")
        sys.exit(1)
        
    if not os.path.isfile(file_path):
        logger.error(f"Resource description file {file_path} does not exist")
        sys.exit(1)
        
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if not content:
                logger.error(f"Resource description file {file_path} is empty")
                sys.exit(1)
            logger.info(f"Loaded resource description from {file_path}")
            return content
    except Exception as e:
        logger.error(f"Error reading resource description file {file_path}: {e}")
        sys.exit(1)

mysql_resource_description = load_resource_description()

# Define tool specifications
tool_specs = [
    {
        "name": "mysql_execute_read_query",
        "description": "Execute read-only MySQL SQL code. Only SELECT, SHOW, DESCRIBE, EXPLAIN allowed; queries are validated before execution.\n\n"+mysql_resource_description,
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The SQL code to execute (only SELECT, SHOW, DESCRIBE, EXPLAIN allowed)"
                },
                "params": {
                    "type": "object",
                    "description": "Parameters for the SQL query (for parameterized queries)",
                    "additionalProperties": True
                },
                "max_rows": {
                    "type": "integer",
                    "description": "Maximum number of rows to return",
                    "default": DB_CONFIG["max_rows"]
                }
            },
            "required": ["query"]
        }
    },
]

# --- Lifespan Management ---
@asynccontextmanager
async def app_lifespan() -> AsyncIterator[None]:
    """Manage application lifecycle with MySQL connection pool."""
    pool = None
    if MYSQL_AVAILABLE and DB_CONFIG["enabled"]:
        config = DB_CONFIG
        try:
            pool = await aiomysql.create_pool(
                host=config["host"],
                port=config["port"],
                user=config["username"],
                password=config["password"],
                db=config["database"],
                minsize=config["pool_minsize"],
                maxsize=config["pool_maxsize"],
                autocommit=True # Set autocommit based on your needs
            )
            app_context.pool = pool
            logger.info(f"Connected to MySQL and created connection pool for {config['host']}:{config['port']}")
        except Exception as e:
            logger.error(f"Failed to create MySQL connection pool during startup: {e}")
            app_context.pool = None # Ensure pool is None if setup fails

    try:
        yield # Server runs here
    finally:
        # Cleanup on shutdown
        if app_context.pool:
            logger.info("Closing MySQL connection pool")
            app_context.pool.close()
            await app_context.pool.wait_closed()
            logger.info("MySQL connection pool closed")
        else:
            logger.info("No active MySQL connection pool to close.")

# --- Connection Management Functions ---

def format_query_results(result: Dict) -> str:
    """格式化查询结果为字符串表格"""
    if not result.get("data"):
        return f"Query executed. Rows returned: {result.get('row_count', 0)}"
        
    # 格式化为表格
    output = []
    column_names = result["column_names"]
    
    # 计算列宽
    widths = [len(col) for col in column_names]
    for row in result["data"]:
        for i, col in enumerate(column_names):
            value = str(row.get(col, ''))
            widths[i] = max(widths[i], len(value))
    
    # 创建表头
    header = " | ".join(col.ljust(widths[i]) for i, col in enumerate(column_names))
    separator = "-+-".join("-" * w for w in widths)
    
    output.append(header)
    output.append(separator)
    
    # 创建数据行
    for row in result["data"]:
        row_str = " | ".join(str(row.get(col, '')).ljust(widths[i]) for i, col in enumerate(column_names))
        output.append(row_str)
    
    output.append("")
    output.append(f"Total rows: {result['row_count']} (showing first {len(result['data'])})")
    
    return "\n".join(output)

# --- Tool Implementations ---

async def tool_mysql_execute_read_query(args: dict) -> str:
    try:
        query = args["query"]
        params = args.get("params", {})
        # Get max_rows from args or config, no longer capping here
        max_rows = args.get("max_rows", DB_CONFIG["max_rows"])

        # 执行查询
        return await execute_db_query(query, params, max_rows)

    except KeyError as e:
        return f"Error: Missing required parameter: {e.args[0]}"
    except Exception as e:
        return f"Error: {str(e)}\n{traceback.format_exc()}"


async def execute_db_query(query: str, params: tuple | list, max_rows: int) -> str:
    """执行数据库查询并返回结果"""
    if not app_context.pool:
        return "Error: MySQL connection pool not available or not enabled."

    try:
        # 安全检查：只允许SELECT、SHOW、DESCRIBE、EXPLAIN等只读操作
        query_lower = query.strip().lower()
        allowed_prefixes = ("select", "show", "describe", "desc", "explain")
        if not query_lower.startswith(allowed_prefixes):
            return f"Error: Only read operations (SELECT, SHOW, DESCRIBE, EXPLAIN) are allowed. Rejected query: {query}"

        # 防止多语句执行 (basic check)
        # Note: aiomysql might handle this better, but a basic check is good defense.
        if query.count(';') > 1 or (query.count(';') == 1 and not query.strip().endswith(';')):
             return f"Error: Multiple statements are not allowed. Rejected query: {query}"

        # Check if query already has a LIMIT clause (case-insensitive regex)
        # Simple check for now, can be improved with regex if needed
        original_query = query.strip()
        if "limit" not in original_query.lower().split()[-2:]:
            # Append LIMIT clause only if it's a SELECT query
            if query_lower.startswith("select"):
                query = f"{original_query} LIMIT %s"
                # Add max_rows to params. Need to handle dict vs list/tuple params.
                if isinstance(params, dict):
                    # aiomysql with dict params uses %(key)s format.
                    # Cannot easily mix formats. We'll stick to list/tuple for simplicity when adding LIMIT.
                    # If original params were dict, this might break. Consider converting dict to list based on query placeholders.
                    # For now, assume params are list/tuple or empty if we add LIMIT.
                    if params: 
                        logger.warning("Mixing dict params with automatic LIMIT might not work as expected.")
                        # Attempt conversion (naive: assumes order matches simple query)
                        # This is fragile and might need a more robust solution based on parsing query placeholders
                        params_list = list(params.values())
                        params_list.append(max_rows)
                        params = tuple(params_list)
                    else:
                         params = (max_rows,)
                elif isinstance(params, (list, tuple)):
                    params = tuple(params) + (max_rows,)
                else: # Assuming params is empty or None
                    params = (max_rows,)
                logger.info(f"Appending LIMIT {max_rows} to query.")
            else:
                logger.info(f"Query is not SELECT, skipping automatic LIMIT addition.")

        async with app_context.pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                # Execute query
                logger.info(f"Executing query: {query} with params: {params}")
                await cursor.execute(query, params)

                # Fetch results - fetchall now since LIMIT is in query
                rows = await cursor.fetchall()
                total_row_count = cursor.rowcount # Might still be useful, or could be len(rows)

                # Simplify total_row_count logic as LIMIT is applied
                if total_row_count == -1:
                    total_row_count = len(rows)

                # Get column names from cursor description
                column_names = [desc[0] for desc in cursor.description] if cursor.description else []

                result_data = {
                    "success": True,
                    "data": rows,
                    "error": None,
                    "row_count": total_row_count if total_row_count != -1 else len(rows),
                    "column_names": column_names
                }

        # 格式化结果为字符串
        return format_query_results(result_data)

    except Exception as e:
        return f"Database error: {str(e)}\n{traceback.format_exc()}"

# --- MCP Handlers ---
@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name=spec["name"],
            description=spec["description"],
            inputSchema=spec["inputSchema"],
        ) for spec in tool_specs
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict[str, Any] | None) -> list[types.TextContent]:
    try:
        arguments = arguments or {}
        
        if name == "mysql_execute_read_query":
            result = await tool_mysql_execute_read_query(arguments)
        else:
            result = f"Unknown tool: {name}"
            
        return [types.TextContent(type="text", text=result)]
    except Exception as e:
        return [types.TextContent(type="text", text=f"Error: {str(e)}")]

# --- Main entry ---
async def main():
    from mcp.server.stdio import stdio_server
    
    # 打印数据库配置信息
    logger.info("MySQL configuration:")
    enabled = "ENABLED" if DB_CONFIG["enabled"] else "DISABLED"
    logger.info(f"- MySQL ({enabled}): {DB_CONFIG['host']}:{DB_CONFIG['port']}, database: {DB_CONFIG['database']}")
    if DB_CONFIG["enabled"]:
        logger.info(f"  Pool MinSize: {DB_CONFIG['pool_minsize']}, Pool MaxSize: {DB_CONFIG['pool_maxsize']}")
    
    # 打印资源描述文件信息
    logger.info(f"- Resource description file: {DB_CONFIG['resource_desc_file']}")

    # Use lifespan manager
    async with app_lifespan():
        async with stdio_server() as (read_stream, write_stream):
            logger.info("MySQL MCP server running with stdio transport")
            await server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="mysql-mcp",
                    server_version="0.1.0",
                    capabilities=server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )

if __name__ == "__main__":
    # Ensure aiomysql is available before running
    if not MYSQL_AVAILABLE:
        logger.error("aiomysql library is required but not installed. Please install it: pip install aiomysql")
        sys.exit(1)
    asyncio.run(main()) 