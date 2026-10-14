"""BONUS: serve tools over MCP.  Run:  python mcp_server.py   (port 8001)"""
from mcp.server.fastmcp import FastMCP
from tools import get_stock_price, get_stock_news, search_financial_documents

mcp = FastMCP("finance-tools", port=8001)

# .func gives the raw python function behind the @tool wrapper
for t in (get_stock_price, get_stock_news, search_financial_documents):
    mcp.tool(name=t.name, description=t.description)(t.func)

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
