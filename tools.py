"""Step 6-7: custom tools. Docstrings state units explicitly."""
import os
import yfinance as yf
from langchain_core.tools import tool
from ingest import get_retriever

_retriever = None


@tool
def get_stock_price(ticker: str) -> str:
    """LIVE MARKET DATA. Get the latest traded price for a stock ticker.
    Use Yahoo Finance tickers: NSE stocks end in '.NS' (e.g. RELIANCE.NS, price in INR),
    BSE in '.BO' (INR), US stocks have no suffix (e.g. AAPL, price in USD).
    Returns the price with its currency. Prices are per share, NOT in crores/millions."""
    try:
        fi = yf.Ticker(ticker).fast_info
        price, cur = fi["last_price"], fi["currency"]
        return f"[SOURCE: LIVE MARKET] {ticker.upper()} last price: {price:,.2f} {cur} per share."
    except Exception as e:
        return f"[SOURCE: LIVE MARKET] Could not fetch price for {ticker}: {e}"


@tool
def get_stock_news(ticker: str) -> str:
    """LIVE MARKET DATA. Get up to 5 recent news headlines for a stock ticker
    (same ticker format as get_stock_price: RELIANCE.NS, AAPL, ...). Returns headlines only."""
    try:
        items = yf.Ticker(ticker).news[:5]
        lines = []
        for n in items:
            c = n.get("content", n)  # yfinance changed schema across versions
            title = c.get("title", "(no title)")
            pub = (c.get("provider") or {}).get("displayName") or c.get("publisher", "")
            lines.append(f"- {title} ({pub})")
        return "[SOURCE: LIVE NEWS]\n" + ("\n".join(lines) or "No recent news found.")
    except Exception as e:
        return f"[SOURCE: LIVE NEWS] Could not fetch news for {ticker}: {e}"


@tool
def search_financial_documents(query: str) -> str:
    """STATIC DOCUMENT DATA. Search the ingested annual reports / fund fact sheets / guides
    and return the 3 most relevant passages with file name and page number.
    Figures are as reported in the document, in the document's own units and currency
    (e.g. INR crores, USD millions) and for the document's reporting period, NOT today's values.
    Always quote the unit and period found in the passage. Use specific queries like
    'consolidated revenue Q2 FY25' or 'expense ratio direct plan'."""
    global _retriever
    if _retriever is None:
        _retriever = get_retriever()
    docs = _retriever.invoke(query)
    if not docs:
        return "[SOURCE: STATIC DOCUMENT] No relevant passages found."
    out = ["[SOURCE: STATIC DOCUMENT]"]
    for d in docs:
        out.append(f"(file: {os.path.basename(d.metadata.get('source',''))}, "
                   f"page: {d.metadata.get('page','?')})\n{d.page_content}")
    return "\n\n---\n\n".join(out)


TOOLS = [get_stock_price, get_stock_news, search_financial_documents]
