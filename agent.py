"""Step 1 + 8: LLM + ReAct-style agent."""
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
import config
from tools import TOOLS

SYSTEM_PROMPT = """You are a personal investment research assistant.

Follow this procedure for EVERY question:
1. Decide what is needed: (a) LIVE data (current price, recent news), (b) HISTORICAL/STATIC
   data from the ingested documents (reported revenue, profit, expense ratio, definitions,
   investing basics), or (c) BOTH.
2. Call the right tool(s):
   - get_stock_price / get_stock_news for live data
   - search_financial_documents for anything in the reports/fact sheets
   If a question needs both (e.g. "is the current price high relative to last year's EPS?"),
   call both tools before answering.
3. Combine the results into ONE concise, grounded answer. Never invent numbers; if the tools
   do not return a figure, say so.
4. ALWAYS label every number with its source and unit:
   "(live market, INR per share)" or "(annual report, p.X, INR crores, FY2024)".
   Never mix units (INR vs USD, crores vs millions) without converting explicitly and saying so.
5. Do not give personalized buy/sell advice. End investment-related answers with:
   "This is not personalized financial advice."
"""

llm = ChatOllama(model=config.LLM_MODEL, temperature=0)
agent = create_agent(model=llm, tools=TOOLS, system_prompt=SYSTEM_PROMPT)


def ask(messages: list[dict]) -> str:
    result = agent.invoke({"messages": messages})
    return result["messages"][-1].content


if __name__ == "__main__":
    print(ask([{"role": "user", "content": "What is the current price of AAPL?"}]))
