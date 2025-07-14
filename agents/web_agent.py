# === agents/web_agent.py ===
import requests
from bs4 import BeautifulSoup
from langchain.agents import Tool, initialize_agent, AgentType
from langchain_ollama import OllamaLLM

def google_scrape_search(query: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0"}
    url = f"https://www.google.com/search?q={query.replace(' ', '+')}"

    try:
        resp = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        results = []

        for g in soup.find_all("div", class_="tF2Cxc")[:5]:
            title_elem = g.find("h3")
            link_elem = g.find("a")
            snippet_elem = g.find("div", class_="VwiC3b")

            if not (title_elem and link_elem):
                continue

            title = title_elem.text.strip()
            link = link_elem['href'].strip()
            snippet = snippet_elem.text.strip() if snippet_elem else ""

            results.append(f"- **{title}**\n  {snippet}\n  🌐 [Source]({link})")

        if not results:
            return "⚠ No results found."

        return "\n".join(results)

    except Exception as e:
        return f"⚠ Error during scraping: {e}"

def build_web_agent(llm=None):
    if llm is None:
        llm = OllamaLLM(model="qwen3:32b", temperature=0.0, streaming=True)

    web_tool = Tool(
        name="WebSearch",
        func=google_scrape_search,
        description="Scrapes top 5 Google search results with titles, snippets, and URLs."
    )

    return initialize_agent(
        [web_tool],
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        max_iterations=5,
        early_stopping_method="generate",
        verbose=True
    )
