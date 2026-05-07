import requests
from bs4 import BeautifulSoup
from urllib.parse import quote_plus


class WebSearchTool:
    def __init__(self, max_results: int = 5, max_text_chars: int = 3000, timeout: int = 10):
        self.max_results = max_results
        self.max_text_chars = max_text_chars
        self.timeout = timeout
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        }

    def search(self, query: str) -> list[dict]:
        url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
        try:
            resp = requests.get(url, headers=self.headers, timeout=self.timeout)
            resp.raise_for_status()
        except Exception as e:
            return [{"title": "検索失敗", "url": "", "snippet": str(e)}]

        soup = BeautifulSoup(resp.text, "html.parser")
        results = []
        for r in soup.select(".result")[:self.max_results]:
            title_el = r.select_one(".result__a")
            snippet_el = r.select_one(".result__snippet")
            url_el = r.select_one(".result__url")
            title = title_el.get_text(strip=True) if title_el else ""
            snippet = snippet_el.get_text(strip=True) if snippet_el else ""
            href = url_el.get_text(strip=True) if url_el else ""
            if title:
                results.append({"title": title, "url": href, "snippet": snippet[:self.max_text_chars // self.max_results]})

        return results if results else [{"title": "結果なし", "url": "", "snippet": ""}]
