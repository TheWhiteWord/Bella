from typing import List, Dict, Optional, Set
import asyncio
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from trafilatura import fetch_url, extract
from loguru import logger
import ollama
from functools import lru_cache

class SearchAgent:
    """Autonomous agent for web search and content summarization using Ollama."""
    
    def __init__(self, 
                 max_depth: int = 2, 
                 max_links_per_page: int = 3,
                 model: str = "Gemma"):
        """
        Initialize search and summarization components.
        
        Args:
            max_depth: Maximum depth for recursive search (default: 2)
            max_links_per_page: Maximum links to follow per page (default: 3)
            model: Ollama model name (default: "Gemma")
        """
        self.ddgs = DDGS()
        self.model = model
        self.max_depth = max_depth
        self.max_links_per_page = max_links_per_page
        self.visited_urls: Set[str] = set()

    async def summarize_text(self, text: str, max_length: int = 150) -> str:
        """
        Summarize text using Ollama model.
        
        Args:
            text: Text to summarize
            max_length: Maximum length of summary (used in prompt)
            
        Returns:
            Summarized text
        """
        try:
            prompt = (
                f"Summarize the following text in {max_length} words or less, "
                "focusing on key facts and main ideas. Be direct and concise:\n\n"
                f"{text}"
            )
            
            # Run ollama.chat in a thread pool since it's synchronous
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: ollama.chat(
                    model=self.model,
                    messages=[{
                        'role': 'system',
                        'content': 'You are a precise summarizer. Provide direct summaries without phrases like "Here is a summary" or "The text discusses". Focus on key facts and insights.',
                    }, {
                        'role': 'user',
                        'content': prompt
                    }],
                    stream=False,
                    options={
                        "temperature": 0.3,
                        "top_p": 0.8
                    }
                )
            )
            
            return response['message']['content'].strip()
            
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            return "Failed to generate summary."

    async def extract_links(self, html: str, base_url: str) -> List[str]:
        """Extract links from HTML content."""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            links = []
            
            for link in soup.find_all('a', href=True):
                href = link['href']
                if href.startswith(('http://', 'https://')):
                    links.append(href)
                elif not href.startswith(('#', 'javascript:', 'mailto:')):
                    full_url = urljoin(base_url, href)
                    links.append(full_url)
                    
            return links[:self.max_links_per_page]
        except Exception as e:
            logger.error(f"Link extraction failed: {e}")
            return []

    async def recursive_fetch(self, url: str, depth: int = 0) -> List[Dict]:
        """
        Recursively fetch and process content from URLs.
        
        Args:
            url: URL to process
            depth: Current depth of recursion
            
        Returns:
            List of dictionaries containing summaries and sources
        """
        if depth >= self.max_depth or url in self.visited_urls:
            return []
            
        self.visited_urls.add(url)
        results = []
        
        try:
            content = await self.fetch_content(url)
            if content:
                summary = await self.summarize_text(content)
                results.append({
                    'summary': summary,
                    'source': url,
                    'depth': depth
                })
                
                if depth < self.max_depth:
                    downloaded = fetch_url(url)
                    if downloaded:
                        links = await self.extract_links(downloaded, url)
                        tasks = []
                        for link in links[:self.max_links_per_page]:
                            if link not in self.visited_urls:
                                tasks.append(self.recursive_fetch(link, depth + 1))
                        if tasks:
                            nested_results = await asyncio.gather(*tasks)
                            for nested in nested_results:
                                results.extend(nested)
                            
        except Exception as e:
            logger.error(f"Recursive fetch failed for {url}: {e}")
            
        return results

    async def search(self, query: str, max_results: int = 3) -> List[Dict]:
        """
        Perform web search asynchronously.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            List of search results with standardized structure
        """
        try:
            # Run search in thread pool since DDGS is synchronous
            loop = asyncio.get_event_loop()
            logger.info(f"Searching for: {query}")
            raw_results = await loop.run_in_executor(
                None, 
                lambda: list(self.ddgs.text(query, max_results=max_results))
            )
            logger.info(f"Raw results: {raw_results}")
            
            # Standardize result structure
            results = []
            for result in raw_results:
                logger.info(f"Processing result: {result}")
                standardized = {
                    'title': result.get('title'),
                    'link': result.get('href'),  # DuckDuckGo uses 'href' instead of 'link'
                    'snippet': result.get('body')
                }
                if standardized['title'] and standardized['link']:
                    results.append(standardized)
            
            logger.info(f"Standardized results: {results}")
            return results
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    async def fetch_content(self, url: str) -> Optional[str]:
        """
        Fetch and extract main content from URL.
        
        Args:
            url: URL to fetch content from
            
        Returns:
            Extracted text content or None if failed
        """
        try:
            downloaded = fetch_url(url)
            if downloaded:
                content = extract(downloaded)
                return content if content else None
            return None
        except Exception as e:
            logger.error(f"Content extraction failed for {url}: {e}")
            return None
    
    async def research_topic(self, query: str) -> str:
        """
        Perform autonomous research on a topic.
        
        Args:
            query: Research topic/query
            
        Returns:
            Formatted research results
        """
        self.visited_urls.clear()
        
        try:
            # Initial search
            search_results = await self.search(query)
            if not search_results:
                return "I couldn't find any relevant information."
            
            # Process results recursively
            all_summaries = []
            for result in search_results:
                if 'link' in result:
                    summaries = await self.recursive_fetch(result['link'])
                    all_summaries.extend(summaries)
            
            if not all_summaries:
                return "I found sources but couldn't extract meaningful content."
            
            # Sort by depth and compose response
            all_summaries.sort(key=lambda x: x['depth'])
            
            response = f"# Research Results: {query}\n\n"
            
            current_depth = -1
            for item in all_summaries:
                if item['depth'] != current_depth:
                    current_depth = item['depth']
                    response += f"\n## Depth Level {current_depth}\n\n"
                
                response += f"- {item['summary']}\n"
                response += f"  Source: {item['source']}\n\n"
            
            return response
            
        except Exception as e:
            logger.error(f"Research failed: {e}")
            return f"Research failed: {str(e)}"