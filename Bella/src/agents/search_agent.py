from typing import List, Dict, Optional, Set
import asyncio
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from duckduckgo_search import AsyncDDGS
from trafilatura import fetch_url, extract
from loguru import logger
import ollama

class SearchAgent:
    """Autonomous agent for web search and content summarization using Ollama."""
    
    def __init__(self, 
                 max_depth: int = 2, 
                 max_links_per_page: int = 3,
                 model: str = "Gemma3:latest"):
        """
        Initialize search and summarization components.
        
        Args:
            max_depth: Maximum depth for recursive search (default: 2)
            max_links_per_page: Maximum links to follow per page (default: 3)
            model: Ollama model name (default: "Gemma3:latest")
        """
        self.ddgs = AsyncDDGS()
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
            # Create a more structured prompt for better summarization
            prompt = (
                "You are a precise summarizer. "
                "Analyze and summarize the following text. "
                f"Create a concise summary in {max_length} words or less. "
                "Focus on key facts and main ideas.\n\n"
                "Text to summarize:\n"
                f"{text}\n\n"
                "Summary:"
            )
        
            response = await ollama.chat(
                model=self.model,
                messages=[{
                    'role': 'system',
                    'content': 'You are a precise summarizer focused on factual accuracy.',
                }, {
                    'role': 'user',
                    'content': prompt
                }],
                stream=False,
                options={
                    "temperature": 0.3,  # Lower temperature for more focused summaries
                    "top_p": 0.8        # Reduce randomness while maintaining coherence
                }
            )
        
            return response['message']['content'].strip()
        
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            return "Failed to generate summary."

    @lru_cache(maxsize=100)
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
            # Fetch and process current page
            downloaded = fetch_url(url)
            if downloaded:
                content = extract(downloaded)
                if content:
                    summary = self.summarize_text(content)
                    results.append({
                        'summary': summary,
                        'source': url,
                        'depth': depth
                    })
                    
                    # Extract and process nested links if not at max depth
                    if depth < self.max_depth:
                        links = await self.extract_links(downloaded, url)
                        tasks = [
                            self.recursive_fetch(link, depth + 1)
                            for link in links
                            if link not in self.visited_urls
                        ]
                        nested_results = await asyncio.gather(*tasks)
                        for nested in nested_results:
                            results.extend(nested)
                            
        except Exception as e:
            logger.error(f"Recursive fetch failed for {url}: {e}")
            
        return results

    async def research_topic(self, query: str) -> str:
        """Perform autonomous research on a topic with recursive search."""
        self.visited_urls.clear()  # Reset visited URLs for new search
        
        # Initial search
        search_results = await self.search(query)
        if not search_results:
            return "I couldn't find any relevant information."
        
        # Process each result recursively
        all_summaries = []
        for result in search_results:
            summaries = await self.recursive_fetch(result['link'])
            all_summaries.extend(summaries)
        
        # Sort by depth and compose response
        all_summaries.sort(key=lambda x: x['depth'])
        
        if not all_summaries:
            return "I found sources but couldn't extract meaningful content."
        
        response = f"# Research Results: {query}\n\n"
        
        # Group by depth level
        current_depth = -1
        for item in all_summaries:
            if item['depth'] != current_depth:
                current_depth = item['depth']
                response += f"\n## Depth Level {current_depth}\n\n"
            
            response += f"- {item['summary']}\n"
            response += f"  Source: {item['source']}\n\n"
        
        return response
        
    async def search(self, query: str, max_results: int = 3) -> List[Dict]:
        """Perform web search asynchronously."""
        try:
            results = await self.ddgs.text(query, max_results=max_results)
            return list(results)  # Convert generator to list
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    async def fetch_content(self, url: str) -> Optional[str]:
        """Fetch and extract main content from URL."""
        try:
            downloaded = fetch_url(url)
            if downloaded:
                return extract(downloaded)
            return None
        except Exception as e:
            logger.error(f"Content extraction failed for {url}: {e}")
            return None
    
    def summarize_text(self, text: str, max_length: int = 150) -> str:
        """Summarize text using transformers."""
        try:
            summary = self.summarizer(
                text, 
                max_length=max_length, 
                min_length=30,
                do_sample=False
            )
            return summary[0]['summary_text']
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            return "Failed to generate summary."
    
    async def research_topic(self, query: str) -> str:
        """Perform autonomous research on a topic."""
        # Search for relevant content
        search_results = await self.search(query)
        if not search_results:
            return "I couldn't find any relevant information."
        
        summaries = []
        for result in search_results:
            # Fetch and extract content
            content = await self.fetch_content(result['link'])
            if content:
                # Summarize the content
                summary = self.summarize_text(content)
                summaries.append({
                    'summary': summary
                })
        
        # Compose final response
        if not summaries:
            return "I found sources but couldn't extract meaningful content."
        
        response = f"Here's what I found about {query}:\n\n"
        for i, item in enumerate(summaries, 1):
            response += f"{i}. {item['summary']}\n"
            response += f"Source: {item['source']}\n\n"
        
        return response