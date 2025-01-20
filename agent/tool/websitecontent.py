import logging
from typing import Dict, List
from utils.logger import setup_logger
import requests
from bs4 import BeautifulSoup
import html2text

logger = setup_logger("WebsiteContentTool")

class WebsiteContentTool:
    """Tool for fetching and converting website content to markdown"""
    
    def __init__(self):
        self.logger = logger
        self.html2text = html2text.HTML2Text()
        self.html2text.ignore_links = False
        self.html2text.ignore_images = False
        
    def get_content(self, url: str) -> str:
        """
        Fetches website content and converts it to markdown.
        
        Args:
            url (str): The website URL to fetch
            
        Returns:
            str: Website content converted to markdown format, or empty content notice if fetch fails
            
        """
        try:
            self.logger.info(f"Fetching content from: {url}")
            response = requests.get(url)
            
            if response.status_code != 200:
                self.logger.warning(f"Failed to fetch content from {url} (Status: {response.status_code})")
                return f"--- CONTENT FROM {url} NOT AVAILABLE (Status: {response.status_code}) ---\n\n"
            
            soup = BeautifulSoup(response.text, 'html.parser')
            markdown = self.html2text.handle(str(soup))
            markdown = f"--- START OF CONTENT FROM {url} ---\n\n" + markdown + "\n\n--- END OF CONTENT FROM {url} ---\n\n"
            
            self.logger.info(f"Successfully got content from {url}")
            return markdown
            
        except Exception as e:
            self.logger.error(f"Failed to fetch/convert content: {str(e)}")
            return f"--- CONTENT FROM {url} NOT AVAILABLE (Error: {str(e)}) ---\n\n"

    def get_content_from_urls(self, urls: List[str]) -> str:
        """
        Fetches content from multiple websites and concatenates them into a single string
        """
        content = ""
        self.logger.info(f"WebsiteContentTool: Fetching content from {len(urls)} websites")
        for url in urls:
            self.logger.info(f"WebsiteContentTool: Fetching content from: {url}")
            content += self.get_content(url)
        self.logger.info(f"WebsiteContentTool: Content from {len(urls)} websites fetched")
        return content
