import requests
from typing import Optional
import base64
from urllib.parse import urlparse
from utils.logger import setup_logger
import os
from dotenv import load_dotenv

# Get the parent directory of the current file's directory
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(dotenv_path)

logger = setup_logger("GithubReader")

class GithubReader:
    """Utility for reading content from GitHub files"""
    
    def __init__(self):
        self.logger = logger
        self.base_api_url = "https://api.github.com/repos"
        self.github_token = os.getenv('GITHUB_TOKEN')
        if not self.github_token:
            self.logger.warning("No GitHub token found in environment variables")
            
    def get_headers(self):
        """Get headers for GitHub API requests"""
        headers = {'Accept': 'application/vnd.github.v3+json'}
        if self.github_token:
            headers['Authorization'] = f'token {self.github_token}'
        return headers
    
    def extract_repo_info(self, github_url: str) -> tuple[str, str, str]:
        """Extract owner, repo, and path from GitHub URL"""
        parsed = urlparse(github_url)
        parts = parsed.path.strip('/').split('/')
        
        if len(parts) < 3:
            raise ValueError("Invalid GitHub URL format")
            
        owner = parts[0]
        repo = parts[1]
        path = '/'.join(parts[2:]) if len(parts) > 2 else ''
        
        return owner, repo, path
    
    def read_file(self, url: str) -> Optional[str]:
        """
        Fetches and returns the content of a file from GitHub using the API.
        
        Args:
            url (str): GitHub URL to the file
                (e.g., https://github.com/owner/repo/blob/main/path/to/file.txt)
            
        Returns:
            Optional[str]: File content if successful, None if failed
        """
        try:
            owner, repo, path = self.extract_repo_info(url)
            
            # Remove 'blob/main' or 'blob/master' from path
            path = path.replace('blob/main/', '').replace('blob/master/', '')
            
            # Construct API URL
            api_url = f"{self.base_api_url}/{owner}/{repo}/contents/{path}"
            
            self.logger.info(f"Fetching content from GitHub API: {api_url}")
            response = requests.get(api_url, headers=self.get_headers())
            
            if response.status_code != 200:
                self.logger.warning(f"Failed to fetch GitHub content: {response.status_code}")
                return None
                
            # GitHub API returns content as base64
            content = response.json().get('content', '')
            if content:
                decoded_content = base64.b64decode(content).decode('utf-8')
                self.logger.info("Successfully fetched GitHub content")
                return decoded_content
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error fetching GitHub content: {str(e)}")
            return None 