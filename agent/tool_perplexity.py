import logging
from typing import Dict, List
from typing_extensions import TypedDict
from utils.logger import setup_logger
from langchain_community.chat_models import ChatPerplexity
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

logger = setup_logger("PerplexityTool")

class ResearchResponse(TypedDict):
    content: str
    sources: List[str]

class PerplexityTool:
    """Tool for interacting with Perplexity AI API"""
    
    def __init__(self):
        self.logger = logger
        self.logger.info("Initializing PerplexityTool")
        self.pplx_chat = ChatPerplexity()
    
    def query(self, initial_topic: str, blog_outline: str) -> Dict:
        """
        Queries the Perplexity AI API for background research on a blog topic.
        
        This function leverages Perplexity AI's capabilities to gather comprehensive
        background information based on the blog topic and outline. The API provides:
        - Enhanced web searching with direct, comprehensive answers
        - In-depth research capabilities with source verification
        - Content creation and summarization features
        - Real-time information processing
        - Fact-checking with citations
        
        Args:
            initial_topic (str): The main topic or title of the blog post
            blog_outline (str): The structured outline of the blog post
            
        Returns:
            dict: Response from Perplexity containing:
                - content: Synthesized background research
                - sources: List of sources used
                
        Example:
            tool = PerplexityTool()
            response = tool.query(
                "AI Ethics in Healthcare",
                "TITLE: The Future of AI in Healthcare\\nTHESIS: AI is transforming..."
            )
            print(response['content'])
        """
        try:
            self.logger.info(f"Querying Perplexity AI for topic: {initial_topic}")
            human_prompt = f"I want to research the following topic, \
                so tell me all about this topic given the outline I've provided, provide as much detail as you can, be thorough. \
                Topic: {initial_topic}\n\n\
                Outline: {blog_outline}"
            messages = [HumanMessage(content=human_prompt)]
            response = self.pplx_chat.invoke(messages)
            pplx_response = ResearchResponse()

            if isinstance(response, AIMessage):
                self.logger.info("Got research response from Perplexity")
                research_content = response.content
                citations = response.additional_kwargs['citations']
                pplx_response["content"] = research_content
                pplx_response["sources"] = citations
                self.logger.debug(f"Perplexity response: {pplx_response}")
                return pplx_response
            else:
                self.logger.error("Received unexpected response type from Perplexity API")
                raise ValueError(f"Unexpected response type from Perplexity API: {type(response)}")
            
        except Exception as e:
            self.logger.error(f"Perplexity API call failed: {str(e)}")
            raise
