import logging
from typing import Dict, List
from utils.logger import setup_logger
from langchain_community.chat_models import ChatPerplexity
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from utils.envvars import PPLX_API_KEY
from agent.data_class.blog_data import ResearchResponse

logger = setup_logger("PerplexityTool")

class PerplexityTool:
    """Tool for interacting with Perplexity AI API"""
    
    def __init__(self):
        self.logger = logger
        self.logger.info("Initializing PerplexityTool")
        self.pplx_chat = ChatPerplexity(api_key=PPLX_API_KEY)
    
    def query(self, thesis: str) -> ResearchResponse:
        try:
            self.logger.info(f"Querying Perplexity AI for topic")
            human_prompt = f"I want to research this thesis that I'm writing about, \
                so tell me all about this topic given the outline I've provided, provide as much detail as you can, be thorough. \
                Thesis: {thesis}"
            messages = [HumanMessage(content=human_prompt)]
            response = self.pplx_chat.invoke(messages)
            pplx_response = ResearchResponse(content="", sources=[])

            if isinstance(response, AIMessage):
                self.logger.info("Got research response from Perplexity")
                research_content = response.content
                citations = response.additional_kwargs['citations']
                pplx_response.content = research_content
                pplx_response.sources = citations
                self.logger.debug(f"Perplexity response: {pplx_response}")
                return pplx_response
            else:
                self.logger.error("Received unexpected response type from Perplexity API")
                raise ValueError(f"Unexpected response type from Perplexity API: {type(response)}")
            
        except Exception as e:
            self.logger.error(f"Perplexity API call failed: {str(e)}")
            raise
