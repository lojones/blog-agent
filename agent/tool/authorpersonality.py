from typing import Dict
from utils.logger import setup_logger
import os
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from utils.github_reader import GithubReader
from agent.data_class.blog_data import BlogState
logger = setup_logger("PersonalityTool")

# Get the parent directory of the current file's directory
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(dotenv_path)

class PersonalityTool:
    """Tool for analyzing and suggesting author personality traits for blog writing"""
    
    def __init__(self):
        self.logger = logger
        self.github_reader = GithubReader()
        self.personality = self.github_reader.read_file('https://github.com/lojones/lojo-personality/blob/main/blog-post-author.md')
        
        if not self.personality:
            self.logger.error("Couldnt get the personality profile from GitHub")
            raise ValueError("Couldnt get the personality profile from GitHub")
        self.logger.info(f"Initialized with personality profile: {self.personality[:50]}...")
        self.llm_anthropic = ChatAnthropic(model="claude-3-5-sonnet-20240620", 
                                    temperature=0.7,
                                    max_tokens=8000)
        
    def get_author_personality(self) -> str:
        self.logger.info("PersonalityTool: Getting author's personality")
        return self.personality

    def personalize(self, initial_topic: str,blog_outline: str,  background_research: str) -> str:
        """
        Analyzes blog content and generates relevant author personality traits to incorporate.
        
        This method examines the blog outline, initial topic, and background research to determine
        which aspects of the author's experience and personality would be most relevant to
        incorporate into the final article. It uses an LLM to generate a contextual prompt that includes:
        
        - Relevant professional experience
        - Personal anecdotes and stories
        - Past work or research in the topic area
        - Writing style preferences
        - Unique perspectives or insights
        - Domain expertise and credentials
        
        Args:
            initial_topic (str): The original topic or title of the blog post
            blog_outline (str): The structured outline of the blog post
            background_research (str): Additional research and context gathered
            
        Returns:
            Str: A prompt package containing the following sections with titles:
                - prompt: Tailored prompt text describing relevant author details
                - style_guide: Writing style recommendations
                - key_experiences: List of specific experiences to reference
                - credentials: Relevant credentials to establish authority
                
        Example:
            tool = PersonalityTool()
            result = tool.personalize(
                "TITLE: AI Ethics in Healthcare...",
                "AI in Healthcare",
                "Recent developments in medical AI..."
            )
            print(result['prompt'])
        """
        try:
            self.logger.info(f"Analyzing outline for personality traits: {initial_topic}")

            prompt = f"Analyze the following blog topic, outline, and background research and determine the most relevant author personality traits:\
                Blog topic: {initial_topic}\nBlog Outline: {blog_outline}\nBackground research: {background_research}"
            sysprompt =f"You are an expert in author personality traits and will be given the topic, outline, and background research \
                of a blog post and asked to determine the most relevant author personality traits.  \
                You ALWAYS return the result as a json object that contains the following keys, Ensure \
                    the output is a valid JSON string without any extra text or formatting, not markdown just plain json: \
                - prompt: Tailored prompt text describing relevant author details, this should be very detailed and also optimally consumable by an LLM \
                - style_guide: Writing style recommendations \
                - key_experiences: List of specific experiences to reference \
                - credentials: Relevant credentials to establish authority \
                This is the authors personality info: {self.personality}"
            messages = [HumanMessage(content=prompt), SystemMessage(content=sysprompt)]
            response = self.llm_anthropic.invoke(messages)
            self.logger.info(f"Response received, checking for validity")

            if isinstance(response, AIMessage):
                self.logger.info("Got personality response from LLM")
                self.logger.debug(f"Response OK: {response.content}")
                return response.content
            else:
                self.logger.error(f"Unexpected response type: {type(response)}")
                raise ValueError("LLM returned unexpected response type")
                
        except Exception as e:
            self.logger.error(f"Failed to analyze outline: {str(e)}")
            raise
