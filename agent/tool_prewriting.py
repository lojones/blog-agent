from typing import TypedDict, Optional
from utils.logger import setup_logger
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from utils.github_reader import GithubReader
from utils.envvars import LLM_CLAUDE_SONNET
from utils.envvars import ANTHROPIC_API_KEY
from agent.data_class.blog_data import ArticleFraming
from langchain_core.tools import tool

logger = setup_logger("PrewritingTool")

class PrewritingTool:
    """Tool for deriving topic, title, and outline from instructions"""
    
    def __init__(self):
        self.logger = logger
        self.github_reader = GithubReader()
        self.llm = ChatAnthropic(
            model=LLM_CLAUDE_SONNET,
            temperature=0.7,
            api_key=ANTHROPIC_API_KEY
        ).with_structured_output(ArticleFraming)
        
        # Load system prompts from GitHub
        self.system_prompt = self.github_reader.read_file(

            "https://github.com/lojones/blog-agent-data/blob/main/prewriting.md"
        )

        if not self.system_prompt:
            raise ValueError("Failed to load prewriting system prompt")
  
            
    @tool
    def derive_prewriting(self, instructions: str) -> ArticleFraming:
        """
        Analyzes instructions to determine topic, title, and outline.
       
        Args:
            instructions (str): User's instructions or requirements for the blog post
            
        Returns:
            ArticleFraming containing:
                - topic: Main topic extracted from instructions
                - title: Engaging title for the blog post
                - outline: Structured outline including key points
                
        Example:
            tool = PrewritingTool()
            plan = tool.derive_prewriting(
                "Write about AI safety, focusing on current challenges"
            )
        """
        try:
            self.logger.info("Deriving content plan from instructions")
           
            prompt = f"""
            Please analyze these instructions and create prewriting artifacts:
            INSTRUCTIONS:
            {instructions}

            """
           
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(messages)
           
               
            self.logger.info(f"Successfully created content plan for: {instructions}")
            return response
               
            
                
        except Exception as e:

            self.logger.error(f"Failed to derive content plan: {str(e)}")
            raise

