from typing import TypedDict, Optional
from utils.logger import setup_logger
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from utils.github_reader import GithubReader
from utils.envvars import LLM_CLAUDE_SONNET
from utils.envvars import ANTHROPIC_API_KEY
from agent.data_class.blog_data import ArticleFraming

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
  
            

    def derive_content_plan(self, instructions: str) -> ArticleFraming:
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
            plan = tool.derive_content_plan(
                "Write about AI safety, focusing on current challenges"
            )
        """
        try:
            self.logger.info("Deriving content plan from instructions")
           
            prompt = f"""
            Please analyze these instructions and create a content plan:
            INSTRUCTIONS:
            {instructions}

            Create a plan that includes:
            1. The main topic
            2. An engaging title
            3. A detailed outline following this structure:
                - TITLE: <title>
                - OBJECTIVE: <informing|persuading|entertaining>
                - THESIS: <main point>
                - KEY POINTS: (bullet points)
                - STRUCTURE:
                    INTRO: <intro point>
                    BODY: <4 body points>
                    CONCLUSION: <conclusion point>
            """
           
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(messages)
           
            if isinstance(response, AIMessage):
                # Parse the response into components
                content = response.content
               
                # Extract topic, title, and outline
                # (Assuming the LLM returns them in a structured format)
                lines = content.split('\n')
                topic = next(line for line in lines if line.startswith('TOPIC:')).replace('TOPIC:', '').strip()
                title = next(line for line in lines if line.startswith('TITLE:')).replace('TITLE:', '').strip()
                outline = '\n'.join(lines[lines.index('OUTLINE:'):]) if 'OUTLINE:' in lines else content
               
                result = ArticleFraming(
                    topic=topic,
                    title=title,
                    outline=outline
                )
               
                self.logger.info(f"Successfully created content plan for: {topic}")
                return result
               
            else:

                raise ValueError(f"Unexpected response type: {type(response)}")
                

        except Exception as e:

            self.logger.error(f"Failed to derive content plan: {str(e)}")
            raise

