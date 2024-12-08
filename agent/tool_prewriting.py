from typing import TypedDict, Optional
from utils.logger import setup_logger
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from utils.github_reader import GithubReader
from utils.envvars import LLM_CLAUDE_SONNET

logger = setup_logger("PrewritingTool")

class PrewritingOutput(TypedDict):
    topic: str
    title: str
    outline: str

class PrewritingTool:
    """Tool for deriving topic, title, and outline from instructions"""
    
    def __init__(self):
        self.logger = logger
        self.github_reader = GithubReader()
        self.llm = ChatAnthropic(
            model=LLM_CLAUDE_SONNET,
            temperature=0.7
        )
        
        # Load system prompts from GitHub
        self.system_prompt = self.github_reader.read_file(
            "https://github.com/lojones/blog-agent-data/blob/main/prewriting.md"
        )
        if not self.system_prompt:
            raise ValueError("Failed to load prewriting system prompt")

    
            
    def derive_content_plan(self, instructions: str) -> PrewritingOutput:
        """
        Analyzes instructions to determine topic, title, and outline.
        
        Args:
            instructions (str): User's instructions or requirements for the blog post
            
        Returns:
            PrewritingOutput containing:
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
                
                result = PrewritingOutput(
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

    def derive_topic(self, instructions: str) -> str:
        """
        Analyzes instructions to determine the main topic.
        
        Args:
            instructions (str): User's instructions or requirements for the blog post
            
        Returns:
            str: The main topic extracted from instructions
            
        Example:
            tool = PrewritingTool()
            topic = tool.derive_topic(
                "Write about how AI is changing software development, focus on practical impacts"
            )
        """
        try:
            self.logger.info("Deriving topic from instructions")
            
            prompt = f"""
            Please analyze these instructions and extract the main topic:

            INSTRUCTIONS:
            {instructions}

            Requirements:
            1. Extract the core topic, not just surface keywords
            2. Make it specific enough to guide research but broad enough to allow exploration
            3. Format the response exactly like this:
               TOPIC: <the topic>
               REASONING: <brief explanation of why this is the core topic>

            Example:
            Input: "Write about how ChatGPT is being used by developers, focusing on productivity"
            Output:
            TOPIC: AI-Powered Developer Productivity Tools
            REASONING: While ChatGPT is mentioned specifically, the core topic is about how AI tools are enhancing developer productivity, which allows for broader exploration of the theme.
            """
            
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(messages)
            
            if isinstance(response, AIMessage):
                content = response.content
                
                # Extract just the topic line
                lines = content.split('\n')
                topic_line = next(line for line in lines if line.startswith('TOPIC:'))
                topic = topic_line.replace('TOPIC:', '').strip()
                
                self.logger.info(f"Successfully derived topic: {topic}")
                self.logger.debug(f"Full response: {content}")  # Log the reasoning too
                return topic
                
            else:
                raise ValueError(f"Unexpected response type: {type(response)}")
                
        except Exception as e:
            self.logger.error(f"Failed to derive topic: {str(e)}")
            raise
