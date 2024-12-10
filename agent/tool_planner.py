from typing import TypedDict, List, Optional
from utils.logger import setup_logger
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from utils.github_reader import GithubReader
from agent.data_class.blog_data import BlogAgentState
from pydantic import BaseModel, Field
from utils.envvars import LLM_CLAUDE_SONNET, ANTHROPIC_API_KEY
from pydantic import BaseModel, Field
from agent.data_class.blog_data import ExecutionPlan

logger = setup_logger("PlannerTool")



class PlannerTool:
    """Tool for creating and revising execution plans"""
    
    def __init__(self):
        self.logger = logger
        self.github_reader = GithubReader()
        self.llm = ChatAnthropic(
            model=LLM_CLAUDE_SONNET,
            api_key=ANTHROPIC_API_KEY,
            temperature=0.3  # Lower temperature for more consistent planning
        ).with_structured_output(ExecutionPlan)
        
        # Load system prompts
        self.create_plan_prompt = self.github_reader.read_file(
            "https://github.com/lojones/blog-agent-data/blob/main/create-execution-plan.md"
        )


    def create_execution_plan(self, instructions: str) -> ExecutionPlan:
        """
        Creates an execution plan based on instructions.
        
        Args:
            instructions (str): User's instructions for the blog post
            
        Returns:
            ExecutionPlan: Structured plan with steps and dependencies
        """
        try:
            self.logger.info("Creating execution plan")
            
            prompt = f"""
            Create a detailed execution plan for writing a blog post based on these instructions:

            INSTRUCTIONS:
            {instructions}

            """
            
            messages = [
                SystemMessage(content=self.create_plan_prompt),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(messages)
            
            plan : ExecutionPlan = response

            self.logger.info("Successfully created execution plan")
            self.logger.debug(f"Plan: {plan}")
            return plan
   
                
        except Exception as e:
            self.logger.error(f"Failed to create execution plan: {str(e)}")
            raise
