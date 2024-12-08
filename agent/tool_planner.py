from typing import TypedDict, List, Optional
from utils.logger import setup_logger
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from utils.github_reader import GithubReader
from agent.data_class.blog_data import BlogAgentState
from utils.envvars import LLM_CLAUDE_SONNET

logger = setup_logger("PlannerTool")

class ExecutionStep(TypedDict):
    step: str
    description: str
    dependencies: List[str]
    status: str  # 'pending', 'in_progress', 'completed', 'failed'

class ExecutionPlan(TypedDict):
    steps: List[ExecutionStep]
    current_step: int
    notes: str

class PlannerTool:
    """Tool for creating and revising execution plans"""
    
    def __init__(self):
        self.logger = logger
        self.github_reader = GithubReader()
        self.llm = ChatAnthropic(
            model=LLM_CLAUDE_SONNET,
            temperature=0.3  # Lower temperature for more consistent planning
        )
        
        # Load system prompts
        self.create_plan_prompt = self.github_reader.read_file(
            "https://github.com/lojones/blog-agent-data/blob/main/create-execution-plan.md"
        )
        self.revise_plan_prompt = self.github_reader.read_file(
            "https://github.com/lojones/blog-agent-data/blob/main/revise-execution-plan.md"
        )
        
        if not all([self.create_plan_prompt, self.revise_plan_prompt]):
            raise ValueError("Failed to load planning system prompts")

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

            The plan should include:
            1. Sequential steps with dependencies
            2. Clear success criteria for each step
            3. Potential challenges and mitigation strategies
            4. Resource requirements (e.g., research needs)

            Format the response as a JSON object with this structure:
            {{
                "steps": [
                    {{
                        "step": "step name",
                        "description": "detailed description",
                        "dependencies": ["dependent step names"],
                        "status": "pending"
                    }}
                ],
                "current_step": 0,
                "notes": "any additional planning notes"
            }}
            """
            
            messages = [
                SystemMessage(content=self.create_plan_prompt),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(messages)
            
            if isinstance(response, AIMessage):
                import json
                plan = json.loads(response.content)
                self.logger.info("Successfully created execution plan")
                self.logger.debug(f"Plan: {plan}")
                return plan
            else:
                raise ValueError(f"Unexpected response type: {type(response)}")
                
        except Exception as e:
            self.logger.error(f"Failed to create execution plan: {str(e)}")
            raise

    def revise_plan(self, 
                    instructions: str, 
                    current_plan: ExecutionPlan, 
                    state: BlogAgentState) -> ExecutionPlan:
        """
        Revises the execution plan based on current state and progress.
        
        Args:
            instructions (str): Original instructions
            current_plan (ExecutionPlan): Current execution plan
            state (BlogAgentState): Current state of the blog creation process
            
        Returns:
            ExecutionPlan: Revised plan with updated steps and status
        """
        try:
            self.logger.info("Revising execution plan")
            
            prompt = f"""
            Review and revise the execution plan based on current progress:

            ORIGINAL INSTRUCTIONS:
            {instructions}

            CURRENT PLAN:
            {current_plan}

            CURRENT STATE:
            {state}

            Requirements:
            1. Evaluate progress and identify any blockers
            2. Adjust steps based on what we've learned
            3. Add new steps if needed
            4. Update status of completed steps
            5. Provide reasoning for any changes

            Return the revised plan in the same JSON format as the original plan.
            """
            
            messages = [
                SystemMessage(content=self.revise_plan_prompt),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(messages)
            
            if isinstance(response, AIMessage):
                import json
                revised_plan = json.loads(response.content)
                self.logger.info("Successfully revised execution plan")
                self.logger.debug(f"Revised plan: {revised_plan}")
                return revised_plan
            else:
                raise ValueError(f"Unexpected response type: {type(response)}")
                
        except Exception as e:
            self.logger.error(f"Failed to revise execution plan: {str(e)}")
            raise 