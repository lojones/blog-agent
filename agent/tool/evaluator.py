from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from utils.logger import setup_logger
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from agent.data_class.blog_data import BlogOutline, BlogOutlineEvaluation   
load_dotenv()


class Evaluator:
    def __init__(self):
        self.name = "Evaluator"
        self.logger = setup_logger("Evaluator")
        self.logger.info("Initializing Evaluator")
        self.llm_google = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0.7)
        # self.llm_google_structured = self.llm_google.with_structured_output(BlogOutlineEvaluation)


    def test_google_gemini(self):
        messages = [
            HumanMessage(content="What is the capital of France?")
        ]
        response = self.llm_google.invoke(messages)
        print(response)

    def evaluate_outline(self, outline: BlogOutline) -> BlogOutline:
        """
        Evaluates the outline of the blog post
        """
        self.logger.info("Evaluator: Evaluating outline")
        messages = [
            HumanMessage(content="Evaluate the following article outline, and tell me if it is good to go or needs more work.  \
                         The first word of your response should be YES or NO to indicate if the outline is good to go or needs more work. \
                         If it needs more work give me that detailed evaluation feedback.  Use the following guidance \
                         to evaluate the outline - \
                         1. Does the short title grab attention? \
                         2. Is the full title and subtitle clear and descriptive? \
                         3. Does the intro set the stage effectively, does it explain the context, does it hook the readers interest? \
                         4. Does the body present a logical flow of points? does each bullet add new value? is the structure cohesive? \
                         5. Does the conclusion wrap up and offer a call to action? Does it summarize main ideas? does it give next steps? \
                         If the answer to all those questions is yes then the outline is good to go. Otherwise the outline needs more work, \
                         and provide details feedback about how to improve the outline so all those questions will be answered. \
                         Remember that you're evaluating the outline, not the actual article so it won't be detailed but thats ok. \
                         DO NOT penalize the outline for not being detailed. \
                         : " + outline.outline.model_dump_json())
        ]
        response = self.llm_google.invoke(messages)
        # Check if response starts with 'yes' (case insensitive)
        is_good_to_go = response.content.lower().startswith('yes')
        
        outline.outline_evaluation.iteration_number += 1
        outline.outline_evaluation.evaluation = response.content
        outline.outline_evaluation.good_to_go = is_good_to_go
        
        return outline
