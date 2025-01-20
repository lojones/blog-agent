from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from utils.logger import setup_logger
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from agent.data_class.blog_data import BlogOutline, BlogOutlineEvaluation, BlogOutlineSimple, BlogArticle, BlogArticleEvaluation
from utils.github_reader import GithubReader
load_dotenv()


class Evaluator:
    def __init__(self):
        self.name = "Evaluator"
        self.logger = setup_logger("Evaluator")
        self.logger.info("Initializing Evaluator")
        self.llm_google = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0.7)
        self.github_reader = GithubReader()
        # self.llm_google_structured = self.llm_google.with_structured_output(BlogOutlineEvaluation)
        self.system_prompt_outline_url = "https://github.com/lojones/blog-agent-data/blob/main/system-prompts/evaluate-outline.md"
        self.system_prompt_outline_text = self.github_reader.read_file(self.system_prompt_outline_url)
        self.system_prompt_outline_obj = SystemMessage(content=self.system_prompt_outline_text)
        self.system_prompt_article_url = "https://github.com/lojones/blog-agent-data/blob/main/system-prompts/evaluate-article.md"
        self.system_prompt_article_text = self.github_reader.read_file(self.system_prompt_article_url)
        self.system_prompt_article_obj = SystemMessage(content=self.system_prompt_article_text)


    def test_google_gemini(self):
        messages = [
            HumanMessage(content="What is the capital of France?")
        ]
        response = self.llm_google.invoke(messages)
        print(response)

    def get_initial_outline_prompt(self, outline: BlogOutlineSimple) -> HumanMessage:
        first : str = "Evaluate the following article outline, and tell me if it is good to go or needs more work.  \
                         The first word of your response should be YES or NO to indicate if the outline is good to go or needs more work. \
                         If it needs more work give me that detailed evaluation feedback.  \
                         DO NOT penalize the outline for not being detailed enough because this is the outline of the article not the full article. \
                         : " + outline.model_dump_json()
        return HumanMessage(content=first)
    
    def get_revision_outline_prompt(self, outline: BlogOutlineSimple) -> HumanMessage:
        revision : str = "This is the revised outline based on your previous feedback. Evaluate it again and tell if if its good to go or not. \
                            If its still not good enough provide more detailed feedback on what to improve. \
                         DO NOT penalize the outline for not being detailed enough because this is the outline of the article not the full article. \
                         : " + outline.model_dump_json()
        return HumanMessage(content=revision)

    def get_initial_article_prompt(self, article_text: str) -> HumanMessage:
        first : str = "Evaluate the following article, and tell me if it is good to go or needs more work.  \
                         The first word of your response should be YES or NO to indicate if the article is good to go or needs more work. \
                         If it needs more work give me that detailed evaluation feedback.  \
                         : " + article_text
        return HumanMessage(content=first)
    

    def evaluate_outline(self, outline: BlogOutline) -> BlogOutline:
        """
        Evaluates the outline of the blog post
        """
        self.logger.info("Evaluator: Evaluating outline")
        eval : BlogOutlineEvaluation = outline.outline_evaluation
        if len(eval.messages) == 0:
            eval.messages = [self.system_prompt_outline_obj, self.get_initial_outline_prompt(outline.outline)]
        else:
            eval.messages.append(self.get_revision_outline_prompt(outline.outline))

        try:
            response = self.llm_google.invoke(eval.messages)
            eval.messages.append(AIMessage(content=response.content))
            # Check if response starts with 'yes' (case insensitive)
            is_good_to_go = response.content.lower().startswith('yes')
            outline.outline_evaluation.iteration_number += 1
            outline.outline_evaluation.evaluation = response.content
            outline.outline_evaluation.good_to_go = is_good_to_go

            self.logger.debug(f"Evaluator: Outline evaluation: {outline.outline_evaluation.model_dump_json()}")
            
            return outline
        except Exception as e:
            self.logger.error(f"Evaluator: Error evaluating outline: {str(e)}")
            raise


    def evaluate_article(self, article: BlogArticle) -> BlogArticle:
        """
        Evaluates the article
        """
        self.logger.info("Evaluator: Evaluating article")
        article_text = article.article_text
        eval : BlogArticleEvaluation = article.article_evaluation
        if len(eval.messages) == 0:
            eval.messages = [self.system_prompt_article_obj, self.get_initial_article_prompt(article_text)]
        else:
            eval.messages.append(self.get_revision_outline_prompt(article_text))

        try:
            response = self.llm_google.invoke(eval.messages)
            eval.messages.append(AIMessage(content=response.content))
            is_good_to_go = response.content.lower().startswith('yes')
            article.article_evaluation.iteration_number += 1
            article.article_evaluation.evaluation = response.content
            article.article_evaluation.good_to_go = is_good_to_go
            return article
        except Exception as e:
            self.logger.error(f"Evaluator: Error evaluating article: {str(e)}")
            raise

