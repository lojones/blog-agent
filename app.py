from flask import Flask, request, jsonify
# from agent.agent_captain import AgentCaptain
from agent.data_class.blog_data import BAState
from agent.supervisor import Supervisor
from utils.logger import setup_logger
from agent.tool.evaluator import Evaluator

app = Flask(__name__)
logger = setup_logger("BlogAgent App")

try:
    # captain = AgentCaptain()
    supervisor = Supervisor()
    logger.info("Successfully initialized Supervisor")
except Exception as e:
    logger.error(f"Failed to initialize Supervisor: {str(e)}")

@app.route('/')
def hello_world():
    logger.info("Received request to /")
    return "Hello, World!"


@app.route('/showgraph')
def show_graph():
    logger.info("Received request to /showgraph")
    try:
        # return captain.showgraph()
        return supervisor.showgraph()
        
    except Exception as e:
        logger.error(f"Failed to show graph: {str(e)}")
        return f"Error displaying graph: {str(e)}", 500

@app.route('/showplan')
def show_plan():
    logger.info("Received request to /showplan")
    try:
        state = BAState(instructions="Write a blog post about the benefits of using AI to write blog posts")
        return supervisor.create_plan(state).plan.model_dump()
    except Exception as e:
        logger.error(f"Failed to show plan: {str(e)}")
        return f"Error displaying plan: {str(e)}", 500


@app.route('/create/blogpost', methods=['POST'])
def create_blogpost():
    logger.info("Received request to /create/blogpost")
    try:
        data = request.get_json()
        topic = data.get('topic')
        
        if not topic:
            logger.warning("No topic provided in request")
            return jsonify({"error": "Topic is required"}), 400
            
        # result = captain.create_blogpost(topic)
        result = supervisor.create_blogpost(topic)
        return jsonify({"result": result.model_dump()})
        
    except Exception as e:
        logger.error(f"Failed to create blog post: {str(e)}")
        return jsonify({"error": str(e)}), 500
    
@app.route('/test_google_gemini')
def test_google_gemini():
    logger.info("Received request to /test_google_gemini")
    evaluator = Evaluator()
    evaluator.test_google_gemini()
    return "Google Gemini test completed"


if __name__ == '__main__':
    logger.info("Starting BlogAgent application")
    app.run(debug=False)