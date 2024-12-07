from flask import Flask, request, jsonify
from agent.agent_captain import AgentCaptain
from utils.logger import setup_logger

app = Flask(__name__)
logger = setup_logger("BlogAgent App")

try:
    captain = AgentCaptain()
    logger.info("Successfully initialized AgentCaptain")
except Exception as e:
    logger.error(f"Failed to initialize AgentCaptain: {str(e)}")

@app.route('/')
def hello_world():
    logger.info("Received request to /")
    return "Hello, World!"


@app.route('/showgraph')
def show_graph():
    logger.info("Received request to /showgraph")
    try:
        return captain.showgraph()
        
    except Exception as e:
        logger.error(f"Failed to show graph: {str(e)}")
        return f"Error displaying graph: {str(e)}", 500

@app.route('/create/blogpost', methods=['POST'])
def create_blogpost():
    logger.info("Received request to /create/blogpost")
    try:
        data = request.get_json()
        topic = data.get('topic')
        
        if not topic:
            logger.warning("No topic provided in request")
            return jsonify({"error": "Topic is required"}), 400
            
        result = captain.create_blogpost(topic)
        return jsonify({"result": result})
        
    except Exception as e:
        logger.error(f"Failed to create blog post: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting BlogAgent application")
    app.run(debug=True)