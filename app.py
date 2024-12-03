from flask import Flask
from agent.agent_captain import AgentCaptain
from utils.logger import setup_logger

app = Flask(__name__)
logger = setup_logger("BlogAgent")

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

if __name__ == '__main__':
    logger.info("Starting BlogAgent application")
    app.run(debug=True)