import io
from flask import send_file
import os
from datetime import datetime

def showgraph(graph, logger, graph_name):
    try:
        img_data = graph.get_graph(xray=2).draw_mermaid_png()
        timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        
        # Write image data to file
        output_path = f"logs/{timestamp}-{graph_name}.png"
        with open(output_path, "wb") as f:
            f.write(img_data)
        logger.info(f"Graph saved to {output_path}")
            
        return output_path
    except Exception as e:
        logger.error(f"Failed to display graph: {str(e)}")
        raise

def write_to_file(content: any, filename: str, logger) -> str:
    """
    Writes content to a file in the logs directory with timestamp.
    
    Args:
        content: Content to write
        filename: Base name of the file
        logger: Logger instance for error handling
        
    Returns:
        str: Path to the written file
    """
    try:
        extension = "txt"
        if type(content) == str:
            extension = "txt"
        elif hasattr(content, 'model_dump_json'):
            extension = "json"
            content = content.model_dump_json()
        # Ensure logs directory exists
        if not os.path.exists('logs'):
            os.makedirs('logs')
        # Convert any literal \n strings to actual newlines
        content = content.replace('\\n', '\n')
        # Add timestamp to filename
        timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        filename_with_timestamp = f"{timestamp}-{filename}.{extension}"
        
        filepath = os.path.join('logs', filename_with_timestamp)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
            
        logger.info(f"Content written to {filepath}")
        return filepath
        
    except Exception as e:
        logger.error(f"Failed to write to file: {str(e)}")
        raise