import io
from flask import send_file

def showgraph(graph, logger, graph_name):
    try:
        img_data = graph.get_graph().draw_mermaid_png()
        
        # Write image data to file
        output_path = f"logs/{graph_name}.png"
        with open(output_path, "wb") as f:
            f.write(img_data)
        logger.info(f"Graph saved to {output_path}")
            
        return output_path
    except Exception as e:
        logger.error(f"Failed to display graph: {str(e)}")
        raise