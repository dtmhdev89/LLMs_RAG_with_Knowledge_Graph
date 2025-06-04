from src.services.graph_builder.Graph.graph import Graph


if __name__ == "__main__":
    graph_app = Graph().graph
    
    png_bytes = graph_app.get_graph().draw_mermaid_png()

    with open("langgraph_diagram.png", "wb") as f:
        f.write(png_bytes)
