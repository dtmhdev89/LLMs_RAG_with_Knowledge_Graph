from src.services.graph_builder.Graph.graph import Graph


if __name__ == "__main__":
    graph_app = Graph().graph
    
    png_bytes = graph_app.get_graph().draw_mermaid_png()

    with open("langgraph_diagram.png", "wb") as f:
        f.write(png_bytes)

    graph_qa_result = graph_app.invoke(
        {"question": "find top 5 cited articles and return their title"}
    )

    print(graph_qa_result['documents'])
    result = graph_app.invoke(
        {"question": "find articles about oxidative stress. Return the title of the most relevant article"}
    )

    print(result.keys())
    print(result['subqueries'])
    print(result['documents'])

