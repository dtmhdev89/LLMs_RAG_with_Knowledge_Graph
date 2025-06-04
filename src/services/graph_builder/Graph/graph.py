from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from src.services.graph_builder.Graph.state import GraphState
from src.services.graph_builder.Graph.node_labels import NodeLabels
from src.services.graph_builder.Chains.question_router import QuestionRouter
from src.services.graph_builder.Graph.nodes import Nodes

load_dotenv()


class Graph:
    """Graph flow"""

    def __init__(self) -> None:
        self._nodes = Nodes()
        self.graph = self._create_graph()
        self._question_router = QuestionRouter(
            provider='ollama',
            model_name='llama3.1:8b'
        ).build_router()

    def _create_graph(self):
        """Create langgraph graph"""
        graph = StateGraph(GraphState)
        # nodes for graph qa
        graph.add_node(NodeLabels.PROMPT_TEMPLATE, self._nodes.prompt_template)
        graph.add_node(NodeLabels.GRAPH_QA, self._nodes.graph_qa)
        # node for graph qa with vector search
        graph.add_node(NodeLabels.DECOMPOSER, self._nodes.decomposer)
        graph.add_node(NodeLabels.VECTOR_SEARCH, self._nodes.vector_search)
        graph.add_node(NodeLabels.PROMPT_TEMPLATE_WITH_CONTEXT, self._nodes.prompt_template_with_context)
        graph.add_node(NodeLabels.GRAPH_QA_WITH_CONTEXT, self._nodes.graph_qa_with_context)

        # set conditional entrypoint for vector search or graph qa
        graph.set_conditional_entry_point(
            self._route_question,
            {
                'decomposer': NodeLabels.DECOMPOSER,
                'prompt_template': NodeLabels.PROMPT_TEMPLATE
            }
        )

        # Edges for the graph with qa vector search
        graph.add_edge(NodeLabels.DECOMPOSER, NodeLabels.VECTOR_SEARCH)
        graph.add_edge(NodeLabels.VECTOR_SEARCH, NodeLabels.PROMPT_TEMPLATE_WITH_CONTEXT)
        graph.add_edge(NodeLabels.PROMPT_TEMPLATE_WITH_CONTEXT, NodeLabels.GRAPH_QA_WITH_CONTEXT)
        graph.add_edge(NodeLabels.GRAPH_QA_WITH_CONTEXT, END)
        # Edges for graph qa
        graph.add_edge(NodeLabels.PROMPT_TEMPLATE, NodeLabels.GRAPH_QA)
        graph.add_edge(NodeLabels.GRAPH_QA, END)

        return graph.compile()

    def _route_question(self, state: GraphState):
        print("---ROUTE QUESTION---")
        question = state["question"]
        source = self._question_router.invoke({"question": question})
        if source.datasource == "vector search":
            print("---ROUTE QUESTION TO VECTOR SEARCH---")
            return "decomposer"
        elif source.datasource == "graph query":
            print("---ROUTE QUESTION TO GRAPH QA---")
            return "prompt_template"
