import os
from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub
from src.services.graph_builder.Indexes.index import Index


class VectorGraphChain:
    """Vector Graph Chain"""

    def __init__(
        self,
        provider: Literal['ollama', 'openai']
    ):
        if provider == 'ollama':
            self.llm = ChatOllama(
                model='llama3.1:8b',
                temperature=0,
                base_url=os.getenv("OLLAMA_CONNECT_URL")
            )
        elif provider == 'openai':
            self.llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0,
                api_key=os.environ.get("OPENAI_API_KEY"),
            )
        
        self._vector_index = Index().get_neo4j_vector_index()

    def get_vector_graph_chain(self):
        '''
        Create a Neo4j Retrieval QA Chain. Returns top K most relevant articles
        '''

        # vector_graph_chain = RetrievalQA.from_chain_type(
        #     llm, 
        #     chain_type="stuff", 
        #     retriever = self._vector_index.as_retriever(search_kwargs={'k':3}), 
        #     verbose=True,
        #     return_source_documents=True,
        # )
        retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
        question_answer_chain = create_stuff_documents_chain(
            self.llm,
            retrieval_qa_chat_prompt
        )

        vector_graph_chain = create_retrieval_chain(
            self._vector_index.as_retriever(search_kwargs={'k': 3}),
            question_answer_chain
        )

        return vector_graph_chain
