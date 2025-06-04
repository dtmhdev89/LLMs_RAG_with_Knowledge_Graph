# LLMs_RAG_with_Knowledge_Graph
Small tutorial project for RAG with knowledge graph

## Description:
- Using local Ollama for development;
- Using Neo4j Aura Graph Database to store graph documents from data source;
- Try to construct graph documents by manual functions and LLMs way;
- Build a terminal Q&A from graph database;
- Build an agentic AI using LangGraph;

## How to run:
- Create a virtualenv if needed. Then activate it.
- Install libs by requirements.txt;
- Prepare an online Neo4j Aura instance or a local one with docker;
- Create .env in the project folder: LLMs_RAG_with_Knowledge_Graph/.env
- Use below command at the project folder to run app:
  `python3 -m src.<file_name_without_py>`

  e.g: `python3 -m src.simple_cli_app`

- .env keys:
```
OLLAMA_CONNECT_URL=<your_ollama_uri>
NEO4J_URI=<your_neo4j_database_instance>
NEO4J_USERNAME=<your_neo4j_username>
NEO4J_PASSWORD=<your_neo4j_password>
OPENAI_API_KEY=<your_open_ai_key>
```

## Future implement:
- The data sources for **agentic_ai app with LangGraph** isn't ready since the source of data change. Will implement with the change in near future. Or maybe replace with new data sources and make code changes. Anyway, the app still run but have no result at all.
