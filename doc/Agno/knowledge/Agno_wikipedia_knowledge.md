Knowledge
Wikipedia KnowledgeBase
The WikipediaKnowledgeBase reads wikipedia topics, converts them into vector embeddings and loads them to a vector database.

​
Usage
We are using a local PgVector database for this example. Make sure it’s running


Copy
pip install wikipedia
knowledge_base.py

Copy
from agno.knowledge.wikipedia import WikipediaKnowledgeBase
from agno.vectordb.pgvector import PgVector

knowledge_base = WikipediaKnowledgeBase(
    topics=["Manchester United", "Real Madrid"],
    # Table name: ai.wikipedia_documents
    vector_db=PgVector(
        table_name="wikipedia_documents",
        db_url="postgresql+psycopg://ai:ai@localhost:5532/ai",
    ),
)
Then use the knowledge_base with an Agent:

agent.py

Copy
from agno.agent import Agent
from knowledge_base import knowledge_base

agent = Agent(
    knowledge=knowledge_base,
    search_knowledge=True,
)
agent.knowledge.load(recreate=False)

agent.print_response("Ask me about something from the knowledge base")
​
Params
Parameter	Type	Default	Description
topics	List[str]	[]	Topics to read
WikipediaKnowledgeBase is a subclass of the AgentKnowledge class and has access to the same params.