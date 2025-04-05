Knowledge
Website Knowledge Base
The WebsiteKnowledgeBase reads websites, converts them into vector embeddings and loads them to a vector_db.

​
Usage
We are using a local PgVector database for this example. Make sure it’s running


Copy
pip install bs4
knowledge_base.py

Copy
from agno.knowledge.website import WebsiteKnowledgeBase
from agno.vectordb.pgvector import PgVector

knowledge_base = WebsiteKnowledgeBase(
    urls=["https://docs.agno.com/introduction"],
    # Number of links to follow from the seed URLs
    max_links=10,
    # Table name: ai.website_documents
    vector_db=PgVector(
        table_name="website_documents",
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
urls	List[str]	[]	URLs to read
reader	Optional[WebsiteReader]	None	A WebsiteReader that reads the urls and converts them into Documents for the vector database.
max_depth	int	3	Maximum depth to crawl.
max_links	int	10	Number of links to crawl.
WebsiteKnowledgeBase is a subclass of the AgentKnowledge class and has access to the same params.

​
