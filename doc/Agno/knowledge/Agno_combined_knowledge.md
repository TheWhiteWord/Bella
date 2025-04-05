Combined KnowledgeBase
The CombinedKnowledgeBase combines multiple knowledge bases into 1 and is used when your app needs information using multiple sources.

​
Usage
We are using a local PgVector database for this example. Make sure it’s running


Copy
pip install pypdf bs4
knowledge_base.py

Copy
from agno.knowledge.combined import CombinedKnowledgeBase
from agno.vectordb.pgvector import PgVector
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.knowledge.website import WebsiteKnowledgeBase
from agno.knowledge.pdf import PDFKnowledgeBase


url_pdf_knowledge_base = PDFUrlKnowledgeBase(
    urls=["pdf_url"],
    # Table name: ai.pdf_documents
    vector_db=PgVector(
        table_name="pdf_documents",
        db_url="postgresql+psycopg://ai:ai@localhost:5532/ai",
    ),
)

website_knowledge_base = WebsiteKnowledgeBase(
    urls=["https://docs.agno.com/introduction"],
    # Number of links to follow from the seed URLs
    max_links=10,
    # Table name: ai.website_documents
    vector_db=PgVector(
        table_name="website_documents",
        db_url="postgresql+psycopg://ai:ai@localhost:5532/ai",
    ),
)

local_pdf_knowledge_base = PDFKnowledgeBase(
    path="data/pdfs",
    # Table name: ai.pdf_documents
    vector_db=PgVector(
        table_name="pdf_documents",
        db_url="postgresql+psycopg://ai:ai@localhost:5532/ai",
    ),
    reader=PDFReader(chunk=True),
)

knowledge_base = CombinedKnowledgeBase(
    sources=[
        url_pdf_knowledge_base,
        website_knowledge_base,
        local_pdf_knowledge_base,
    ],
    vector_db=PgVector(
        # Table name: ai.combined_documents
        table_name="combined_documents",
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
sources	List[AgentKnowledge]	[]	List of knowledge bases.
CombinedKnowledgeBase is a subclass of the AgentKnowledge class and has access to the same params.

​
