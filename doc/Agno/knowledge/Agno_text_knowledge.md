Knowledge
Text Knowledge Base
The TextKnowledgeBase reads local txt files, converts them into vector embeddings and loads them to a vector database.

​
Usage
We are using a local PgVector database for this example. Make sure it’s running

knowledge_base.py

Copy
from agno.knowledge.text import TextKnowledgeBase
from agno.vectordb.pgvector import PgVector

knowledge_base = TextKnowledgeBase(
    path="data/txt_files",
    # Table name: ai.text_documents
    vector_db=PgVector(
        table_name="text_documents",
        db_url="postgresql+psycopg://ai:ai@localhost:5532/ai",
    ),
)
Then use the knowledge_base with an Agent:

agent.py

Copy
from agno.agent import Agent
from knowledge_base import knowledge_base

agent = Agent(
    knowledge_base=knowledge_base,
    search_knowledge=True,
)
agent.knowledge.load(recreate=False)

agent.print_response("Ask me about something from the knowledge base")
​
Params
Parameter	Type	Default	Description
path	Union[str, Path]	-	Path to text files. Can point to a single text file or a directory of text files.
formats	List[str]	[".txt"]	Formats accepted by this knowledge base.
reader	TextReader	TextReader()	A TextReader that converts the text files into Documents for the vector database.
TextKnowledgeBase is a subclass of the AgentKnowledge class and has access to the same params.