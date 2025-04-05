VectorDbs
ChromaDB Agent Knowledge
​
Setup

Copy
pip install chromadb
​
Example
agent_with_knowledge.py

Copy
import typer
from rich.prompt import Prompt
from typing import Optional

from agno.agent import Agent
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.vectordb.chroma import ChromaDb

knowledge_base = PDFUrlKnowledgeBase(
    urls=["https://agno-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    vector_db=ChromaDb(collection="recipes"),
)

def pdf_agent(user: str = "user"):
    run_id: Optional[str] = None

    agent = Agent(
        run_id=run_id,
        user_id=user,
        knowledge_base=knowledge_base,
        use_tools=True,
        show_tool_calls=True,
        debug_mode=True,
    )
    if run_id is None:
        run_id = agent.run_id
        print(f"Started Run: {run_id}\n")
    else:
        print(f"Continuing Run: {run_id}\n")

    while True:
        message = Prompt.ask(f"[bold] :sunglasses: {user} [/bold]")
        if message in ("exit", "bye"):
            break
        agent.print_response(message)

if __name__ == "__main__":
    # Comment out after first run
    knowledge_base.load(recreate=False)

    typer.run(pdf_agent)
​
ChromaDb Params
Parameter	Type	Default	Description
collection	str	-	The name of the collection to use.
embedder	Embedder	OpenAIEmbedder()	The embedder to use for embedding document contents.
distance	Distance	cosine	The distance metric to use.
path	str	"tmp/chromadb"	The path where ChromaDB data will be stored.
persistent_client	bool	False	Whether to use a persistent ChromaDB client.