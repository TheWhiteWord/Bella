"""Script to test search functionality and save results."""
import asyncio
import datetime
from pathlib import Path
from Bella.src.agents.search_agent import SearchAgent
from src.llm.chat_manager import generate_chat_response

async def test_search_and_save():
    """Test search functionality and save results."""
    query = "Search for the latest AI developments"
    
    # Generate timestamp for the filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create results directory if it doesn't exist
    results_dir = Path(__file__).parents[2] / "results" / "search_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Perform the search
    print(f"\nProcessing search query: {query}")
    response = await generate_chat_response(query, "", model="Lexi")
    
    # Save the results
    output_file = results_dir / f"search_results_{timestamp}.md"
    with open(output_file, "w") as f:
        f.write(f"# Search Results - Latest AI Developments\n")
        f.write(f"Query: {query}\n")
        f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Response:\n")
        f.write(response)
    
    print(f"\nResults saved to: {output_file}")
    print("\nSearch Response:")
    print("-" * 50)
    print(response)
    print("-" * 50)

if __name__ == "__main__":
    asyncio.run(test_search_and_save())