#!/usr/bin/env python3
"""
ChromaDB Inspection Tool for Bella memory system.

This script provides utilities to inspect, analyze, and visualize the contents
of the ChromaDB vector database used for semantic memory storage in Bella.

Usage:
    python chroma_inspect.py [command] [options]

Commands:
    list        - List all memory entries in the database
    stats       - Display statistics about the database
    search      - Perform a semantic search
    dump        - Export all database contents to a file
    visualize   - Generate a visualization of memory embeddings
"""

import os
import sys
import json
import argparse
import asyncio
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import chromadb
from chromadb.utils import embedding_functions
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tabulate import tabulate

# Use absolute paths for ChromaDB to ensure consistency
import os.path
CHROMA_DB_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "memories", "chroma_db"))
CHROMA_COLLECTION_NAME = "bella_memories"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("chroma_inspect")


class ChromaInspector:
    """Tool for inspecting and analyzing ChromaDB database contents."""
    
    def __init__(self, db_path: str = CHROMA_DB_PATH, collection_name: str = CHROMA_COLLECTION_NAME):
        """Initialize the ChromaDB inspector.
        
        Args:
            db_path: Path to the ChromaDB database directory
            collection_name: Name of the collection to inspect
        """
        self.db_path = db_path
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        
    def connect(self) -> bool:
        """Connect to the ChromaDB database.
        
        Returns:
            bool: Success status
        """
        try:
            logger.info(f"Connecting to ChromaDB at {self.db_path}")
            self.client = chromadb.PersistentClient(path=self.db_path)
            
            # Get collection
            collections = self.client.list_collections()
            collection_names = [c.name for c in collections]
            
            if self.collection_name not in collection_names:
                logger.error(f"Collection '{self.collection_name}' not found.")
                logger.info(f"Available collections: {collection_names}")
                return False
                
            self.collection = self.client.get_collection(name=self.collection_name)
            count = self.collection.count()
            logger.info(f"Connected to collection '{self.collection_name}'. Items: {count}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to ChromaDB: {e}")
            return False
    
    def list_entries(self, limit: int = 100, offset: int = 0) -> pd.DataFrame:
        """List all entries in the collection.
        
        Args:
            limit: Maximum number of entries to return
            offset: Offset for pagination
            
        Returns:
            DataFrame containing the entries
        """
        if not self.collection:
            if not self.connect():
                return pd.DataFrame()
        
        try:
            # Get all entries
            results = self.collection.get(limit=limit, offset=offset, include=['metadatas', 'embeddings'])
            
            # Create a DataFrame
            data = []
            for i, memory_id in enumerate(results['ids']):
                metadata = results['metadatas'][i] if i < len(results['metadatas']) else {}
                
                # Convert embedding to dimensionality and norm for display
                embedding = results['embeddings'][i] if i < len(results['embeddings']) else None
                
                # Safe handling of embedding dimensions and norm
                if embedding is not None:
                    embedding_dim = len(embedding)
                    # Use explicit float cast for scalar value to avoid numpy array truth value ambiguity
                    embedding_norm = float(np.linalg.norm(embedding))
                    embedding_norm_rounded = round(embedding_norm, 4)
                else:
                    embedding_dim = 0
                    embedding_norm_rounded = 0.0
                
                entry = {
                    'id': memory_id,
                    'file_path': metadata.get('file_path', 'Unknown'),
                    'title': metadata.get('title', 'Untitled'),
                    'memory_type': metadata.get('memory_type', 'Unknown'),
                    'created_at': metadata.get('created_at', 'Unknown'),
                    'embedding_dimensions': embedding_dim,
                    'embedding_norm': embedding_norm_rounded
                }
                
                # Add any other metadata keys we find
                for key, value in metadata.items():
                    if key not in entry:
                        entry[key] = value
                
                data.append(entry)
                
            df = pd.DataFrame(data)
            return df
            
        except Exception as e:
            logger.error(f"Error listing entries: {e}")
            return pd.DataFrame()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection.
        
        Returns:
            Dict containing statistics
        """
        if not self.collection:
            if not self.connect():
                return {}
        
        try:
            # Get all entries for analysis
            results = self.collection.get(include=['metadatas', 'embeddings'])
            
            if not results['ids'] or len(results['ids']) == 0:
                return {"count": 0, "message": "No entries found"}
            
            # Basic stats
            count = len(results['ids'])
            
            # Memory types
            memory_types = {}
            for metadata in results['metadatas']:
                memory_type = metadata.get('memory_type', 'Unknown')
                memory_types[memory_type] = memory_types.get(memory_type, 0) + 1
            
            # Dates
            dates = []
            for metadata in results['metadatas']:
                date_str = metadata.get('created_at', '')
                if date_str:
                    try:
                        date = datetime.fromisoformat(date_str)
                        dates.append(date)
                    except (ValueError, TypeError):
                        pass
            
            oldest = min(dates) if dates else None
            newest = max(dates) if dates else None
            date_range = (newest - oldest).days if oldest and newest else 0
            
            # Embedding dimensions - properly handle numpy arrays
            embedding_dims = 0
            if len(results['embeddings']) > 0:
                # Safely get the first embedding's dimensions
                first_embedding = results['embeddings'][0]
                if isinstance(first_embedding, (list, np.ndarray)):
                    embedding_dims = len(first_embedding)
            
            # File existence check
            existing_files = 0
            missing_files = 0
            for metadata in results['metadatas']:
                file_path = metadata.get('file_path', '')
                # Check if path exists and is not empty
                if isinstance(file_path, str) and file_path.strip() and os.path.exists(file_path):
                    existing_files += 1
                elif isinstance(file_path, str) and file_path.strip():
                    missing_files += 1
            
            stats_dict = {
                "count": count,
                "memory_types": memory_types,
                "oldest_entry": oldest.isoformat() if oldest else "Unknown",
                "newest_entry": newest.isoformat() if newest else "Unknown",
                "date_range_days": date_range,
                "embedding_dimensions": embedding_dims,
                "existing_files": existing_files,
                "missing_files": missing_files
            }
            return stats_dict
            
        except Exception as e:
            logger.error(f"Error getting stats: {e}", exc_info=True)
            return {"error": str(e), "count": 0}
    
    def search(self, query: str, n_results: int = 5) -> pd.DataFrame:
        """Perform a semantic search.
        
        Args:
            query: Search query
            n_results: Number of results to return
            
        Returns:
            DataFrame containing search results
        """
        if not self.collection:
            if not self.connect():
                return pd.DataFrame()
        
        try:
            # Import only if needed to avoid circular import
            sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
            from memory.enhanced_memory import EnhancedMemoryProcessor
            
            # Create processor and generate embedding
            processor = EnhancedMemoryProcessor()
            
            # Generate embedding (run in asyncio event loop)
            loop = asyncio.get_event_loop()
            embedding = loop.run_until_complete(processor.generate_embedding(query))
            
            if not embedding:
                logger.error("Failed to generate embedding for search query")
                return pd.DataFrame()
            
            # Query ChromaDB
            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=n_results,
                include=['metadatas', 'distances']
            )
            
            # Process results
            data = []
            ids = results.get('ids', [[]])[0]
            distances = results.get('distances', [[]])[0]
            metadatas = results.get('metadatas', [[]])[0]
            
            for i in range(len(ids)):
                memory_id = ids[i]
                distance = distances[i]
                metadata = metadatas[i] if i < len(metadatas) else {}
                
                # Convert distance to similarity score
                similarity_score = max(0.0, 1.0 - distance)
                
                entry = {
                    'id': memory_id,
                    'similarity': round(similarity_score, 4),
                    'distance': round(distance, 4),
                    'file_path': metadata.get('file_path', 'Unknown'),
                    'title': metadata.get('title', 'Untitled'),
                    'memory_type': metadata.get('memory_type', 'Unknown'),
                    'created_at': metadata.get('created_at', 'Unknown'),
                }
                
                # Check if file exists
                file_exists = os.path.exists(entry['file_path']) if entry['file_path'] != 'Unknown' else False
                entry['file_exists'] = file_exists
                
                # Add file content preview if it exists
                if file_exists:
                    try:
                        with open(entry['file_path'], 'r') as f:
                            content = f.read(500)  # First 500 chars
                            entry['content_preview'] = content + ('...' if len(content) >= 500 else '')
                    except Exception as e:
                        entry['content_preview'] = f"Error reading file: {e}"
                else:
                    entry['content_preview'] = "File not found"
                
                data.append(entry)
            
            df = pd.DataFrame(data)
            return df
            
        except Exception as e:
            logger.error(f"Error performing search: {e}", exc_info=True)
            return pd.DataFrame()
    
    def dump_to_json(self, output_path: str) -> bool:
        """Dump all database contents to a JSON file.
        
        Args:
            output_path: Path to save the JSON file
            
        Returns:
            Success status
        """
        if not self.collection:
            if not self.connect():
                return False
        
        try:
            # Get all entries
            results = self.collection.get(include=['metadatas'])
            
            # Convert to serializable format
            output = {
                "collection_name": self.collection_name,
                "count": len(results['ids']),
                "entries": []
            }
            
            for i, memory_id in enumerate(results['ids']):
                metadata = results['metadatas'][i] if i < len(results['metadatas']) else {}
                
                entry = {
                    "id": memory_id,
                    "metadata": metadata
                }
                
                # Check if file exists and add content if it does
                file_path = metadata.get('file_path', '')
                if file_path and os.path.exists(file_path):
                    try:
                        with open(file_path, 'r') as f:
                            entry["content"] = f.read()
                    except Exception as e:
                        entry["content_error"] = str(e)
                else:
                    entry["content"] = "File not found"
                
                output["entries"].append(entry)
            
            # Write to file
            with open(output_path, 'w') as f:
                json.dump(output, f, indent=2)
            
            logger.info(f"Dumped {len(output['entries'])} entries to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error dumping to JSON: {e}")
            return False
    
    def visualize_embeddings(self, output_path: str = None, method: str = 'pca') -> None:
        """Generate a visualization of memory embeddings.
        
        Args:
            output_path: Path to save the visualization image
            method: Dimensionality reduction method ('pca' or 'tsne')
        """
        if not self.collection:
            if not self.connect():
                return
        
        try:
            # Get all entries with embeddings
            results = self.collection.get(include=['metadatas', 'embeddings'])
            
            if not results['embeddings'] or len(results['embeddings']) == 0:
                logger.error("No embeddings found in collection")
                return
            
            # Stack embeddings and convert to array
            embeddings = np.array(results['embeddings'])
            
            # Extract memory types for coloring
            memory_types = []
            for metadata in results['metadatas']:
                memory_types.append(metadata.get('memory_type', 'Unknown'))
            
            # Create unique color for each memory type
            unique_types = list(set(memory_types))
            type_to_color = {t: plt.cm.tab10(i % 10) for i, t in enumerate(unique_types)}
            colors = [type_to_color[t] for t in memory_types]
            
            # Apply dimensionality reduction
            if method == 'tsne':
                reducer = TSNE(n_components=2, random_state=42)
                reduced = reducer.fit_transform(embeddings)
                title = 't-SNE visualization of memory embeddings'
            else:  # default to PCA
                reducer = PCA(n_components=2)
                reduced = reducer.fit_transform(embeddings)
                title = 'PCA visualization of memory embeddings'
            
            # Create scatter plot
            plt.figure(figsize=(12, 8))
            for i, t in enumerate(unique_types):
                mask = np.array(memory_types) == t
                plt.scatter(reduced[mask, 0], reduced[mask, 1], c=[type_to_color[t]], label=t, alpha=0.7)
            
            plt.title(title)
            plt.xlabel('Component 1')
            plt.ylabel('Component 2')
            plt.legend()
            plt.grid(alpha=0.3)
            
            if output_path:
                plt.savefig(output_path)
                logger.info(f"Visualization saved to {output_path}")
            else:
                plt.show()
            
        except Exception as e:
            logger.error(f"Error visualizing embeddings: {e}", exc_info=True)
    
    def fix_missing_files(self, dry_run: bool = True) -> Tuple[int, int]:
        """Find and fix entries with missing files.
        
        This will try to locate files in the standard memory directories and
        update the metadata if files have moved.
        
        Args:
            dry_run: If True, don't make any changes, just report
            
        Returns:
            Tuple of (fixed_count, failed_count)
        """
        if not self.collection:
            if not self.connect():
                return 0, 0
        
        try:
            # Get all entries with metadatas
            results = self.collection.get(include=['metadatas'])
            
            fixed_count = 0
            failed_count = 0
            
            # Memory directories to search
            memory_dirs = ["facts", "preferences", "conversations", "reminders", "general"]
            base_memory_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "memories")
            
            for i, memory_id in enumerate(results['ids']):
                metadata = results['metadatas'][i]
                file_path = metadata.get('file_path', '')
                
                if not file_path or os.path.exists(file_path):
                    continue  # Skip if file path is empty or file exists
                
                # Extract filename
                filename = os.path.basename(file_path)
                
                # Check each memory directory for the file
                found = False
                new_path = None
                
                for memory_dir in memory_dirs:
                    potential_path = os.path.join(base_memory_dir, memory_dir, filename)
                    if os.path.exists(potential_path):
                        new_path = potential_path
                        found = True
                        break
                
                if found:
                    logger.info(f"Found missing file: {filename} at {new_path}")
                    
                    if not dry_run:
                        # Update metadata
                        updated_metadata = metadata.copy()
                        updated_metadata['file_path'] = new_path
                        
                        try:
                            self.collection.update(
                                ids=[memory_id],
                                metadatas=[updated_metadata]
                            )
                            logger.info(f"Updated metadata for {memory_id}")
                            fixed_count += 1
                        except Exception as e:
                            logger.error(f"Failed to update metadata for {memory_id}: {e}")
                            failed_count += 1
                    else:
                        logger.info(f"Would update {memory_id} with new path: {new_path}")
                        fixed_count += 1
                else:
                    logger.warning(f"Could not find file {filename} for {memory_id}")
                    failed_count += 1
            
            return fixed_count, failed_count
            
        except Exception as e:
            logger.error(f"Error fixing missing files: {e}")
            return 0, 0


def main():
    """Main entry point for the command line interface."""
    parser = argparse.ArgumentParser(description="ChromaDB Inspection Tool for Bella")
    
    # Subparsers for commands
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List all memory entries')
    list_parser.add_argument('-l', '--limit', type=int, default=100, help='Maximum entries to return')
    list_parser.add_argument('-o', '--offset', type=int, default=0, help='Offset for pagination')
    list_parser.add_argument('--output', type=str, help='Save output to file (CSV format)')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Display statistics about the database')
    stats_parser.add_argument('--output', type=str, help='Save statistics to file (JSON format)')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Perform a semantic search')
    search_parser.add_argument('query', type=str, help='Search query')
    search_parser.add_argument('-n', '--n-results', type=int, default=5, help='Number of results to return')
    search_parser.add_argument('--output', type=str, help='Save results to file (CSV format)')
    
    # Dump command
    dump_parser = subparsers.add_parser('dump', help='Export all database contents to a file')
    dump_parser.add_argument('--output', type=str, required=True, help='Output file path (JSON format)')
    
    # Visualize command
    viz_parser = subparsers.add_parser('visualize', help='Generate a visualization of memory embeddings')
    viz_parser.add_argument('--output', type=str, help='Output image file path')
    viz_parser.add_argument('--method', type=str, choices=['pca', 'tsne'], default='pca',
                           help='Dimensionality reduction method')
    
    # Fix command
    fix_parser = subparsers.add_parser('fix', help='Find and fix missing files')
    fix_parser.add_argument('--dry-run', action='store_true', help="Don't make any changes, just report")
    
    # Config arguments
    parser.add_argument('--db-path', type=str, default=CHROMA_DB_PATH, help='ChromaDB database path')
    parser.add_argument('--collection', type=str, default=CHROMA_COLLECTION_NAME, help='ChromaDB collection name')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create inspector
    inspector = ChromaInspector(args.db_path, args.collection)
    
    # Execute command
    if args.command == 'list':
        df = inspector.list_entries(limit=args.limit, offset=args.offset)
        if df.empty:
            print("No entries found.")
            return
        
        if args.output:
            df.to_csv(args.output, index=False)
            print(f"Saved {len(df)} entries to {args.output}")
        else:
            print(tabulate(df, headers='keys', tablefmt='pretty', showindex=False))
            print(f"\nTotal: {len(df)} entries")
    
    elif args.command == 'stats':
        stats = inspector.get_stats()
        if not stats:
            print("Failed to get statistics.")
            return
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(stats, f, indent=2, default=str)
            print(f"Saved statistics to {args.output}")
        else:
            print("\nChromaDB Collection Statistics:")
            print("------------------------------")
            print(f"Total entries: {stats['count']}")
            print("\nMemory Types:")
            for mem_type, count in stats.get('memory_types', {}).items():
                print(f"  - {mem_type}: {count}")
            
            print(f"\nOldest entry: {stats.get('oldest_entry', 'Unknown')}")
            print(f"Newest entry: {stats.get('newest_entry', 'Unknown')}")
            print(f"Date range: {stats.get('date_range_days', 0)} days")
            print(f"\nEmbedding dimensions: {stats.get('embedding_dimensions', 0)}")
            print(f"Existing files: {stats.get('existing_files', 0)}")
            print(f"Missing files: {stats.get('missing_files', 0)}")
    
    elif args.command == 'search':
        df = inspector.search(args.query, n_results=args.n_results)
        if df.empty:
            print("No results found.")
            return
        
        if args.output:
            df.to_csv(args.output, index=False)
            print(f"Saved {len(df)} results to {args.output}")
        else:
            # Display basic info in a table
            display_df = df[['id', 'similarity', 'title', 'memory_type']].copy()
            print("\nSearch Results:")
            print("--------------")
            print(tabulate(display_df, headers='keys', tablefmt='pretty', showindex=False))
            
            # Display content previews
            print("\nContent Previews:")
            print("----------------")
            for i, row in df.iterrows():
                print(f"\n[{i+1}] {row['title']} (ID: {row['id']}):")
                print(f"    Similarity: {row['similarity']:.4f}")
                print(f"    File: {row['file_path']} ({'EXISTS' if row['file_exists'] else 'MISSING'})")
                print("    ---")
                preview = row.get('content_preview', 'No preview available')
                print(f"    {preview[:500]}...")
                print("    ---")
    
    elif args.command == 'dump':
        success = inspector.dump_to_json(args.output)
        if success:
            print(f"Successfully dumped database to {args.output}")
        else:
            print("Failed to dump database.")
    
    elif args.command == 'visualize':
        inspector.visualize_embeddings(args.output, method=args.method)
        if args.output:
            print(f"Visualization saved to {args.output}")
        else:
            print("Visualization displayed. Close the window to continue.")
    
    elif args.command == 'fix':
        fixed, failed = inspector.fix_missing_files(dry_run=args.dry_run)
        if args.dry_run:
            print(f"Dry run: Would fix {fixed} entries, {failed} would still be missing.")
        else:
            print(f"Fixed {fixed} entries, {failed} still missing.")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
