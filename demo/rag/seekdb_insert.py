import sys
import os
from glob import glob
from tqdm import tqdm
from typing import Dict, List, Iterator

from embedding_function_factory import create_embedding_function
from seekdb_utils import get_seekdb_client, get_seekdb_collection, insert_embeddings
from dotenv import load_dotenv

load_dotenv()

MAX_TEXT_LENGTH = 8000
BATCH_SIZE = 10


def load_markdown_files(data_path: str) -> Dict[str, List[str]]:
    """Load markdown files from a directory or a single file and split into chunks by headers."""
    text_dict = {}
    
    # Check if input is a single file
    if os.path.isfile(data_path):
        if data_path.endswith('.md'):
            print(f"Loading single markdown file: {data_path}")
            files = [data_path]
        else:
            print(f"Warning: File '{data_path}' is not a markdown file (.md). Skipping.")
            return text_dict
    else:
        # Treat as directory
        files = list(glob(os.path.join(data_path, "**/*.md"), recursive=True))
        print(f"Loading {len(files)} markdown files from directory...")
    
    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                chunks = [c.strip() for c in f.read().split("# ") if c.strip()]
                if chunks:
                    text_dict[file_path] = chunks
        except Exception as e:
            print(f"Warning: Failed to load {file_path}: {e}")
            continue
    
    return text_dict


def prepare_data(text_dict: Dict[str, List[str]]) -> Iterator[dict]:
    """Prepare data for insertion (text only, embeddings will be auto-generated)."""
    total_chunks = sum(len(chunks) for chunks in text_dict.values())
    
    with tqdm(total=total_chunks, desc="Preparing data") as pbar:
        for filepath, chunks in text_dict.items():
            for idx, chunk_text in enumerate(chunks):
                pbar.update(1)
                
                if not chunk_text.strip():
                    continue
                
                # Truncate if too long
                if len(chunk_text) > MAX_TEXT_LENGTH:
                    chunk_text = chunk_text[:MAX_TEXT_LENGTH]
                
                yield {
                    "text": chunk_text,
                    "source_file": filepath,
                    "chunk_index": idx
                }


def process_and_insert_data(
    data_path: str, 
    db_dir: str = None, 
    db_name: str = None, 
    collection_name: str = None
):
    """Process text data and insert into seekdb using local embedding model."""
    # Read from environment variables if not provided
    if db_dir is None:
        db_dir = os.getenv("SEEKDB_DIR")
    if db_name is None:
        db_name = os.getenv("SEEKDB_NAME")
    if collection_name is None:
        collection_name = os.getenv("COLLECTION_NAME")
    
    # Initialize clients
    print("Initializing clients...")
    seekdb_client = get_seekdb_client(db_dir=db_dir, db_name=db_name)
    
    # Create collection with embedding function
    collection = get_seekdb_collection(
        seekdb_client, 
        collection_name=collection_name, 
        embedding_function=create_embedding_function(),
        drop_if_exists=True
    )
    
    # Load documents (supports both single file and directory)
    text_dict = load_markdown_files(data_path)
    if not text_dict:
        print("❌ No documents found.")
        return
    
    total_chunks = sum(len(c) for c in text_dict.values())
    print(f"Loaded {len(text_dict)} files with {total_chunks} chunks")
    
    # Process and insert data
    print("Inserting data (embeddings will be auto-generated)...")
    batch = []
    processed = 0
    
    for data in prepare_data(text_dict):
        batch.append(data)
        
        # Insert in batches
        if len(batch) >= BATCH_SIZE:
            try:
                insert_embeddings(collection, batch)
                processed += len(batch)
            except Exception as e:
                print(f"\n⚠️  Error inserting batch: {e}")
            finally:
                batch = []
    
    # Insert remaining data
    if batch:
        try:
            insert_embeddings(collection, batch)
            processed += len(batch)
        except Exception as e:
            print(f"\n⚠️  Error inserting final batch: {e}")
    
    print(f"\n✅ Successfully processed {processed} chunks")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python seekdb_insert.py <markdown_file_or_directory>")
        print("  Examples:")
        print("    python seekdb_insert.py document.md")
        print("    python seekdb_insert.py ./docs")
        sys.exit(1)
    
    data_path = sys.argv[1]
    
    # Validate inputs
    if not os.path.exists(data_path):
        print(f"❌ Path '{data_path}' not found.")
        sys.exit(1)
    
    # Check if it's a file, validate extension
    if os.path.isfile(data_path) and not data_path.endswith('.md'):
        print(f"❌ File '{data_path}' is not a markdown file (.md).")
        sys.exit(1)
    
    process_and_insert_data(data_path)


if __name__ == "__main__":
    main()
