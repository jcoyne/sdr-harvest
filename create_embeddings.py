import json
import os
from pathlib import Path

import numpy as np
import torch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoModel, AutoTokenizer


class StreamingMarkdownEmbedder:
    def __init__(self, model_name="Qwen/Qwen3-Embedding-0.6B"):
        """Initialize the embedding model"""
        print("Loading model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        # Use MPS (Metal) on Mac if available, otherwise CPU
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using MPS (Apple Silicon GPU)")
        else:
            self.device = torch.device("cpu")
            print("Using CPU")

        self.model.to(self.device)
        self.model.eval()

    def get_embedding(self, text):
        """Generate embedding for a single text chunk"""
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1)

        # Convert to float32 before converting to numpy to avoid BFloat16 issues
        return embeddings.float().cpu().numpy()[0]

    def chunk_markdown(self, text, chunk_size=500, chunk_overlap=50):
        """Split markdown into chunks"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", " ", ""],
        )
        return splitter.split_text(text)

    def process_directory_streaming(
        self, directory_path, output_file="embeddings.jsonl"
    ):
        """
        Process all markdown files and stream results to JSONL format.
        Each line is a separate JSON object, avoiding loading everything into memory.
        """
        md_files = list(Path(directory_path).rglob("*.md"))
        print(f"Found {len(md_files)} markdown files")

        total_chunks = 0

        # Open output file in write mode
        with open(output_file, "w", encoding="utf-8") as outfile:
            for idx, file_path in enumerate(md_files, 1):
                print(f"Processing {idx}/{len(md_files)}: {file_path}")

                try:
                    # Read file
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()

                    # Chunk the content
                    chunks = self.chunk_markdown(content)
                    print(f"  - Split into {len(chunks)} chunks")

                    # Process each chunk and write immediately
                    for chunk_idx, chunk in enumerate(chunks):
                        embedding = self.get_embedding(chunk)

                        result = {
                            "file": str(file_path),
                            "chunk_index": chunk_idx,
                            "text": chunk,
                            "embedding": embedding.tolist(),
                            "embedding_dim": len(embedding),
                        }

                        # Write as single line JSON (JSONL format)
                        outfile.write(json.dumps(result) + "\n")
                        total_chunks += 1

                    # Clear cache periodically
                    if idx % 10 == 0:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        if torch.backends.mps.is_available():
                            torch.mps.empty_cache()

                except Exception as e:
                    print(f"  - Error processing {file_path}: {e}")

        print(f"\nDone! Processed {total_chunks} total chunks")
        print(f"Output saved to {output_file}")
        return total_chunks


class StreamingEmbeddingReader:
    """Helper class to read embeddings back from JSONL file"""

    @staticmethod
    def read_chunks(jsonl_file):
        """Generator that yields one chunk at a time"""
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)

    @staticmethod
    def search_similar(query, embedder, jsonl_file, top_k=5):
        """Search similar chunks without loading all into memory"""
        query_embedding = embedder.get_embedding(query)

        # Use a min-heap to keep only top k results
        from heapq import nlargest

        def similarity_generator():
            """Generator that yields (similarity, item) tuples"""
            for item in StreamingEmbeddingReader.read_chunks(jsonl_file):
                doc_embedding = np.array(item["embedding"], dtype=np.float32)

                # Cosine similarity
                similarity = np.dot(query_embedding, doc_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                )
                yield (similarity, item)

        # Get top k without loading all into memory
        top_results = nlargest(top_k, similarity_generator(), key=lambda x: x[0])
        return top_results

    @staticmethod
    def count_chunks(jsonl_file):
        """Count total chunks without loading into memory"""
        count = 0
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for _ in f:
                count += 1
        return count

    @staticmethod
    def get_files_list(jsonl_file):
        """Get unique list of source files"""
        files = set()
        for chunk in StreamingEmbeddingReader.read_chunks(jsonl_file):
            files.add(chunk["file"])
        return sorted(list(files))


# Usage Example
if __name__ == "__main__":
    # Create embeddings
    embedder = StreamingMarkdownEmbedder()

    total_chunks = embedder.process_directory_streaming(
        directory_path="./extracted_texts",
        output_file="embeddings.jsonl",  # Note: .jsonl extension
    )

    print(f"\nProcessed {total_chunks} total chunks")

    # Example: Read chunks back one at a time
    print("\nReading first 3 chunks:")
    reader = StreamingEmbeddingReader()
    for i, chunk in enumerate(reader.read_chunks("embeddings.jsonl")):
        if i >= 3:
            break
        print(f"Chunk {i}: {chunk['file']} - {chunk['text'][:50]}...")

    # Example: Search without loading all into memory
    print("\nSearching for similar chunks:")
    results = reader.search_similar(
        query="your search query",
        embedder=embedder,
        jsonl_file="embeddings.jsonl",
        top_k=5,
    )

    for similarity, item in results:
        print(f"Similarity: {similarity:.4f} - {item['file']}")
        print(f"  {item['text'][:100]}...\n")
