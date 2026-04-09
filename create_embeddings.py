import os
import queue
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoModel, AutoTokenizer


def read_and_chunk_file(args):
    """
    Worker function to read and chunk a single file.
    This is I/O bound, so we can use threads.
    """
    file_path, chunk_size, chunk_overlap = args

    try:
        # Extract object_id
        parts = Path(file_path).parts
        object_id = None
        try:
            extracted_idx = parts.index("extracted_texts")
            if extracted_idx + 1 < len(parts):
                object_id = parts[extracted_idx + 1]
        except (ValueError, IndexError):
            pass

        # Read file
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Chunk the content
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", " ", ""],
        )
        chunks = splitter.split_text(content)

        # Return chunks with metadata
        results = []
        for chunk_idx, chunk in enumerate(chunks):
            results.append(
                {
                    "object_id": object_id,
                    "file": str(file_path),
                    "chunk_index": chunk_idx,
                    "text": chunk,
                }
            )

        return {"success": True, "file": str(file_path), "chunks": results}

    except Exception as e:
        return {"success": False, "file": str(file_path), "error": str(e)}


class ResumableMarkdownEmbedder:
    def __init__(self, model_name="Qwen/Qwen3-Embedding-0.6B"):
        """Initialize the embedding model"""
        print("Loading model...")
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        # Use MPS (Metal) on Mac if available, otherwise CPU
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using MPS (Apple Silicon GPU)")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Using CUDA GPU")
        else:
            self.device = torch.device("cpu")
            print("Using CPU")

        self.model.to(self.device)
        self.model.eval()

    def get_embeddings_batch(self, texts, batch_size=32):
        """Generate embeddings for multiple texts in batches"""
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_embeddings = outputs.last_hidden_state.mean(dim=1)
                batch_embeddings = batch_embeddings.float().cpu().numpy()

            embeddings.extend(batch_embeddings)

        return embeddings

    def get_processed_files(self, parquet_file):
        """
        Read the existing Parquet file and return set of already processed files.
        """
        if not os.path.exists(parquet_file):
            return set()

        try:
            table = pq.read_table(parquet_file, columns=["file"])
            df = table.to_pandas()
            processed_files = set(df["file"].unique())
            print(
                f"Found {len(processed_files)} already processed files in {parquet_file}"
            )
            return processed_files
        except Exception as e:
            print(f"Warning: Could not read existing parquet file: {e}")
            return set()

    def process_directory_resumable(
        self,
        directory_path,
        output_file="embeddings.parquet",
        batch_size=100,
        force_reprocess=False,
        chunk_size=500,
        chunk_overlap=50,
        embedding_batch_size=32,
        num_io_workers=4,
    ):
        """
        Process markdown files using threading for I/O and batched embeddings.

        Args:
            embedding_batch_size: Number of chunks to embed at once
            num_io_workers: Number of threads for reading/chunking files
        """
        md_files = list(Path(directory_path).rglob("*.md"))
        print(f"Found {len(md_files)} total markdown files")

        # Get already processed files
        if force_reprocess:
            processed_files = set()
            print("Force reprocess enabled - will process all files")
            if os.path.exists(output_file):
                os.remove(output_file)
                print(f"Removed existing {output_file}")
        else:
            processed_files = self.get_processed_files(output_file)

        # Filter to only unprocessed files
        files_to_process = [f for f in md_files if str(f) not in processed_files]

        if not files_to_process:
            print("No new files to process!")
            return 0

        print(f"Will process {len(files_to_process)} new files")
        print(
            f"Skipping {len(md_files) - len(files_to_process)} already processed files"
        )
        print(f"Using {num_io_workers} I/O worker threads")
        print(f"Embedding batch size: {embedding_batch_size}")

        # Prepare arguments for chunking
        chunk_args = [(str(f), chunk_size, chunk_overlap) for f in files_to_process]

        total_chunks = 0
        all_chunk_data = []

        # Phase 1: Read and chunk files using threads (I/O bound)
        print("\n=== Phase 1: Reading and chunking files ===")
        with ThreadPoolExecutor(max_workers=num_io_workers) as executor:
            future_to_file = {
                executor.submit(read_and_chunk_file, args): args[0]
                for args in chunk_args
            }

            completed = 0
            for future in as_completed(future_to_file):
                completed += 1

                try:
                    result = future.result()

                    if result["success"]:
                        num_chunks = len(result["chunks"])
                        print(
                            f"[{completed}/{len(files_to_process)}] ✓ {result['file']} - {num_chunks} chunks"
                        )
                        all_chunk_data.extend(result["chunks"])
                    else:
                        print(
                            f"[{completed}/{len(files_to_process)}] ✗ {result['file']} - Error: {result['error']}"
                        )

                except Exception as e:
                    print(f"[{completed}/{len(files_to_process)}] ✗ Exception: {e}")

        if not all_chunk_data:
            print("No chunks to process!")
            return 0

        print(f"\nTotal chunks to embed: {len(all_chunk_data)}")

        # Phase 2: Generate embeddings in batches (GPU/CPU bound)
        print("\n=== Phase 2: Generating embeddings ===")
        writer = None

        for i in range(0, len(all_chunk_data), batch_size):
            batch_chunks = all_chunk_data[i : i + batch_size]

            # Extract texts for embedding
            texts = [chunk["text"] for chunk in batch_chunks]

            # Generate embeddings in sub-batches
            print(
                f"Embedding chunks {i + 1}-{min(i + batch_size, len(all_chunk_data))} of {len(all_chunk_data)}..."
            )
            embeddings = self.get_embeddings_batch(texts, embedding_batch_size)

            # Prepare batch data for writing
            batch_data = {
                "object_id": [chunk["object_id"] for chunk in batch_chunks],
                "file": [chunk["file"] for chunk in batch_chunks],
                "chunk_index": [chunk["chunk_index"] for chunk in batch_chunks],
                "text": texts,
                "embedding": [emb.astype(np.float32) for emb in embeddings],
            }

            # Write batch
            writer = self._write_batch(batch_data, output_file, writer)
            total_chunks += len(batch_chunks)

            # Clear cache periodically
            if (i // batch_size) % 10 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()

        # Close the writer
        if writer is not None:
            writer.close()

        print(f"\nDone! Processed {total_chunks} new chunks")
        print(f"Output saved to {output_file}")

        # Print file size
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file) / (1024 * 1024)
            print(f"Total file size: {file_size:.2f} MB")

        return total_chunks

    def _write_batch(self, batch_data, output_file, writer):
        """Helper method to write a batch to Parquet"""
        if not batch_data["file"]:
            return writer

        # Convert to PyArrow table
        embedding_dim = len(batch_data["embedding"][0])

        table = pa.table(
            {
                "object_id": pa.array(batch_data["object_id"], type=pa.string()),
                "file": pa.array(batch_data["file"], type=pa.string()),
                "chunk_index": pa.array(batch_data["chunk_index"], type=pa.int32()),
                "text": pa.array(batch_data["text"], type=pa.string()),
                "embedding": pa.array(
                    batch_data["embedding"],
                    type=pa.list_(pa.float32(), embedding_dim),
                ),
            }
        )

        # Create writer if needed
        if writer is None:
            writer = pq.ParquetWriter(
                output_file,
                table.schema,
                compression="zstd",
                compression_level=9,
            )

        # Write the table
        writer.write_table(table)

        return writer

    def reprocess_specific_files(
        self,
        file_paths,
        output_file="embeddings.parquet",
        embedding_batch_size=32,
        num_io_workers=4,
    ):
        """
        Reprocess specific files and update the Parquet file.
        """
        if not os.path.exists(output_file):
            print(f"No existing file {output_file}, processing normally")
            return self.process_specific_files(
                file_paths, output_file, embedding_batch_size, num_io_workers
            )

        # Read existing data
        print("Reading existing embeddings...")
        table = pq.read_table(output_file)
        df = table.to_pandas()

        # Convert file_paths to set of strings for comparison
        files_to_reprocess = set(str(Path(f)) for f in file_paths)

        # Filter out the files we're reprocessing
        df_keep = df[~df["file"].isin(files_to_reprocess)]
        print(f"Keeping {len(df_keep)} existing chunks")
        print(f"Removing {len(df) - len(df_keep)} chunks from files being reprocessed")

        # Write the filtered data back
        if len(df_keep) > 0:
            table_keep = pa.Table.from_pandas(df_keep, schema=table.schema)
            pq.write_table(
                table_keep,
                output_file,
                compression="zstd",
                compression_level=9,
            )
        else:
            os.remove(output_file)

        # Now process the specified files
        print(f"\nProcessing {len(files_to_reprocess)} files...")
        return self.process_specific_files(
            file_paths, output_file, embedding_batch_size, num_io_workers
        )

    def process_specific_files(
        self,
        file_paths,
        output_file="embeddings.parquet",
        embedding_batch_size=32,
        num_io_workers=4,
        batch_size=100,
    ):
        """Process a specific list of files"""
        chunk_args = [(str(f), 500, 50) for f in file_paths if Path(f).exists()]

        all_chunk_data = []

        # Read and chunk files
        print("Reading and chunking files...")
        with ThreadPoolExecutor(max_workers=num_io_workers) as executor:
            future_to_file = {
                executor.submit(read_and_chunk_file, args): args[0]
                for args in chunk_args
            }

            completed = 0
            for future in as_completed(future_to_file):
                completed += 1
                result = future.result()

                if result["success"]:
                    print(f"[{completed}/{len(chunk_args)}] ✓ {result['file']}")
                    all_chunk_data.extend(result["chunks"])

        if not all_chunk_data:
            return 0

        # Generate embeddings
        print(f"\nGenerating embeddings for {len(all_chunk_data)} chunks...")
        total_chunks = 0
        writer = None

        for i in range(0, len(all_chunk_data), batch_size):
            batch_chunks = all_chunk_data[i : i + batch_size]
            texts = [chunk["text"] for chunk in batch_chunks]

            embeddings = self.get_embeddings_batch(texts, embedding_batch_size)

            batch_data = {
                "object_id": [chunk["object_id"] for chunk in batch_chunks],
                "file": [chunk["file"] for chunk in batch_chunks],
                "chunk_index": [chunk["chunk_index"] for chunk in batch_chunks],
                "text": texts,
                "embedding": [emb.astype(np.float32) for emb in embeddings],
            }

            writer = self._write_batch(batch_data, output_file, writer)
            total_chunks += len(batch_chunks)

        if writer is not None:
            writer.close()

        return total_chunks


if __name__ == "__main__":
    embedder = ResumableMarkdownEmbedder()

    # Process with threading for I/O and batched embeddings
    embedder.process_directory_resumable(
        directory_path="./extracted_texts",
        output_file="embeddings.parquet",
        batch_size=100,  # Chunks to write at once
        embedding_batch_size=32,  # Chunks to embed at once
        num_io_workers=4,  # Number of I/O threads
    )

    # Second run - only processes new files
    # print("\n=== Resumable Processing (run again after adding files) ===")
    # embedder.process_directory_resumable(
    #     directory_path="./extracted_texts",
    #     output_file="embeddings.parquet",
    #     batch_size=100,
    # )

    # Force reprocess everything
    # print("\n=== Force Reprocess All ===")
    # embedder.process_directory_resumable(
    #     directory_path="./extracted_texts",
    #     output_file="embeddings.parquet",
    #     batch_size=100,
    #     force_reprocess=True,
    # )

    # Reprocess specific files (e.g., if they changed)
    # print("\n=== Reprocess Specific Files ===")
    # files_to_update = [
    #     "extracted_texts/druid123/file1.md",
    #     "extracted_texts/druid456/file2.md",
    # ]
    # embedder.reprocess_specific_files(
    #     file_paths=files_to_update,
    #     output_file="embeddings.parquet",
    # )

    # Second run - only processes new files
    # print("\n=== Resumable Processing (run again after adding files) ===")
    # embedder.process_directory_resumable(
    #     directory_path="./extracted_texts",
    #     output_file="embeddings.parquet",
    #     batch_size=100,
    # )

    # Force reprocess everything
    # print("\n=== Force Reprocess All ===")
    # embedder.process_directory_resumable(
    #     directory_path="./extracted_texts",
    #     output_file="embeddings.parquet",
    #     batch_size=100,
    #     force_reprocess=True,
    # )

    # Reprocess specific files (e.g., if they changed)
    # print("\n=== Reprocess Specific Files ===")
    # files_to_update = [
    #     "extracted_texts/druid123/file1.md",
    #     "extracted_texts/druid456/file2.md",
    # ]
    # embedder.reprocess_specific_files(
    #     file_paths=files_to_update,
    #     output_file="embeddings.parquet",
    # )
