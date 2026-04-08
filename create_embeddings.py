import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoModel, AutoTokenizer


class ResumableMarkdownEmbedder:
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

    def extract_object_id(self, file_path):
        """
        Extract object_id from file path.
        For path like 'extracted_texts/kq478vz7750/FEMGEN100CManuscript.md',
        returns 'kq478vz7750'
        """
        parts = Path(file_path).parts
        try:
            # Find 'extracted_texts' in the path and get the next part
            extracted_idx = parts.index("extracted_texts")
            if extracted_idx + 1 < len(parts):
                return parts[extracted_idx + 1]
        except (ValueError, IndexError):
            pass

        # Fallback: return None or empty string if pattern not found
        return None

    def get_embedding(self, text):
        """Generate embedding for a single text chunk"""
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)

        return embeddings.float().cpu().numpy()[0]

    def chunk_markdown(self, text, chunk_size=500, chunk_overlap=50):
        """Split markdown into chunks"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", " ", ""],
        )
        return splitter.split_text(text)

    def get_processed_files(self, parquet_file):
        """
        Read the existing Parquet file and return set of already processed files.
        Only reads the 'file' column for efficiency.
        """
        if not os.path.exists(parquet_file):
            return set()

        try:
            # Only read the 'file' column - very fast even for large files
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
    ):
        """
        Process markdown files, skipping those already in the Parquet file.
        """
        md_files = list(Path(directory_path).rglob("*.md"))
        print(f"Found {len(md_files)} total markdown files")

        # Get already processed files
        if force_reprocess:
            processed_files = set()
            print("Force reprocess enabled - will process all files")
            # Remove existing file if force reprocessing
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

        total_chunks = 0
        batch_data = {
            "object_id": [],
            "file": [],
            "chunk_index": [],
            "text": [],
            "embedding": [],
        }
        writer = None

        for idx, file_path in enumerate(files_to_process, 1):
            print(f"Processing {idx}/{len(files_to_process)}: {file_path}")

            try:
                # Extract object_id
                object_id = self.extract_object_id(file_path)

                # Read file
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # Chunk the content
                chunks = self.chunk_markdown(content)
                print(f"  - Split into {len(chunks)} chunks")

                # Process each chunk
                for chunk_idx, chunk in enumerate(chunks):
                    embedding = self.get_embedding(chunk)

                    batch_data["object_id"].append(object_id)
                    batch_data["file"].append(str(file_path))
                    batch_data["chunk_index"].append(chunk_idx)
                    batch_data["text"].append(chunk)
                    batch_data["embedding"].append(embedding.astype(np.float32))

                    total_chunks += 1

                    # Write batch when full
                    if len(batch_data["file"]) >= batch_size:
                        writer = self._write_batch(batch_data, output_file, writer)
                        batch_data = {k: [] for k in batch_data}

                # Clear cache periodically
                if idx % 10 == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()

            except Exception as e:
                print(f"  - Error processing {file_path}: {e}")

        # Write remaining data
        if batch_data["file"]:
            writer = self._write_batch(batch_data, output_file, writer)

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

    def reprocess_specific_files(self, file_paths, output_file="embeddings.parquet"):
        """
        Reprocess specific files and update the Parquet file.
        This removes old entries for these files and adds new ones.

        Args:
            file_paths: List of file paths to reprocess
            output_file: Parquet file to update
        """
        if not os.path.exists(output_file):
            print(f"No existing file {output_file}, processing normally")
            return self.process_specific_files(file_paths, output_file)

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
            # Remove file if no data left
            os.remove(output_file)

        # Now process the specified files
        print(f"\nProcessing {len(files_to_reprocess)} files...")
        return self.process_specific_files(file_paths, output_file)

    def process_specific_files(
        self, file_paths, output_file="embeddings.parquet", batch_size=100
    ):
        """Process a specific list of files (helper method)"""
        total_chunks = 0
        batch_data = {
            "object_id": [],
            "file": [],
            "chunk_index": [],
            "text": [],
            "embedding": [],
        }
        writer = None

        for idx, file_path in enumerate(file_paths, 1):
            file_path = Path(file_path)

            if not file_path.exists():
                print(f"Warning: {file_path} does not exist, skipping")
                continue

            print(f"Processing {idx}/{len(file_paths)}: {file_path}")

            try:
                # Extract object_id
                object_id = self.extract_object_id(file_path)

                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                chunks = self.chunk_markdown(content)
                print(f"  - Split into {len(chunks)} chunks")

                for chunk_idx, chunk in enumerate(chunks):
                    embedding = self.get_embedding(chunk)

                    batch_data["object_id"].append(object_id)
                    batch_data["file"].append(str(file_path))
                    batch_data["chunk_index"].append(chunk_idx)
                    batch_data["text"].append(chunk)
                    batch_data["embedding"].append(embedding.astype(np.float32))

                    total_chunks += 1

                    if len(batch_data["file"]) >= batch_size:
                        writer = self._write_batch(batch_data, output_file, writer)
                        batch_data = {k: [] for k in batch_data}

            except Exception as e:
                print(f"  - Error processing {file_path}: {e}")

        # Write remaining data
        if batch_data["file"]:
            writer = self._write_batch(batch_data, output_file, writer)

        if writer is not None:
            writer.close()

        return total_chunks


# Usage Examples
if __name__ == "__main__":
    embedder = ResumableMarkdownEmbedder()

    # First run - processes all files
    print("=== Initial Processing ===")
    embedder.process_directory_resumable(
        directory_path="./extracted_texts",
        output_file="embeddings.parquet",
        batch_size=100,
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
