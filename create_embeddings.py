import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import AutoModel, AutoTokenizer


class MarkdownEmbedder:
    def __init__(self, model_name="Qwen/Qwen3-Embedding-0.6B"):
        """Load the embedding model onto the best available device."""
        print("Loading model...")
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

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
        """Generate embeddings for a list of texts, processing in sub-batches."""
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

    def get_already_embedded_files(self, output_file):
        """Return the set of file paths already present in the embeddings parquet."""
        if not os.path.exists(output_file):
            return set()

        try:
            table = pq.read_table(output_file, columns=["file"])
            df = table.to_pandas()
            embedded = set(df["file"].unique())
            print(f"Found {len(embedded)} already-embedded files in {output_file}")
            return embedded
        except Exception as e:
            print(f"Warning: Could not read existing embeddings file: {e}")
            return set()

    def embed_chunks(
        self,
        chunks_file="chunks.parquet",
        output_file="embeddings.parquet",
        batch_size=100,
        embedding_batch_size=32,
        force_reprocess=False,
    ):
        """
        Read chunks from `chunks_file` and write embeddings to `output_file`.

        Supports resumability: files whose chunks are already present in
        `output_file` are skipped unless `force_reprocess=True`.

        Args:
            chunks_file:          Path to the chunks parquet produced by create_chunks.py.
            output_file:          Destination parquet file (adds an `embedding` column).
            batch_size:           Number of chunks to embed and flush to disk at once.
            embedding_batch_size: Number of texts passed to the model per forward pass.
            force_reprocess:      Re-embed every chunk, even if already present.
        """
        if not os.path.exists(chunks_file):
            raise FileNotFoundError(
                f"Chunks file not found: {chunks_file}\nRun create_chunks.py first."
            )

        print(f"Reading chunks from {chunks_file}...")
        chunks_table = pq.read_table(chunks_file)
        chunks_df = chunks_table.to_pandas()
        print(f"Total chunks available: {len(chunks_df)}")

        if force_reprocess:
            already_embedded = set()
            print("Force reprocess enabled – will embed all chunks")
            if os.path.exists(output_file):
                os.remove(output_file)
                print(f"Removed existing {output_file}")
        else:
            already_embedded = self.get_already_embedded_files(output_file)

        # Filter to only chunks from files not yet embedded
        if already_embedded:
            df_todo = chunks_df[~chunks_df["file"].isin(already_embedded)].reset_index(
                drop=True
            )
        else:
            df_todo = chunks_df

        if df_todo.empty:
            print("No new chunks to embed!")
            return 0

        skipped_files = len(chunks_df["file"].unique()) - len(df_todo["file"].unique())
        print(f"Chunks to embed   : {len(df_todo)}")
        print(f"Files skipped     : {skipped_files} (already embedded)")
        print(f"Embedding batch   : {embedding_batch_size}")
        print(f"Write batch size  : {batch_size}")

        writer = None
        total_embedded = 0

        for i in range(0, len(df_todo), batch_size):
            batch_df = df_todo.iloc[i : i + batch_size]
            texts = batch_df["text"].tolist()

            end_idx = min(i + batch_size, len(df_todo))
            print(f"Embedding chunks {i + 1}–{end_idx} of {len(df_todo)}...")

            embeddings = self.get_embeddings_batch(texts, embedding_batch_size)

            embedding_dim = len(embeddings[0])
            table = pa.table(
                {
                    "object_id": pa.array(
                        batch_df["object_id"].tolist(), type=pa.string()
                    ),
                    "file": pa.array(batch_df["file"].tolist(), type=pa.string()),
                    "chunk_index": pa.array(
                        batch_df["chunk_index"].tolist(), type=pa.int32()
                    ),
                    "text": pa.array(texts, type=pa.string()),
                    "embedding": pa.array(
                        [emb.astype(np.float32) for emb in embeddings],
                        type=pa.list_(pa.float32(), embedding_dim),
                    ),
                }
            )

            if writer is None:
                writer = pq.ParquetWriter(
                    output_file,
                    table.schema,
                    compression="zstd",
                    compression_level=9,
                )

            writer.write_table(table)
            total_embedded += len(batch_df)

            # Periodically free GPU memory
            batch_num = i // batch_size
            if batch_num % 10 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()

        if writer is not None:
            writer.close()

        print(f"\nDone! Embedded {total_embedded} chunks")
        print(f"Output saved to {output_file}")

        if os.path.exists(output_file):
            size_mb = os.path.getsize(output_file) / (1024 * 1024)
            print(f"File size: {size_mb:.2f} MB")

        return total_embedded


if __name__ == "__main__":
    embedder = MarkdownEmbedder()

    embedder.embed_chunks(
        chunks_file="chunks.parquet",
        output_file="embeddings.parquet",
        batch_size=100,
        embedding_batch_size=32,
    )

    # Resume (only embed new chunks):
    # embedder.embed_chunks(
    #     chunks_file="chunks.parquet",
    #     output_file="embeddings.parquet",
    # )

    # Force re-embed everything:
    # embedder.embed_chunks(
    #     chunks_file="chunks.parquet",
    #     output_file="embeddings.parquet",
    #     force_reprocess=True,
    # )
