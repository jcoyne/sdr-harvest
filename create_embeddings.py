import concurrent.futures
import logging
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from google import genai
from google.genai import types


class MarkdownEmbedder:
    def __init__(
        self,
        model_name="models/gemini-embedding-2",
        log_file="embedding_process.log",
        max_workers=8,
    ):
        """Set up the Gemini embedding API client."""
        # Set up logging
        self.setup_logging(log_file)

        self.logger.info("=" * 80)
        self.logger.info(
            f"Starting MarkdownEmbedder initialization at {datetime.now()}"
        )
        self.logger.info(f"Model: {model_name}")

        self.model_name = model_name

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable is not set. "
                "Please set it before running this script."
            )

        self.client = genai.Client(api_key=api_key)
        self.max_workers = max_workers
        self.logger.info("Gemini API client initialized successfully")
        self.logger.info(f"Parallel workers: {max_workers}")
        print(f"Using Gemini API with model: {model_name}")
        print(f"Parallel API workers: {max_workers}")

    def setup_logging(self, log_file):
        """Configure logging to both file and console."""
        self.logger = logging.getLogger("MarkdownEmbedder")
        self.logger.setLevel(logging.INFO)

        # Remove existing handlers
        self.logger.handlers = []

        # File handler
        fh = logging.FileHandler(log_file, mode="a")
        fh.setLevel(logging.INFO)

        # Console handler for errors only
        ch = logging.StreamHandler()
        ch.setLevel(logging.ERROR)

        # Formatter
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def get_embeddings_batch(self, texts, batch_size=32):
        """Generate embeddings for a list of texts via the Gemini API.

        Sub-batches of `batch_size` texts are dispatched in parallel using a
        ThreadPoolExecutor (``self.max_workers`` concurrent requests), so API
        round-trip latency is overlapped rather than paid sequentially.

        Each text is wrapped in its own Content object so the API returns a
        separate embedding per text (rather than one aggregated embedding).
        Includes exponential-backoff retry logic for transient API errors.
        """
        # Build all sub-batches up front so we can submit them all at once.
        sub_batches = [
            texts[i : i + batch_size] for i in range(0, len(texts), batch_size)
        ]

        def _call_api(batch_idx, batch_texts):
            """Embed one sub-batch with retry; returns list of np.ndarray."""
            contents = [
                types.Content(parts=[types.Part.from_text(text=t)]) for t in batch_texts
            ]
            max_retries = 5
            delay = 2.0  # seconds
            for attempt in range(max_retries):
                try:
                    result = self.client.models.embed_content(
                        model=self.model_name,
                        contents=contents,
                        config=types.EmbedContentConfig(
                            output_dimensionality=768
                        ),  # reduced from default of 3072
                    )
                    return [
                        np.array(emb.values, dtype=np.float32)
                        for emb in result.embeddings
                    ]
                except Exception as e:
                    if attempt < max_retries - 1:
                        self.logger.warning(
                            f"API error on sub-batch {batch_idx}, attempt "
                            f"{attempt + 1}/{max_retries}: {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                        delay *= 2  # exponential back-off
                    else:
                        self.logger.error(
                            f"Failed to embed sub-batch {batch_idx} after "
                            f"{max_retries} attempts: {e}"
                        )
                        self.logger.error(
                            f"Batch text samples: {[t[:100] for t in batch_texts[:3]]}"
                        )
                        raise

        # Submit all sub-batches concurrently and collect results in order.
        results = [None] * len(sub_batches)
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            future_to_idx = {
                executor.submit(_call_api, idx, batch): idx
                for idx, batch in enumerate(sub_batches)
            }
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()  # re-raises any exception

        # Flatten sub-batch results back into a single ordered list.
        return [emb for batch_result in results for emb in batch_result]

    def get_already_embedded_chunks(self, output_file):
        """Return the set of (file, chunk_index) tuples already present in the embeddings parquet."""
        if not os.path.exists(output_file):
            self.logger.info(f"Output file {output_file} does not exist yet")
            return set()

        try:
            table = pq.read_table(output_file, columns=["file", "chunk_index"])
            df = table.to_pandas()

            # Create a set of (file, chunk_index) tuples
            embedded_chunks = set(zip(df["file"], df["chunk_index"]))

            num_files = df["file"].nunique()
            num_chunks = len(embedded_chunks)

            self.logger.info(
                f"Found {num_chunks} already-embedded chunks from {num_files} files in {output_file}"
            )
            print(
                f"Found {num_chunks} already-embedded chunks from {num_files} files in {output_file}"
            )

            # Log a sample
            sample_files = df["file"].unique()[:5]
            self.logger.info(f"Sample of files with embeddings: {list(sample_files)}")

            return embedded_chunks
        except Exception as e:
            self.logger.error(f"Could not read existing embeddings file: {e}")
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

        Supports resumability: chunks already present in `output_file`
        are skipped unless `force_reprocess=True`.

        Args:
            chunks_file:          Path to the chunks parquet produced by create_chunks.py.
            output_file:          Destination parquet file (adds an `embedding` column).
            batch_size:           Number of chunks to embed and flush to disk at once.
            embedding_batch_size: Number of texts sent to the Gemini API per request.
            force_reprocess:      Re-embed every chunk, even if already present.
        """
        self.logger.info("=" * 80)
        self.logger.info(f"Starting embed_chunks at {datetime.now()}")
        self.logger.info(f"Chunks file: {chunks_file}")
        self.logger.info(f"Output file: {output_file}")
        self.logger.info(f"Batch size: {batch_size}")
        self.logger.info(f"Embedding batch size: {embedding_batch_size}")
        self.logger.info(f"Force reprocess: {force_reprocess}")

        if not os.path.exists(chunks_file):
            error_msg = (
                f"Chunks file not found: {chunks_file}\nRun create_chunks.py first."
            )
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        print(f"Reading chunks from {chunks_file}...")
        self.logger.info(f"Reading chunks from {chunks_file}...")

        try:
            chunks_table = pq.read_table(chunks_file)
            chunks_df = chunks_table.to_pandas()
            self.logger.info(f"Successfully read {len(chunks_df)} chunks")
        except Exception as e:
            self.logger.error(f"Error reading chunks file: {e}")
            raise

        total_files = chunks_df["file"].nunique()
        total_chunks = len(chunks_df)
        print(f"Total chunks available: {total_chunks}")
        self.logger.info(f"Total chunks: {total_chunks} from {total_files} files")

        temp_output = output_file + ".new"

        # If a previous run was interrupted, a leftover temp file may exist.
        # Merge it into the output file NOW, before counting already-embedded
        # chunks, so this run picks up exactly where the last one left off.
        if os.path.exists(temp_output) and not force_reprocess:
            print(
                f"Found partial results from a previous interrupted run ({temp_output}). Merging..."
            )
            self.logger.info(
                f"Found leftover temp file: {temp_output}, merging into {output_file}"
            )
            try:
                leftover_table = pq.read_table(temp_output)
                if os.path.exists(output_file):
                    existing_table = pq.read_table(output_file)
                    combined_table = pa.concat_tables([existing_table, leftover_table])
                else:
                    combined_table = leftover_table
                pq.write_table(
                    combined_table, output_file, compression="zstd", compression_level=9
                )
                os.remove(temp_output)
                print(
                    f"Merged {len(leftover_table)} chunks from previous run into {output_file}"
                )
                self.logger.info(
                    f"Merged {len(leftover_table)} partial chunks from temp file"
                )
            except Exception as e:
                self.logger.error(f"Failed to merge leftover temp file: {e}")
                print(f"Warning: Could not merge leftover temp file: {e}")

        if force_reprocess:
            already_embedded = set()
            print("Force reprocess enabled – will embed all chunks")
            self.logger.info("Force reprocess enabled – will embed all chunks")
            if os.path.exists(output_file):
                os.remove(output_file)
                self.logger.info(f"Removed existing {output_file}")
                print(f"Removed existing {output_file}")
            if os.path.exists(temp_output):
                os.remove(temp_output)
                self.logger.info(f"Removed leftover temp file: {temp_output}")
        else:
            already_embedded = self.get_already_embedded_chunks(output_file)

        # Filter to only chunks not yet embedded (based on file + chunk_index)
        if already_embedded:
            # Create a column for matching
            chunks_df["_key"] = list(zip(chunks_df["file"], chunks_df["chunk_index"]))
            df_todo = chunks_df[~chunks_df["_key"].isin(already_embedded)].copy()
            df_todo = df_todo.drop(columns=["_key"]).reset_index(drop=True)
        else:
            df_todo = chunks_df

        if df_todo.empty:
            self.logger.info("No new chunks to embed!")
            print("No new chunks to embed!")
            return 0

        skipped_chunks = total_chunks - len(df_todo)
        files_to_process = df_todo["file"].nunique()

        print(f"Total chunks      : {total_chunks}")
        print(f"Already embedded  : {skipped_chunks}")
        print(f"Chunks to embed   : {len(df_todo)}")
        print(f"Files to process  : {files_to_process}")
        print(f"Embedding batch   : {embedding_batch_size}")
        print(f"Write batch size  : {batch_size}")

        self.logger.info(f"Total chunks: {total_chunks}")
        self.logger.info(f"Already embedded: {skipped_chunks}")
        self.logger.info(f"Chunks to embed: {len(df_todo)}")
        self.logger.info(f"Files to process: {files_to_process}")

        # Log files to be processed
        sample_files = df_todo["file"].unique()[:10]
        self.logger.info(f"Sample of files to process: {list(sample_files)}")

        # Use a temporary file for new embeddings, then merge at the end

        total_embedded = 0
        files_processed = set()
        last_successful_batch = -1
        writer = None

        try:
            for i in range(0, len(df_todo), batch_size):
                batch_df = df_todo.iloc[i : i + batch_size]
                texts = batch_df["text"].tolist()
                batch_files = batch_df["file"].unique()

                end_idx = min(i + batch_size, len(df_todo))
                print(f"Embedding chunks {i + 1}–{end_idx} of {len(df_todo)}...")
                self.logger.info(
                    f"Processing batch {i // batch_size}: chunks {i + 1}–{end_idx}"
                )
                self.logger.info(f"Files in this batch: {list(batch_files)}")

                try:
                    embeddings = self.get_embeddings_batch(texts, embedding_batch_size)
                    self.logger.info(
                        f"Successfully generated {len(embeddings)} embeddings"
                    )
                except Exception as e:
                    self.logger.error(
                        f"Failed to generate embeddings for batch {i // batch_size}"
                    )
                    self.logger.error(f"Error: {e}")
                    self.logger.error(f"Files affected: {list(batch_files)}")
                    raise

                try:
                    embedding_dim = len(embeddings[0])
                    table = pa.table(
                        {
                            "object_id": pa.array(
                                batch_df["object_id"].tolist(), type=pa.string()
                            ),
                            "file": pa.array(
                                batch_df["file"].tolist(), type=pa.string()
                            ),
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

                    # Write to temporary file
                    if writer is None:
                        writer = pq.ParquetWriter(
                            temp_output,
                            table.schema,
                            compression="zstd",
                            compression_level=9,
                        )
                        self.logger.info(
                            f"Created ParquetWriter for temporary file: {temp_output}"
                        )

                    writer.write_table(table)
                    last_successful_batch = i // batch_size
                    total_embedded += len(batch_df)
                    files_processed.update(batch_files)

                    self.logger.info(
                        f"Successfully wrote batch {i // batch_size} to temp file"
                    )
                    self.logger.info(f"Total embedded so far: {total_embedded}")

                except Exception as e:
                    self.logger.error(
                        f"Failed to create/write table for batch {i // batch_size}"
                    )
                    self.logger.error(f"Error: {e}")
                    self.logger.error(f"Files affected: {list(batch_files)}")
                    raise

        except KeyboardInterrupt:
            self.logger.warning(
                "KeyboardInterrupt received – merging partial results before exit"
            )
            print(
                "\nInterrupted! Merging partial results so the next run can resume from here..."
            )
            if writer is not None:
                writer.close()
                writer = None
            if os.path.exists(temp_output):
                try:
                    partial_table = pq.read_table(temp_output)
                    if os.path.exists(output_file):
                        existing_table = pq.read_table(output_file)
                        combined_table = pa.concat_tables(
                            [existing_table, partial_table]
                        )
                    else:
                        combined_table = partial_table
                    pq.write_table(
                        combined_table,
                        output_file,
                        compression="zstd",
                        compression_level=9,
                    )
                    os.remove(temp_output)
                    print(
                        f"Saved {total_embedded} newly embedded chunks. Next run will resume from there."
                    )
                    self.logger.info(
                        f"Merged {total_embedded} partial chunks into {output_file} on interrupt"
                    )
                except Exception as merge_err:
                    self.logger.error(f"Failed to merge on interrupt: {merge_err}")
                    print(f"Warning: Could not merge partial results: {merge_err}")
            raise
        except Exception as e:
            self.logger.error("=" * 80)
            self.logger.error(f"CRITICAL ERROR during embedding process")
            self.logger.error(f"Last successful batch: {last_successful_batch}")
            self.logger.error(f"Total embedded before error: {total_embedded}")
            self.logger.error(f"Files successfully processed: {len(files_processed)}")
            self.logger.error(f"Exception: {e}")
            self.logger.error("=" * 80)
            raise
        finally:
            if writer is not None:
                writer.close()
                self.logger.info("Closed ParquetWriter for temporary file")

        # Now merge the temporary file with the existing file (if it exists)
        if os.path.exists(temp_output):
            self.logger.info(f"Merging temporary file with existing data...")
            print(f"Merging new embeddings with existing file...")

            try:
                new_table = pq.read_table(temp_output)

                if os.path.exists(output_file) and not force_reprocess:
                    # Read existing and concatenate
                    self.logger.info(f"Reading existing file: {output_file}")
                    existing_table = pq.read_table(output_file)
                    combined_table = pa.concat_tables([existing_table, new_table])
                    self.logger.info(
                        f"Combined table has {len(combined_table)} total rows"
                    )
                else:
                    combined_table = new_table
                    self.logger.info(f"No existing file, using new data only")

                # Write the combined table
                pq.write_table(
                    combined_table,
                    output_file,
                    compression="zstd",
                    compression_level=9,
                )
                self.logger.info(f"Successfully wrote combined data to {output_file}")

                # Remove temporary file
                os.remove(temp_output)
                self.logger.info(f"Removed temporary file: {temp_output}")

            except Exception as e:
                self.logger.error(f"Failed to merge files: {e}")
                self.logger.error(f"Temporary file preserved at: {temp_output}")
                raise

        self.logger.info(f"Successfully embedded {total_embedded} chunks")
        self.logger.info(f"Files processed: {len(files_processed)}")
        print(f"\nDone! Embedded {total_embedded} chunks")
        print(f"Output saved to {output_file}")

        if os.path.exists(output_file):
            size_mb = os.path.getsize(output_file) / (1024 * 1024)
            print(f"File size: {size_mb:.2f} MB")
            self.logger.info(f"Output file size: {size_mb:.2f} MB")

        # Log final statistics
        self.logger.info("=" * 80)
        self.logger.info("FINAL STATISTICS")
        self.logger.info(f"Total files in chunks.parquet: {total_files}")
        self.logger.info(f"Chunks already embedded (skipped): {skipped_chunks}")
        self.logger.info(f"Files newly processed: {len(files_processed)}")
        self.logger.info(f"Total chunks embedded: {total_embedded}")
        self.logger.info(f"Completed at {datetime.now()}")
        self.logger.info("=" * 80)

        return total_embedded


if __name__ == "__main__":
    embedder = MarkdownEmbedder(
        log_file="embedding_process.log",
        max_workers=8,  # concurrent Gemini API requests
    )

    embedder.embed_chunks(
        chunks_file="chunks.parquet",
        output_file="embeddings.parquet",
        batch_size=100,
        embedding_batch_size=50,  # texts per API request
        force_reprocess=True,
    )

    # Force re-embed everything:
    # embedder.embed_chunks(
    #     chunks_file="chunks.parquet",
    #     output_file="embeddings.parquet",
    #     force_reprocess=True,
    # )
