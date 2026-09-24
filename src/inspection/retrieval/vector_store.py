"""Load image embeddings into PostgreSQL (pgvector) and query similar images."""

import csv
import os
from pathlib import Path

import numpy as np
import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

load_dotenv()


def get_connection() -> psycopg.Connection:
    """Open a connection to the local Postgres database, using .env credentials."""
    conn = psycopg.connect(
        host="localhost",
        port=5432,
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        dbname=os.environ["POSTGRES_DB"],
    )
    register_vector(conn)
    return conn


def create_tables(conn: psycopg.Connection) -> None:
    """Create the images table, with vector columns for both embedding types."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS images (
            id SERIAL PRIMARY KEY,
            category TEXT NOT NULL,
            split TEXT NOT NULL,
            defect_type TEXT NOT NULL,
            image_path TEXT NOT NULL,
            dinov2_embedding vector(768),
            clip_embedding vector(512)
        );
    """)
    conn.commit()


def load_embeddings(conn: psycopg.Connection, embeddings_dir: Path) -> int:
    """Load metadata.csv plus both .npy embedding files into the images table.

    Returns the number of rows inserted.
    """
    dinov2 = np.load(embeddings_dir / "dinov2_embeddings.npy")
    clip = np.load(embeddings_dir / "clip_embeddings.npy")

    with open(embeddings_dir / "metadata.csv", newline="") as f:
        rows = list(csv.DictReader(f))

    with conn.cursor() as cur:
        for row in rows:
            i = int(row["index"])
            cur.execute(
                """
                INSERT INTO images
                    (category, split, defect_type, image_path, dinov2_embedding, clip_embedding)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    row["category"],
                    row["split"],
                    row["defect_type"],
                    row["image_path"],
                    dinov2[i],
                    clip[i],
                ),
            )
    conn.commit()
    return len(rows)


def add_hnsw_indexes(conn: psycopg.Connection) -> None:
    """Add HNSW approximate-nearest-neighbour indexes for both embedding columns."""
    conn.execute(
        "CREATE INDEX IF NOT EXISTS dinov2_hnsw_idx "
        "ON images USING hnsw (dinov2_embedding vector_cosine_ops);"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS clip_hnsw_idx "
        "ON images USING hnsw (clip_embedding vector_cosine_ops);"
    )
    conn.commit()


if __name__ == "__main__":
    conn = get_connection()
    create_tables(conn)
    n = load_embeddings(conn, Path("data/processed/embeddings"))
    add_hnsw_indexes(conn)
    print(f"Loaded {n} rows and created HNSW indexes.")

def find_similar(
    conn: psycopg.Connection, embedding: np.ndarray, column: str, k: int = 5
) -> list[dict]:
    """Return the k most similar images to the given embedding vector.

    `column` must be either "dinov2_embedding" or "clip_embedding".
    """
    if column not in ("dinov2_embedding", "clip_embedding"):
        raise ValueError(f"Unknown embedding column: {column}")

    rows = conn.execute(
        f"""
        SELECT id, category, split, defect_type, image_path,
               {column} <=> %s AS distance
        FROM images
        ORDER BY {column} <=> %s
        LIMIT %s
        """,
        (embedding, embedding, k),
    ).fetchall()

    columns = ["id", "category", "split", "defect_type", "image_path", "distance"]
    return [dict(zip(columns, row, strict=True)) for row in rows]