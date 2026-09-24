"""Evaluate similarity retrieval: mean precision@k and recall@k, per embedding type."""

from collections import defaultdict

import numpy as np
import psycopg

from inspection.retrieval.vector_store import find_similar


def _fetch_all_rows(conn: psycopg.Connection, table_name: str = "images") -> list[dict]:
    rows = conn.execute(
        f"SELECT id, category, defect_type, dinov2_embedding, clip_embedding FROM {table_name}"
    ).fetchall()
    columns = ["id", "category", "defect_type", "dinov2_embedding", "clip_embedding"]
    return [dict(zip(columns, row, strict=True)) for row in rows]


def evaluate_retrieval(
    conn: psycopg.Connection, column: str, k: int = 5, table_name: str = "images"
) -> dict:
    """Compute mean precision@k and recall@k for one embedding column.

    A "hit" is a retrieved image sharing the same category and defect_type
    as the query image (per the project's definition of a relevant match).
    Every image in the table is used as a query in turn. Queries whose
    (category, defect_type) group has no other member besides itself are
    skipped, since recall is undefined with zero relevant items.
    """
    rows = _fetch_all_rows(conn, table_name)

    group_counts: dict[tuple[str, str], int] = defaultdict(int)
    for row in rows:
        group_counts[(row["category"], row["defect_type"])] += 1

    precisions = []
    recalls = []
    skipped = 0

    for row in rows:
        relevant_count = group_counts[(row["category"], row["defect_type"])] - 1

        if relevant_count == 0:
            skipped += 1
            continue

        query_embedding = row[column].to_numpy()
        results = find_similar(
            conn, query_embedding, column, k=k, exclude_id=row["id"], table_name=table_name
        )

        hits = sum(
            1
            for r in results
            if r["category"] == row["category"] and r["defect_type"] == row["defect_type"]
        )

        precisions.append(hits / k)
        recalls.append(hits / relevant_count)

    return {
        "mean_precision_at_k": float(np.mean(precisions)),
        "mean_recall_at_k": float(np.mean(recalls)),
        "num_queries_evaluated": len(precisions),
        "num_queries_skipped": skipped,
        "k": k,
    }


if __name__ == "__main__":
    from inspection.retrieval.vector_store import get_connection

    conn = get_connection()
    for column in ("dinov2_embedding", "clip_embedding"):
        result = evaluate_retrieval(conn, column, k=5)
        print(column, result)