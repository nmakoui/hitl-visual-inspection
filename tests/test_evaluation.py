import numpy as np
import pytest

from inspection.retrieval.evaluation import evaluate_retrieval
from inspection.retrieval.vector_store import create_tables, get_connection


@pytest.fixture
def conn():
    connection = get_connection()
    connection.execute("DROP TABLE IF EXISTS test_images;")
    connection.commit()
    create_tables(connection, table_name="test_images")
    yield connection
    connection.execute("DROP TABLE IF EXISTS test_images;")
    connection.commit()
    connection.close()


def _insert(conn, category, defect_type, vec):
    dummy_clip = np.zeros(512, dtype=np.float32)
    conn.execute(
        """
        INSERT INTO test_images (category, split, defect_type, image_path,
                             dinov2_embedding, clip_embedding)
        VALUES (%s, 'test', %s, 'fake.png', %s, %s)
        """,
        (category, defect_type, vec, dummy_clip),
    )
    conn.commit()


def test_evaluate_retrieval_recall_and_precision(conn):
    base = np.ones(768, dtype=np.float32)
    # Three near-identical vectors, same (category, defect_type) group.
    _insert(conn, "bottle", "scratch", base * 1.00)
    _insert(conn, "bottle", "scratch", base * 1.01)
    _insert(conn, "bottle", "scratch", base * 0.99)
    # One unrelated image, far away, alone in its own group.
    _insert(conn, "hazelnut", "crack", -base)

    result = evaluate_retrieval(conn, "dinov2_embedding", k=2, table_name="test_images")

    assert result["num_queries_evaluated"] == 3
    assert result["num_queries_skipped"] == 1
    assert result["mean_precision_at_k"] == pytest.approx(1.0)
    assert result["mean_recall_at_k"] == pytest.approx(1.0)