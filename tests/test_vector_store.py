import numpy as np
import pytest

from inspection.retrieval.vector_store import (
    add_hnsw_indexes,
    create_tables,
    find_similar,
    get_connection,
)


@pytest.fixture
def conn():
    """A real connection to the local Postgres, with a clean 'test_images' table."""
    connection = get_connection()
    connection.execute("DROP TABLE IF EXISTS test_images;")
    connection.commit()
    create_tables(connection, table_name="test_images")
    yield connection
    connection.execute("DROP TABLE IF EXISTS test_images;")
    connection.commit()
    connection.close()


def _insert_fake_image(conn, category, dinov2_vec, clip_vec):
    conn.execute(
        """
        INSERT INTO test_images (category, split, defect_type, image_path,
                             dinov2_embedding, clip_embedding)
        VALUES (%s, 'test', 'good', 'fake.png', %s, %s)
        """,
        (category, dinov2_vec, clip_vec),
    )
    conn.commit()


def test_find_similar_returns_closest_match_first(conn):
    add_hnsw_indexes(conn, table_name="test_images")

    close_vec = np.ones(768, dtype=np.float32)
    far_vec = -np.ones(768, dtype=np.float32)
    clip_dummy = np.zeros(512, dtype=np.float32)

    _insert_fake_image(conn, "close_match", close_vec, clip_dummy)
    _insert_fake_image(conn, "far_match", far_vec, clip_dummy)

    query_vec = np.ones(768, dtype=np.float32) * 0.9
    results = find_similar(conn, query_vec, "dinov2_embedding", k=2, table_name="test_images")

    assert results[0]["category"] == "close_match"
    assert results[1]["category"] == "far_match"
    assert results[0]["distance"] < results[1]["distance"]