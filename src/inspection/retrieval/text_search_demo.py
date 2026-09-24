"""Demonstrate CLIP's zero-shot text-to-image search on our indexed images."""

from inspection.retrieval.embeddings import embed_text_clip, load_clip
from inspection.retrieval.vector_store import find_similar, get_connection

QUERIES = ["scratch on metal nut", "broken bottle", "crack in capsule"]


def main() -> None:
    model, processor = load_clip()
    conn = get_connection()

    for query in QUERIES:
        print()
        print("Query:", query)
        embedding = embed_text_clip(query, model, processor)
        results = find_similar(conn, embedding, "clip_embedding", k=5)
        for r in results:
            print(f"  {r['category']:12s} {r['defect_type']:15s} distance={r['distance']:.4f}")


if __name__ == "__main__":
    main()