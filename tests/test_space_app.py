from PIL import Image

from inspection.space.app import SUBSET_DIR, find_similar_cases


def test_find_similar_cases_returns_k_results():
    sample_image_path = next((SUBSET_DIR / "images").glob("*.png"))
    query_image = Image.open(sample_image_path).convert("RGB")

    results = find_similar_cases(query_image, k=5)

    assert len(results) == 5
    for image, label in results:
        assert isinstance(image, Image.Image)
        assert isinstance(label, str)