from pathlib import Path

from inspection.datasets import list_mvtec_samples


def _make_fake_category(root: Path) -> None:
    """Build a tiny MVTec-AD-shaped folder tree for testing."""
    (root / "bottle" / "train" / "good").mkdir(parents=True)
    (root / "bottle" / "test" / "good").mkdir(parents=True)
    (root / "bottle" / "test" / "scratch").mkdir(parents=True)
    (root / "bottle" / "ground_truth" / "scratch").mkdir(parents=True)

    (root / "bottle" / "train" / "good" / "000.png").touch()
    (root / "bottle" / "test" / "good" / "000.png").touch()
    (root / "bottle" / "test" / "scratch" / "000.png").touch()
    (root / "bottle" / "ground_truth" / "scratch" / "000_mask.png").touch()


def test_train_split_has_only_good_images(tmp_path):
    _make_fake_category(tmp_path)
    samples = list_mvtec_samples(tmp_path, "bottle", "train")

    assert len(samples) == 1
    assert samples[0].defect_type == "good"
    assert samples[0].mask_path is None


def test_test_split_includes_good_and_defect(tmp_path):
    _make_fake_category(tmp_path)
    samples = list_mvtec_samples(tmp_path, "bottle", "test")

    defect_types = {s.defect_type for s in samples}
    assert defect_types == {"good", "scratch"}


def test_defect_image_has_matching_mask(tmp_path):
    _make_fake_category(tmp_path)
    samples = list_mvtec_samples(tmp_path, "bottle", "test")

    scratch_sample = next(s for s in samples if s.defect_type == "scratch")
    assert scratch_sample.mask_path is not None
    assert scratch_sample.mask_path.name == "000_mask.png"


def test_good_test_image_has_no_mask(tmp_path):
    _make_fake_category(tmp_path)
    samples = list_mvtec_samples(tmp_path, "bottle", "test")

    good_sample = next(s for s in samples if s.defect_type == "good")
    assert good_sample.mask_path is None
    ...