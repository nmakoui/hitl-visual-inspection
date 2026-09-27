"""Convert NEU-DET's PASCAL-VOC-style XML annotations into COCO format."""

import xml.etree.ElementTree as ET
from pathlib import Path


def parse_voc_annotation(xml_path: Path) -> dict:
    """Parse one VOC-style XML file into filename, image size, and objects.

    Each object's bbox is kept in VOC's [xmin, ymin, xmax, ymax] form here;
    the COCO [x, y, width, height] conversion happens in build_coco_dataset,
    so this function's output stays a faithful, testable copy of the XML.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)

    objects = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        bndbox = obj.find("bndbox")
        objects.append(
            {
                "name": name,
                "xmin": int(float(bndbox.find("xmin").text)),
                "ymin": int(float(bndbox.find("ymin").text)),
                "xmax": int(float(bndbox.find("xmax").text)),
                "ymax": int(float(bndbox.find("ymax").text)),
            }
        )

    return {
        "filename": root.find("filename").text,
        "width": width,
        "height": height,
        "objects": objects,
    }


def build_coco_dataset(annotations_dir: Path, categories: list[str]) -> dict:
    """Build a full COCO-format dict from a directory of VOC XML files.

    Category IDs are assigned by position in `categories` (0-indexed), so
    the same list must be used consistently wherever these IDs are read
    back - see configs/neu_det.yaml.
    """
    category_to_id = {name: i for i, name in enumerate(categories)}

    images = []
    annotations = []
    annotation_id = 0

    xml_paths = sorted(annotations_dir.glob("*.xml"))
    for image_id, xml_path in enumerate(xml_paths):
        parsed = parse_voc_annotation(xml_path)

        images.append(
            {
                "id": image_id,
                "file_name": parsed["filename"],
                "width": parsed["width"],
                "height": parsed["height"],
            }
        )

        for obj in parsed["objects"]:
            x = obj["xmin"]
            y = obj["ymin"]
            box_width = obj["xmax"] - obj["xmin"]
            box_height = obj["ymax"] - obj["ymin"]

            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_to_id[obj["name"]],
                    "bbox": [x, y, box_width, box_height],
                    "area": box_width * box_height,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    coco_categories = [{"id": i, "name": name} for i, name in enumerate(categories)]

    return {"images": images, "annotations": annotations, "categories": coco_categories}
if __name__ == "__main__":
    import json

    import yaml

    config = yaml.safe_load(Path("configs/neu_det.yaml").read_text())
    categories = config["neu_det_categories"]

    root = Path("data/raw/neu_det/NEU-DET")
    output_dir = Path("data/processed/neu_det_coco")
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "validation"):
        annotations_dir = root / split / "annotations"
        coco = build_coco_dataset(annotations_dir, categories)

        output_path = output_dir / f"{split}.json"
        output_path.write_text(json.dumps(coco, indent=2))

        print(f"{split}: {len(coco['images'])} images, {len(coco['annotations'])} annotations "
              f"-> {output_path}")