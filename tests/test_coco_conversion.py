from inspection.detection.coco_conversion import build_coco_dataset, parse_voc_annotation

FAKE_XML = """<annotation>
    <filename>crazing_1.jpg</filename>
    <size>
        <width>200</width>
        <height>200</height>
        <depth>1</depth>
    </size>
    <object>
        <name>crazing</name>
        <bndbox>
            <xmin>2</xmin>
            <ymin>2</ymin>
            <xmax>193</xmax>
            <ymax>194</ymax>
        </bndbox>
    </object>
</annotation>
"""


def test_parse_voc_annotation(tmp_path):
    xml_path = tmp_path / "crazing_1.xml"
    xml_path.write_text(FAKE_XML)

    parsed = parse_voc_annotation(xml_path)

    assert parsed["filename"] == "crazing_1.jpg"
    assert parsed["width"] == 200
    assert parsed["height"] == 200
    assert len(parsed["objects"]) == 1
    assert parsed["objects"][0]["name"] == "crazing"
    assert parsed["objects"][0] == {
        "name": "crazing",
        "xmin": 2,
        "ymin": 2,
        "xmax": 193,
        "ymax": 194,
    }


def test_build_coco_dataset_converts_bbox_to_xywh(tmp_path):
    (tmp_path / "crazing_1.xml").write_text(FAKE_XML)

    coco = build_coco_dataset(tmp_path, categories=["crazing", "inclusion"])

    assert len(coco["images"]) == 1
    assert len(coco["annotations"]) == 1
    assert len(coco["categories"]) == 2

    annotation = coco["annotations"][0]
    # VOC box was [2, 2, 193, 194] -> COCO [x, y, width, height]
    assert annotation["bbox"] == [2, 2, 191, 192]
    assert annotation["area"] == 191 * 192
    assert annotation["category_id"] == 0  # "crazing" is index 0 in the categories list