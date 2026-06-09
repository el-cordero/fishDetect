from fishdetect.dive import load_meta_image_map, parse_dive_json


def test_dive_parser_reads_dive_geometry_and_frame_mapping(synthetic_dataset):
    mapping = load_meta_image_map(synthetic_dataset / "meta.json")
    assert mapping[1] == "000001.png"
    rows = parse_dive_json(synthetic_dataset / "annotations.dive.json", synthetic_dataset / "meta.json", synthetic_dataset)
    assert len(rows) == 3
    assert rows[0]["filename"] == "000000.png"
    assert rows[0]["has_dive_geometry"] is True
    assert rows[0]["has_polygon"] is False
    assert rows[1]["has_polygon"] is False
