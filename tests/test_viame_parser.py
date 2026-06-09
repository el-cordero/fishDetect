from fishdetect.viame import parse_viame_csv


def test_viame_parser_reads_rows_and_reorders_boxes(synthetic_dataset):
    rows = parse_viame_csv(synthetic_dataset / "annotations.viame.csv", synthetic_dataset)
    assert len(rows) == 3
    assert rows[0]["class_name"] == "Acantharus coeruleus"
    assert rows[1]["bbox_x1"] == 40
    assert rows[1]["bbox_y1"] == 50
    assert rows[1]["bbox_x2"] == 80
    assert rows[1]["bbox_y2"] == 90
