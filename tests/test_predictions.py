from fishdetect.predictions import load_predictions, match_predictions, save_predictions


def test_prediction_json_schema_and_matching(tmp_path):
    predictions = [
        {
            "image_id": "1",
            "filename": "a.png",
            "class_name": "fish",
            "score": 0.9,
            "bbox_x1": 0,
            "bbox_y1": 0,
            "bbox_x2": 10,
            "bbox_y2": 10,
        }
    ]
    path = tmp_path / "predictions.json"
    save_predictions(path, predictions, metadata={"split": "test"})
    loaded = load_predictions(path)
    assert loaded[0]["score"] == 0.9
    gt = [{**predictions[0], "score": ""}]
    matches, fps, fns = match_predictions(gt, loaded)
    assert len(matches) == 1
    assert not fps
    assert not fns
