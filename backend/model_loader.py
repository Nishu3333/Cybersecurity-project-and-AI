import os, json, joblib
from utils.config import Config

def load_latest_model():
    """
    Returns (model, threshold, meta)
    If no model is available, returns (None, 0.7, {}).
    """
    meta_path = os.path.join(Config.MODELS_DIR, "training_metadata.json")
    if not os.path.exists(meta_path):
        return None, 0.7, {}

    with open(meta_path, "r") as f:
        meta = json.load(f)

    model_path = meta.get("model_path")
    if not model_path or not os.path.exists(model_path):
        return None, float(meta.get("threshold", 0.7)), meta

    model = joblib.load(model_path)
    thr = float(meta.get("threshold", 0.7))
    # allow both keys: "features" or "train_features"
    feats = meta.get("features", meta.get("train_features", []))
    meta["features"] = feats
    return model, thr, meta
