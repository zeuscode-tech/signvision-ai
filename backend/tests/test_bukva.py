import base64
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from app.ml.bukva import model_info, predict, preprocess


def frame() -> str:
    output=BytesIO();Image.new("RGB",(320,240),(20,30,40)).save(output,format="JPEG")
    return "data:image/jpeg;base64,"+base64.b64encode(output.getvalue()).decode()


def test_bukva_preprocess_shape():
    tensor=preprocess([frame()]*8,8)
    assert tensor.shape==(1,8,3,224,224)
    assert tensor.dtype==np.float32
    assert np.isfinite(tensor).all()


def test_bukva_real_model_and_rejection():
    path=Path(__file__).resolve().parents[2]/"models/bukva_mobilenetv2_tsm.onnx"
    info=model_info(path)
    assert info["ready"] and info["window_size"]==8 and len(info["classes"])==34
    result=predict(path,[frame()]*8)
    assert len(result["top3"])==3
    assert not result["accepted"]
    assert result["label"] is None
