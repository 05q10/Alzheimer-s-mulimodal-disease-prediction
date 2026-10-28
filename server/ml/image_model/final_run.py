import sys
import os
import pathlib
import io
from PIL import Image
import torch
import torchvision.transforms as T

# Fix for Linux-trained fastai model loaded on Windows (temporary patch only during load)
_orig_posix = getattr(pathlib, "PosixPath", None)
try:
    if sys.platform.startswith("win"):
        pathlib.PosixPath = pathlib.WindowsPath
    from fastai.vision.all import load_learner
    BASE_DIR = os.path.dirname(__file__)
    learn = load_learner(f"{BASE_DIR}/model.pkl")
finally:
    if sys.platform.startswith("win") and _orig_posix is not None:
        pathlib.PosixPath = _orig_posix

# Temperature scaling constant from config
from ml.twin.constants import MRI_TEMPERATURE

# Preprocessing transforms matching fastai ResNet34 inference
_val_transforms = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def predict_mri_image(image_input, temperature: float = MRI_TEMPERATURE):
    """
    Run ResNet34 inference on an MRI slice image with temperature scaling.
    image_input: str path, bytes, or PIL Image.

    Returns:
        {
            "predicted_class": "Mild",  # NonDemented | VeryMild | Mild | Moderate
            "probs": {
                "NonDemented": 0.0001,
                "VeryMild": 0.0008,
                "Mild": 0.9989,
                "Moderate": 0.0001
            },
            "mapped_probs": {
                "CN": 0.0001,
                "MCI": 0.0008,
                "AD": 0.9991
            }
        }
    """
    if isinstance(image_input, (str, os.PathLike)):
        pil_img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, (bytes, bytearray)):
        pil_img = Image.open(io.BytesIO(image_input)).convert("RGB")
    elif isinstance(image_input, Image.Image):
        pil_img = image_input.convert("RGB")
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    tensor_img = _val_transforms(pil_img).unsqueeze(0)

    learn.model.eval()
    with torch.no_grad():
        logits = learn.model(tensor_img)
        # Apply temperature scaling to logits
        temp = max(0.1, float(temperature))
        scaled_logits = logits / temp
        probs_tensor = torch.softmax(scaled_logits, dim=-1)[0]

    # Model vocabulary: ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
    raw_mild = float(probs_tensor[0])
    raw_mod = float(probs_tensor[1])
    raw_non = float(probs_tensor[2])
    raw_very_mild = float(probs_tensor[3])

    probs = {
        "NonDemented": round(raw_non, 4),
        "VeryMild": round(raw_very_mild, 4),
        "Mild": round(raw_mild, 4),
        "Moderate": round(raw_mod, 4),
    }

    # Best class among the 4
    class_order = ["NonDemented", "VeryMild", "Mild", "Moderate"]
    best_class = max(class_order, key=lambda c: probs[c])

    # Linear mapping to ADNI clinical stages:
    # NonDemented -> CN
    # VeryMild -> MCI
    # Mild + Moderate -> AD
    mapped_cn = raw_non
    mapped_mci = raw_very_mild
    mapped_ad = raw_mild + raw_mod

    mapped_sum = mapped_cn + mapped_mci + mapped_ad
    if mapped_sum > 0:
        mapped_cn /= mapped_sum
        mapped_mci /= mapped_sum
        mapped_ad /= mapped_sum

    mapped_probs = {
        "CN": round(mapped_cn, 4),
        "MCI": round(mapped_mci, 4),
        "AD": round(mapped_ad, 4),
    }

    return {
        "predicted_class": best_class,
        "probs": probs,
        "mapped_probs": mapped_probs,
    }


def predict_image(image_input):
    """Legacy endpoint wrapper for backward compatibility."""
    res = predict_mri_image(image_input)
    # Map back to legacy class string
    legacy_map = {
        "NonDemented": "NonDemented",
        "VeryMild": "VeryMildDemented",
        "Mild": "MildDemented",
        "Moderate": "ModerateDemented",
    }
    pred_str = legacy_map.get(res["predicted_class"], res["predicted_class"])
    confidence = res["probs"][res["predicted_class"]]
    return {
        "prediction": pred_str,
        "confidence": round(confidence, 5)
    }
