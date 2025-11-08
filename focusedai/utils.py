from typing import Optional
import torch
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, classification_report

@torch.no_grad()
def evaluate_logits(logits: torch.Tensor, targets: torch.Tensor, threshold: Optional[float] = None, neutral_index: int = 1):
    # Compute macro-F1 and confusion matrix. If threshold is set, map low-confidence predictions to 'neutral'.
    probs = torch.softmax(logits, dim=-1)
    max_prob, preds = probs.max(dim=-1)

    if threshold is not None:
        # map low confidence to neutral
        under = max_prob < threshold
        preds[under] = neutral_index

    y_true = targets.cpu().numpy()
    y_pred = preds.cpu().numpy()
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=False)
    return macro_f1, cm, report

def device_auto() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
