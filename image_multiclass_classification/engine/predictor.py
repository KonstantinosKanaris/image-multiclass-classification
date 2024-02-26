from typing import Tuple

import torch


def single_image_prediction(
    model: torch.nn.Module, image: torch.Tensor, device: str = "cpu"
) -> Tuple[int, float]:
    """Predicts the label and prediction probabilities
    of target image.

    Args:
        model (torch.nn.Module):
            A trained PyTorch model to use to make the prediction.
        image (torch.Tensor):
            An input image with [`batch_size`, `color_channels`,
            `height`, `width`] dimensions.
        device (str, optional):
            A target device to compute on, i.e., `cuda` or `cpu`
            (default: `cpu`).
    """
    model.to(device)
    model.eval()
    with torch.inference_mode():
        y_logits = model(image.to(device))
    y_pred_probs = torch.softmax(y_logits, dim=1)
    y_pred_label = torch.argmax(y_pred_probs, dim=1)

    return int(y_pred_label.item()), y_pred_probs.max().item()
