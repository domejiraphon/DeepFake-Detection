from enum import Enum

import cv2
import numpy as np
import torch
import utils.visualizer.GuidedBackprop as GuidedBackprop
from pytorch_grad_cam import (
    GradCAM,
    ScoreCAM,
    GradCAMPlusPlus,
    AblationCAM,
    XGradCAM,
    EigenCAM,
    FullGrad,
)


class Visualized_methods(Enum):
    GRADCAM = "gradcam"
    SCORECAM = "scorecam"
    GRADCAMPLUSPLUS = "gradcamplusplus"
    ABLATIONCAM = "ablationcam"
    XGRADCAM = "xgradcam"
    EIGENCAM = "eigencam"
    FULLGRAD = "fullgrad"


_visualized_fn = {
    Visualized_methods.GRADCAM: GradCAM,
    Visualized_methods.SCORECAM: ScoreCAM,
    Visualized_methods.GRADCAMPLUSPLUS: GradCAMPlusPlus,
    Visualized_methods.ABLATIONCAM: AblationCAM,
    Visualized_methods.XGRADCAM: XGradCAM,
    Visualized_methods.EIGENCAM: EigenCAM,
    Visualized_methods.FULLGRAD: FullGrad,
}


def get_visualizer(vis_type, model, target_layers, devices):
    if vis_type.lower() in [
        "gradcam",
        "scorecam",
        "gradcamplusplus",
        "ablationcam",
        "xgradcam",
        "eigencam",
        "fullgrad",
    ]:
        return _visualized_fn[Visualized_methods(vis_type.lower())](
            model=model, target_layers=target_layers,
            use_cuda=True if devices == "cuda" else False,
        )
    elif vis_type.lower() == "guided":
        return GuidedBackprop(model, target_layer=0, devices=devices)


def heat_map(
        visualization_type, model, visualize_model, visualized_method, img,
        target, sel=4,
):
    if sel > 0:
        sel = np.random.choice(range(img.shape[0]), sel, replace=False)
        # sel = np.random.randint(0, img.shape[0], sel)
        img, target = img[sel], target[sel]
    if visualization_type == "guided":
        visualize_model.load_state_dict(model.state_dict())

        gradient = visualized_method.generate_gradients(img, target)
        pos_saliency, neg_saliency = get_positive_negative_saliency(gradient)

        mask = torch.clamp(torch.sum(neg_saliency, dim=1, keepdims=True), 0.0,
                           1.0)

        cat_img = torch.cat(
            [
                neg_saliency,
                0.2 * img + 0.8 * img * mask,
                img,
            ],
            3,
        )

        return cat_img, target[sel]
    elif visualization_type in [
        "eigen",
        "gradcam",
        "scorecam",
        "gradcamplusplus",
        "ablationcam",
        "xgradcam",
        "eigencam",
        "fullgrad",
    ]:
        visualized_method.model.load_state_dict(model.state_dict())
        grayscale_cam = visualized_method(input_tensor=img, targets=target)

        visualization = show_cam_on_image(img, grayscale_cam, use_rgb=True)

        cat_img = torch.cat(
            [
                visualization,
                img[:, :3],
            ],
            3,
        )

        return cat_img, target


def get_positive_negative_saliency(gradient):
    """
        Generates positive and negative saliency maps based on the gradient
    Args:
        gradient (numpy arr): Gradient of the operation to visualize
    returns:
        pos_saliency ( )
    """

    minn = torch.min(gradient.reshape(gradient.shape[0], -1), dim=1)[0]

    maxx = torch.max(gradient.reshape(gradient.shape[0], -1), dim=1)[0]

    pos_saliency = torch.maximum(torch.zeros_like(gradient), gradient) / (
            maxx[:, None, None, None] + 1e-12
    )
    neg_saliency = torch.maximum(torch.zeros_like(gradient), -gradient) / (
            -minn[:, None, None, None] - 1e-12
    )
    return pos_saliency, neg_saliency


def show_cam_on_image(
        img: torch.tensor,
        mask: np.ndarray,
        use_rgb: bool = False,
        colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.
    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :returns: The default image with the cam overlay.
    """

    temp_mask = []
    for batch in mask:
        heatmap = cv2.applyColorMap(np.uint8(255 * batch), colormap)
        if use_rgb:
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        temp_mask.append(heatmap)
    temp_mask = np.stack(temp_mask, 0) / 255
    temp_mask = torch.from_numpy(temp_mask).to(img.get_device()).permute(
        [0, 3, 1, 2])

    cam = temp_mask + img[:, :3]

    cam = cam / torch.max(cam.reshape(cam.shape[0], -1), dim=-1)[0][:, None,
                None, None]

    return cam
