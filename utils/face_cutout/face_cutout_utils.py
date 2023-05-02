import numpy as np
from PIL import Image
import PIL

from facenet_pytorch.models.mtcnn import MTCNN
import torch
import sys
import sys

import PIL
import numpy as np
import torch
from PIL import Image
from facenet_pytorch.models.mtcnn import MTCNN

sys.path.append("./utils/face_cutout")
from face_cutout import face_cutout


class Face_Cutout(torch.nn.Module):
    def __init__(self, image_size):
        super().__init__()
        self.mtcnn = MTCNN(margin=0,
                           thresholds=[0.65, 0.75, 0.75],
                           image_size=image_size,
                           device="cuda")

    def forward(self, image: Image):
        if type(image) != PIL.JpegImagePlugin.JpegImageFile:
            image = image.permute([1, 2, 0]).numpy() * 255
            image = Image.fromarray(image.astype(np.uint8))
        batch_boxes, conf, landmarks = self.mtcnn.detect(image, landmarks=True)

        if landmarks is not None:
            landmarks = np.around(landmarks[0]).astype(np.int16)
            # out = sensory_cutout(np.array(image), landmarks=landmarks, cutout_fill=-1)
            # out = convex_hull_cutout(np.array(image), cutout_fill=-1, probability=0.5)

            out = face_cutout(np.array(image), landmarks=landmarks,
                              cutout_fill=0)
            if out is None:
                return np.array(image)
            return out
        else:
            return np.array(image)
