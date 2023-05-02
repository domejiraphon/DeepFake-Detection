import sys

import PIL
from PIL import JpegImagePlugin, PngImagePlugin, Image
import albumentations as alb
import cv2
import mediapipe as mp
import numpy as np
import torch

from facenet_pytorch import MTCNN
from facenet_pytorch.models import utils as facenet_utils
from loguru import logger
from mediapipe.python.solutions.drawing_utils import \
    _normalized_to_pixel_coordinates
from torchvision import transforms
import matplotlib.pyplot as plt 
sys.path.append("./utils")
from retinaface.pre_trained_models import get_model
import DeepLabV3
from occlusion.occlusion import Occlusion_Generator

class GetFace(torch.nn.Module):
    def __init__(self, image_size=224, device='cuda'):
        super().__init__()
        self.image_size = image_size
        self.detector = MTCNN(keep_all=True, device=device)
        self.margin = 10

    def extract_face(self, img, box):
        """Extract face + margin from PIL Image given bounding box.
        Arguments:
            img {PIL.Image} -- A PIL Image.
            box {numpy.ndarray} -- Four-element bounding box.
            image_size {int} -- Output image size in pixels. The image will be square.
            margin {int} -- Margin to add to bounding box, in terms of pixels in the final image.
                Note that the application of the margin differs slightly from the davidsandberg/facenet
                repo, which applies the margin to the original image before resizing, making the margin
                dependent on the original image size.
        Returns:
            torch.tensor -- tensor representing the extracted face.
        """
        margin = [
            self.margin * (box[2] - box[0]) / (self.image_size - self.margin),
            self.margin * (box[3] - box[1]) / (self.image_size - self.margin),
        ]
        raw_image_size = facenet_utils.detect_face.get_size(img)
        box = [
            int(max(box[0] - margin[0] / 2, 0)),
            int(max(box[1] - margin[1] / 2, 0)),
            int(min(box[2] + margin[0] / 2, raw_image_size[0])),
            int(min(box[3] + margin[1] / 2, raw_image_size[1])),
        ]

        face = facenet_utils.detect_face.crop_resize(img, box, self.image_size)
        face = PIL.Image.fromarray(face)
        return face

    def forward(self, img, not_pil=False, degrade=False):
        if not not_pil and not torch.is_tensor(img): 
            img = np.array(img)
        if img is not None:
            out = self.detector.detect(img, landmarks=True)
            if degrade: return out[0]
            if out[0] is not None:
                box = out[0][0]
              
                face = self.extract_face(img, box)
                return face
            else:
                return None
        else:
            return None

class Face_Mesh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            refine_landmarks=True,
            max_num_faces=2,
            min_detection_confidence=0.5
        )
        self.num_keypoints = 478  # Number from mediapipe

    def forward(self, image: PIL.Image):
        """
        Return: Dictionary of keypoint where the first element is x axis, 
                second element is y axis
        """
        image = np.array(image)
        results = self.face_mesh.process(image)
        idx_to_coordinates = []
        image_rows, image_cols, _ = image.shape
        if results.multi_face_landmarks is not None:
            for landmark in results.multi_face_landmarks[0].landmark:
                landmark_px = _normalized_to_pixel_coordinates(landmark.x,
                                                               landmark.y,
                                                               image_cols,
                                                               image_rows)

                if landmark_px:  idx_to_coordinates.append(
                    np.array(landmark_px)[None])
            idx_to_coordinates = np.concatenate(idx_to_coordinates, 0)
            """
            if (idx_to_coordinates.shape[0] < self.num_keypoints):
                idx_to_coordinates = np.concatenate([
                    idx_to_coordinates,
                    np.zeros((self.num_keypoints - idx_to_coordinates.shape[0], 2))], 
                0)
            """
            return idx_to_coordinates

        else:
            return None

    def create_mask(self, image: Image):
        face_keypoints = self(image)
        if face_keypoints is None: return None
        image = np.array(image)
        mask = np.zeros_like(image[..., 0])
        cv2.fillConvexPoly(mask, cv2.convexHull(face_keypoints), 1.)
        return mask

class RandomDownScale(alb.core.transforms_interface.ImageOnlyTransform):
    def apply(self, img, **params):
        return self.randomdownscale(img)

    def randomdownscale(self, img):
        keep_ratio = True
        keep_input_shape = True
        H, W, C = img.shape
        ratio_list = [2, 4]
        r = ratio_list[np.random.randint(len(ratio_list))]
        img_ds = cv2.resize(img, (int(W / r), int(H / r)),
                            interpolation=cv2.INTER_NEAREST)
        if keep_input_shape:
            img_ds = cv2.resize(img_ds, (W, H), interpolation=cv2.INTER_LINEAR)

        return img_ds

class RetinaGetFace(torch.nn.Module):
    def __init__(self, image_size, device='cuda'):
        super().__init__()
        self.model = get_model("resnet50_2020-07-20",
                               max_size=image_size,
                               device=device)
        self.model.eval()

    def forward(self, frame, image_size=(380, 380), return_pil=False):
        if type(frame) in [JpegImagePlugin.JpegImageFile,
                    PngImagePlugin.PngImageFile]:
            frame = np.array(frame)
       
        faces = self.model.predict_jsons(frame)

        if len(faces) == 0:
            return None

        croppedfaces = []
        for face_idx in range(len(faces)):
            if len(faces[face_idx]['bbox']) == 0:
                return None
            x0, y0, x1, y1 = faces[face_idx]['bbox']
            bbox = np.array([[x0, y0], [x1, y1]])
            m = cv2.resize(self.crop_face(frame, None, bbox, False,
                                crop_by_bbox=True, only_img=True,
                                phase='test'),
                           dsize=image_size)
           
            if return_pil: m = PIL.Image.fromarray(m)
            return m

    def crop_face(self, img, landmark=None, bbox=None,
                  margin=False, crop_by_bbox=True, abs_coord=False,
                  only_img=False, phase='train'
                  ):
        assert phase in ['train', 'val', 'test']

        # crop face------------------------------------------
        H, W = len(img), len(img[0])

        assert landmark is not None or bbox is not None

        H, W = len(img), len(img[0])

        if crop_by_bbox:
            x0, y0 = bbox[0]
            x1, y1 = bbox[1]
            w = x1 - x0
            h = y1 - y0
            w0_margin = w / 4  # 0#np.random.rand()*(w/8)
            w1_margin = w / 4
            h0_margin = h / 4  # 0#np.random.rand()*(h/5)
            h1_margin = h / 4
        else:
            x0, y0 = landmark[:68, 0].min(), landmark[:68, 1].min()
            x1, y1 = landmark[:68, 0].max(), landmark[:68, 1].max()
            w = x1 - x0
            h = y1 - y0
            w0_margin = w / 8  # 0#np.random.rand()*(w/8)
            w1_margin = w / 8
            h0_margin = h / 2  # 0#np.random.rand()*(h/5)
            h1_margin = h / 5

        if margin:
            w0_margin *= 4
            w1_margin *= 4
            h0_margin *= 2
            h1_margin *= 2
        elif phase == 'train':
            w0_margin *= (np.random.rand() * 0.6 + 0.2)  # np.random.rand()
            w1_margin *= (np.random.rand() * 0.6 + 0.2)  # np.random.rand()
            h0_margin *= (np.random.rand() * 0.6 + 0.2)  # np.random.rand()
            h1_margin *= (np.random.rand() * 0.6 + 0.2)  # np.random.rand()
        else:
            w0_margin *= 0.5
            w1_margin *= 0.5
            h0_margin *= 0.5
            h1_margin *= 0.5

        y0_new = max(0, int(y0 - h0_margin))
        y1_new = min(H, int(y1 + h1_margin) + 1)
        x0_new = max(0, int(x0 - w0_margin))
        x1_new = min(W, int(x1 + w1_margin) + 1)

        img_cropped = img[y0_new:y1_new, x0_new:x1_new]
        if landmark is not None:
            landmark_cropped = np.zeros_like(landmark)
            for i, (p, q) in enumerate(landmark):
                landmark_cropped[i] = [p - x0_new, q - y0_new]
        else:
            landmark_cropped = None
        if bbox is not None:
            bbox_cropped = np.zeros_like(bbox)
            for i, (p, q) in enumerate(bbox):
                bbox_cropped[i] = [p - x0_new, q - y0_new]
        else:
            bbox_cropped = None

        if only_img:
            return img_cropped
        if abs_coord:
            return img_cropped, landmark_cropped, bbox_cropped, (
            y0 - y0_new, x0 - x0_new, y1_new - y1,
            x1_new - x1), y0_new, y1_new, x0_new, x1_new
        else:
            return img_cropped, landmark_cropped, bbox_cropped, (
            y0 - y0_new, x0 - x0_new, y1_new - y1, x1_new - x1)

class DeepLabV3_Segment(torch.nn.Module):
    def __init__(self, model, num_classes, output_stride,
                 ckpt, img_shape=224, device='cuda'):
        super().__init__()
        self.device = device
        self.model = DeepLabV3.modeling.__dict__[model](
            num_classes=num_classes,
            output_stride=output_stride).to(device)

        checkpoint = torch.load(ckpt)
        logger.info(f"Use DeepLabV3: {model}")
        logger.info(f"Load DeepLabV3: {ckpt}")
        
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        self.resize = transforms.Compose(
            [transforms.Resize((img_shape, img_shape))])

    def forward(self, img):
        if type(img) == PIL.JpegImagePlugin.JpegImageFile:
            img = torch.from_numpy(np.array(img)).float().to(self.device)
            img = img.permute([2, 0, 1])
            if torch.max(img) > 1:
                img /= 255
        elif type(img) == np.ndarray:
            img = torch.from_numpy(img).float().to(self.device)
            img = img.permute([2, 0, 1])
            if torch.max(img) > 1:
                img = img /255
        img = img[None].to(self.device)
        
        segment_map = self.model(img)
        segment_map = self.resize(segment_map)

        segment_map = (segment_map[0, 1:] > 0.5).float().detach().cpu()
      
        return segment_map

class Occlusion(torch.nn.Module):
    def __init__(self, config, segment_model, img_shape=224):
        super().__init__()
        self.segment_model = segment_model
        self.occlusion = Occlusion_Generator(config.occluder_img_dataset,
                                             config.occluder_mask_dataset,
                                             img_shape)
        self.resize = transforms.Compose([transforms.ToTensor(),
                                          transforms.Resize(
                                              (img_shape, img_shape))])

    def forward(self, img, to_tensor=True, img2=None):
      
        src_img = img
        
        src_mask = self.segment_model(img)
 
        src_mask = src_mask.detach().permute([1, 2, 0]).cpu().numpy()
        def convert(img):
            if type(img) == PIL.JpegImagePlugin.JpegImageFile:
                src_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            elif type(img) == np.ndarray and np.max(img) <= 1:
                src_img = cv2.cvtColor(np.array(img*255).astype(np.uint8),
                                    cv2.COLOR_RGB2BGR)
            elif type(img) == torch.Tensor:
                src_img = img.permute([1,2, 0]).cpu().numpy()
                src_img = cv2.cvtColor(np.array(img*255).astype(np.uint8),
                                    cv2.COLOR_RGB2BGR)
            elif type(img) == np.ndarray and np.max(img) > 1:
                src_img = cv2.cvtColor(img,
                                    cv2.COLOR_RGB2BGR)
            else:
                src_img = img 
            return src_img

        src_img = convert(src_img)
        if img2 is not None:
            img2 = convert(img2)
            (occlude_img, occlude_mask,
                occlude_img2, occlude_mask2) = self.occlusion.occlude_images(src_img,
                                                                  src_mask,
                                                                  img2)
            if to_tensor:
                occlude_img = self.resize(occlude_img)
                occlude_mask = self.resize(occlude_mask)
                occlude_img2 = self.resize(occlude_img2)
                occlude_mask2 = self.resize(occlude_mask2)
            else:
                occlude_img = occlude_img / 255
                occlude_mask = occlude_mask / 255
                occlude_img2 = occlude_img2 / 255
                occlude_mask2 = occlude_mask2 / 255
            return occlude_img, occlude_mask, occlude_img2, occlude_mask2
        else:
            occlude_img, occlude_mask = self.occlusion.occlude_images(src_img,
                                                                  src_mask,
                                                                  img2)
        
            if to_tensor:
                occlude_img = self.resize(occlude_img)
                occlude_mask = self.resize(occlude_mask)
            else:
                occlude_img = occlude_img / 255
                occlude_mask = occlude_mask / 255
            return occlude_img, occlude_mask
