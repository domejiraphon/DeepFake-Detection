import sys
import imutils
import sys

import imutils

# import cupy as cp

sys.path.append('./utils/face_occlusion_generation')
from utils.utils import *
from utils.colour_transfer import *
from utils.paste_over import *
from utils.random_shape_generator import *


class Occlusion_Generator:
    def __init__(self, args, images_list, occluders_list, seeds):
        self.args = args
        self.image_augmentor = get_src_augmentor()
        self.occluder_augmentor = get_occluder_augmentor()
        self.images_list = images_list
        self.occluders_list = occluders_list
        self.seeds = seeds

    def occlude_images(self, index):
        try:
            image = self.images_list[index]
            occluder = self.occluders_list[index]
            seed = self.seeds[index]

            # set seed
            set_random_seed(seed)

            # get source img and mask
            src_img, src_mask = get_srcNmask(image, self.args["srcImageDir"],
                                             self.args["srcMaskDir"])

            # get occluder img and mask
            if self.args["randomOcclusion"]:
                occluder_img, occluder_mask = get_randomOccluderNmask()
            else:
                occluder_img, occluder_mask = get_occluderNmask(occluder,
                                                                self.args[
                                                                    "occluderDir"],
                                                                self.args[
                                                                    "occluderMaskDir"])

            src_rect = cv2.boundingRect(src_mask)

            # colour transfer
            if self.args["colour_transfer_sot"]:
                try:
                    occluder_img = self.colour_transfer(src_img, src_mask,
                                                        occluder_img, src_rect)
                    occluder_img = occluder_img.detach().cpu().numpy()
                except Exception as e:
                    print(e)
            # augment occluders
            occluder_img, occluder_mask = augment_occluder(
                self.occluder_augmentor, occluder_img, occluder_mask, src_rect)
            # random location around src
            occluder_coord = np.random.uniform([src_rect[0], src_rect[1]],
                                               [src_rect[0] + src_rect[2],
                                                src_rect[1] + src_rect[3]])

            if self.args["rotate_around_center"]:
                src_center = (src_rect[0] + (src_rect[2] / 2),
                              (src_rect[1] + src_rect[3] / 2))
                rotation = angle3pt((src_center[0], occluder_coord[1]),
                                    src_center, occluder_coord)
                if occluder_coord[1] > src_center[1]:
                    rotation = rotation + 180
                occluder_img = imutils.rotate_bound(occluder_img, rotation)
                occluder_mask = imutils.rotate_bound(occluder_mask, rotation)

            # overlay occluder to src images

            try:
                occlusion_mask = np.zeros(src_mask.shape, np.uint8)
                occlusion_mask[
                    (occlusion_mask > 0) & (occlusion_mask < 255)] = 255
                # paste occluder to src image
                result_img, result_mask, occlusion_mask = paste_over(
                    occluder_img, occluder_mask, src_img, src_mask,
                    occluder_coord, occlusion_mask,
                    self.args["randomOcclusion"])

            except Exception as e:
                print(e)
                print(f'Failed: {image} , {occluder}')
                return

            # blur edges of occluder
            kernel = np.ones((5, 5), np.uint8)
            occlusion_mask_edges = cv2.dilate(occlusion_mask, kernel,
                                              iterations=2) - cv2.erode(
                occlusion_mask, kernel, iterations=2)
            ret, filtered_occlusion_mask_edges = cv2.threshold(
                occlusion_mask_edges, 240, 255, cv2.THRESH_BINARY)
            blurred_image = cv2.GaussianBlur(result_img, (5, 5), 0)
            result_img = np.where(np.dstack(
                (np.invert(filtered_occlusion_mask_edges == 255),) * 3),
                                  result_img, blurred_image)

            # augment occluded image
            transformed = self.image_augmentor(image=result_img,
                                               mask=result_mask,
                                               mask1=occlusion_mask)
            result_img, result_mask, occlusion_mask = transformed["image"], \
                                                      transformed["mask"], \
                                                      transformed["mask1"]
            # plt.imsave("./test2.jpg", result_mask)
            # exit()
            result_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)

            # save images
            self.save_images(image.split(".")[0], result_img, result_mask,
                             occlusion_mask)
        except Exception as e:
            print(e)
            print(image)

    def save_images(self, img_name, image, mask, occlusion_mask):

        cv2.imwrite(self.args["outputImgDir"] + f"{img_name}.jpg", image)
        cv2.imwrite(self.args["outputMaskDir"] + f"{img_name}.jpg", mask)
        # if self.args["maskForOcclusion"]:
        # cv2.imwrite(self.args["occlusionMaskDir"]+f"{img_name}.png",occlusion_mask)

    def colour_transfer(self, src_img, src_mask, occluder_img, src_rect):
        ##change the colour of the occluder 
        # crop the src image
        temp_src = cv2.bitwise_or(src_img, src_img, mask=src_mask)
        cropped_src = temp_src[src_rect[1]:(src_rect[1] + src_rect[3]),
                      src_rect[0]:(src_rect[0] + src_rect[2])]
        # crop the mask
        cropped_src_mask = src_mask[src_rect[1]:(src_rect[1] + src_rect[3]),
                           src_rect[0]:(src_rect[0] + src_rect[2])]
        cropped_src = cv2.resize(cropped_src, (
        occluder_img.shape[1], occluder_img.shape[0]),
                                 interpolation=cv2.INTER_LANCZOS4)
        # resize to the size of src image
        cropped_src_mask = cv2.resize(cropped_src_mask, (
        occluder_img.shape[1], occluder_img.shape[0]),
                                      interpolation=cv2.INTER_LANCZOS4)

        ##solve black imbalance
        # get the mean and std in each channel
        r = np.mean(cropped_src[:, :, 0][cropped_src[:, :, 0] != 0])
        g = np.mean(cropped_src[:, :, 1][cropped_src[:, :, 1] != 0])
        b = np.mean(cropped_src[:, :, 2][cropped_src[:, :, 2] != 0])
        r_std = np.std(cropped_src[:, :, 0][cropped_src[:, :, 0] != 0])
        g_std = np.std(cropped_src[:, :, 1][cropped_src[:, :, 1] != 0])
        b_std = np.std(cropped_src[:, :, 2][cropped_src[:, :, 2] != 0])

        # calculate the black ratio. src/occluder  
        # current lower threshold is set to half the mean in each channel
        black_ratio = np.round((np.sum(
            cropped_src < (r / 2, g / 2, b / 2)) / np.sum(
            occluder_img == (0, 0, 0))) - 1, 2)

        if black_ratio > 1:
            black_ratio = 1

        if (black_ratio) > 0:
            cropped_src_mask[cropped_src_mask == 0] = np.random.binomial(n=1,
                                                                         p=1 - black_ratio,
                                                                         size=[
                                                                             cropped_src_mask[
                                                                                 cropped_src_mask == 0].size])
            cropped_src[:, :, :3][np.invert(cropped_src_mask.astype(bool))] = [
                r, g, b]
        # handle pixels that is too bright
        # current upper threshold set to mean + 1 std
        r2, g2, b2 = r + r_std, g + g_std, b + b_std
        red, green, blue = cropped_src[:, :, 0], cropped_src[:, :,
                                                 1], cropped_src[:, :, 2]
        mask = (red > r2) | (green > g2) | (blue > b2)
        cropped_src[:, :, :3][mask] = [min(255, r + r_std),
                                       min(255, g + g_std),
                                       min(255, b + b_std)]

        occluder_img = color_transfer_sot(occluder_img / 255,
                                          cropped_src / 255)
        # occluder_img = (np.clip( occluder_img, 0.0, 1.0)*255).astype("uint8")
        occluder_img = (torch.clamp(occluder_img, 0.0, 1.0) * 255).astype(
            "uint8")
        return occluder_img
