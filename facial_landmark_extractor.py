import cv2
import numpy as np
import warnings
import face_alignment
from numpy.lib.arraysetops import isin
import torch
import PIL.Image
import scipy.ndimage
import os
from tqdm import tqdm

from torch._C import Value

class FacialLandmarksExtractor:
    def __init__(self, device='cuda', landmark_weights=None):
        self.fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType._2D, flip_input=False, device=device)
        
        self.resolution = 256

        self.landmarks_dict = {
            'jaw': (0, 16),
            'left_eyebrow': (17, 21),
            'right_eyebrow': (22, 26),
            'nose_bridge': (27, 30),
            'lower_nose': (30, 35),
            'left_eye': (36, 41),
            'right_eye': (42, 47),
            'outer_lip': (48, 59),
            'inner_lip': (60, 67)
        }
        
        self.center_index = 27

        if landmark_weights is None:
            landmark_weights = [ 
                0.05, # jaw
                1.0, # left_eyebrow
                1.0, # right_eyebrow
                0.1, # nose_bridge
                1.0, # lower_nose
                1.0, # left_eye
                1.0, # right_eye
                1.0, # outer_lip
                1.0, # inner_lip
            ]
        self.landmark_weights = np.zeros(68)

        for i, bounds in enumerate(self.landmarks_dict.values()):
            upper, lower = bounds
            self.landmark_weights[upper:lower+1] = landmark_weights[i]

        self.landmark_weights = torch.Tensor(self.landmark_weights).to(device)

    def safely_read(self, obj):
        if isinstance(obj, str):
            img = cv2.imread(obj)
            return img
        else:
            return obj

    def read_and_extract(self, path):
        img = self.safely_read(path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img, self.extract(img_rgb)

    def extract(self, img):
        landmarks = self.fa.get_landmarks(img)
        if len(landmarks) > 1:
            warnings.warn("Multiple faces detecting, choosing first one.")

        return landmarks[0]

    def get_heat_map(self, img):
        if type(img) != torch.Tensor:
            img = torch.tensor(img, dtype=torch.float32)
        
        if len(img.shape) < 4:
            img = img.unsqueeze(0)

        if img.shape[-1] == 3:
            img = img.permute((0, 3, 1, 2))

        elif img.shape[-1] == 1:
            raise ValueError("Needs RGB image.")

        img = torch.nn.functional.interpolate(img, self.resolution)
        img = img - img.min()
        img = img / img.max()

        out = self.fa.face_alignment_net(img)
        return out

    def _drawPoints(self, img, landmarks_np, point_range, col=(255, 200, 0), closed=False):
        points = []
        start_point, end_point = point_range
        for i in range(start_point, end_point+1):
            point = [landmarks_np[i, 0], landmarks_np[i, 1]]
            points.append(point)

        points = np.array(points, dtype=np.int32)
        cv2.polylines(img, [points], closed, col,
                      thickness=2, lineType=cv2.LINE_8)

    def _draw_landmarks_on_img(self, img, landmarks, col=(255, 200, 0)):
        assert(len(landmarks) == 68)

        img_copy = img.copy()
        self._drawPoints(img_copy, landmarks,
                         self.landmarks_dict['jaw'], col)
        self._drawPoints(img_copy, landmarks,
                         self.landmarks_dict['left_eyebrow'], col)
        self._drawPoints(img_copy, landmarks,
                         self.landmarks_dict['right_eyebrow'], col)
        self._drawPoints(img_copy, landmarks,
                         self.landmarks_dict['nose_bridge'], col)
        self._drawPoints(img_copy, landmarks,
                         self.landmarks_dict['lower_nose'], col, True)
        self._drawPoints(img_copy, landmarks,
                         self.landmarks_dict['left_eye'], col, True)
        self._drawPoints(img_copy, landmarks,
                         self.landmarks_dict['right_eye'], col, True)
        self._drawPoints(img_copy, landmarks,
                         self.landmarks_dict['outer_lip'], col, True)
        self._drawPoints(img_copy, landmarks,
                         self.landmarks_dict['inner_lip'], col, True)

        return img_copy

    def display_landmarks_img(self, img, landmarks):
        landmarks_img = self._draw_landmarks_on_img(img, landmarks)
        cv2.imshow("Landmark image", landmarks_img)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def get_cropped_img(self, path_or_img, output_size=512, transform_size=4096, enable_padding=True, processing_size=256):
        """ Face alignment crop from https://gist.github.com/lzhbrian/bde87ab23b499dd02ba4f588258f57d5
        """
        cv_img = self.safely_read(path_or_img) # face_alignment has a problem with png images

        max_dim_size = np.max(cv_img.shape)
        scale_factor = processing_size / max_dim_size

        width = int(cv_img.shape[1] * scale_factor)
        height = int(cv_img.shape[0] * scale_factor)
        
        # resize image
        cv_img = cv2.resize(cv_img, (width, height), interpolation = cv2.INTER_AREA)

        lms = self.fa.get_landmarks_from_image(cv_img)
        if len(lms) == 0:
            warnings.warn("No face detected.")

        if len(lms) > 1:
            warnings.warn("Multiple faces detected, choosing first one.")

        lm = lms[0]

        lm_eye_left      = lm[36 : 42]  # left-clockwise
        lm_eye_right     = lm[42 : 48]  # left-clockwise
        lm_mouth_outer   = lm[48 : 60]  # left-clockwise

        # Calculate auxiliary vectors.
        eye_left     = np.mean(lm_eye_left, axis=0)
        eye_right    = np.mean(lm_eye_right, axis=0)
        eye_avg      = (eye_left + eye_right) * 0.5
        eye_to_eye   = eye_right - eye_left
        mouth_left   = lm_mouth_outer[0]
        mouth_right  = lm_mouth_outer[6]
        mouth_avg    = (mouth_left + mouth_right) * 0.5
        eye_to_mouth = mouth_avg - eye_avg

        # Choose oriented crop rectangle.
        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
        y = np.flipud(x) * [-1, 1]
        c = eye_avg + eye_to_mouth * 0.1
        quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
        qsize = np.hypot(*x) * 2

        # convert pil (code from the internet works with pil. TODO: refactor)
        # img = PIL.Image.open(path_or_img)  
        img = self._cv_to_pil_img(cv_img)      

        # Shrink.
        shrink = int(np.floor(qsize / output_size * 0.5))
        if shrink > 1:
            rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
            img = img.resize(rsize, PIL.Image.ANTIALIAS)
            quad /= shrink
            qsize /= shrink

        # Crop.
        border = max(int(np.rint(qsize * 0.1)), 3)
        crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
        crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
        if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
            img = img.crop(crop)
            quad -= crop[0:2]

        # Pad.
        pad = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
        pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
        if enable_padding and max(pad) > border - 4:
            pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
            img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
            h, w, _ = img.shape
            y, x, _ = np.ogrid[:h, :w, :1]
            mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w-1-x) / pad[2]), 1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h-1-y) / pad[3]))
            blur = qsize * 0.02
            img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
            img += (np.median(img, axis=(0,1)) - img) * np.clip(mask, 0.0, 1.0)
            img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
            quad += pad[:2]

        # Transform.
        img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
        if output_size < transform_size:
            img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)

        return img

     
    def save_cropped_img(self, img_or_path, res=512, save_path="cropped1.png"):
        pil_img = self.get_cropped_img(img_or_path, res)
        
        # cv2.imwrite(save_path, img)
        pil_img.save(save_path)

    def save_landmarks_img(self, img, landmarks, save_path="output.png"):
        landmarks_img = self._draw_landmarks_on_img(img, landmarks)

        cv2.imwrite(save_path, landmarks_img)

    def _pil_to_cv_img(self, img):
        img = np.array(img) 
        img = img[:, :, ::-1].copy() 
        return img

    def _cv_to_pil_img(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im_pil = PIL.Image.fromarray(img)
        return im_pil

    def crop_folder(self, video_path, res, outdir):

        os.makedirs(outdir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        i = 0
        print("Cropping images")
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == False:
                break

            cropped_frame = self.get_cropped_img(frame, res)
            cropped_frame = self._pil_to_cv_img(cropped_frame)

            file_name = os.path.join(outdir, f'{i}.png')
            cv2.imwrite(file_name, cropped_frame)
            i += 1
            print("  {0} images cropped.".format(i))
        
        cap.release()
        cv2.destroyAllWindows()

    # def _drop_features(self, arr, drop_features):
    #     remaining_features = list(self.landmarks_dict.keys())
    #     remaining_points_amount = 0

    #     for drop_feature in drop_features:
    #         remaining_features.remove(drop_feature)

    #     for remaining_feature in remaining_features:
    #         start, end = self.landmarks_dict[remaining_feature]
    #         remaining_points_amount += end - start + 1

    #     new_arr = np.zeros((remaining_points_amount, 2))
    #     count = 0
    #     for remaining_feature in remaining_features:
    #         start, end = self.landmarks_dict[remaining_feature]
    #         diff = end - start + 1
    #         new_arr[count: count + diff, :] = arr[start:end+1, :]
    #         count += diff

    #     return arr

    # def _project_landmarks(self, landmarks1, landmarks2, partial_features=True):
    #     landmarks1_np = landmarks1.reshape(-1, 1, 2)
    #     landmarks2_np = landmarks2.reshape(-1, 1, 2)
    #     mask = np.ones((68), dtype=bool)
    #     if partial_features:
    #         mask = np.zeros((68), dtype=bool)
    #         mask[self.landmarks_dict['jaw'][0]
    #             :self.landmarks_dict['jaw'][1]] = 1
    #         mask[self.landmarks_dict['nose_bridge'][0]
    #             :self.landmarks_dict['nose_bridge'][1]] = 1
    #         # mask[self.landmarks_dict['lower_nose'][0]:self.landmarks_dict['lower_nose'][1]] = 1

    #     H, mask = cv2.findHomography(
    #         landmarks2_np[mask], landmarks1_np[mask], 0)

    #     return cv2.perspectiveTransform(landmarks2_np, H).squeeze()
    # def display_langmark_projection(self, img1, landmarks1, landmarks2):
    #     landmarks2_np_projected = FLE._project_landmarks(
    #         landmarks1, landmarks2)
    #     temp_img = self._draw_landmarks_on_img(
    #         img1, landmarks2_np_projected, col=(0, 255, 200))
    #     self.display_landmarks_img(temp_img, landmarks1)

    # def landmarks_distance(self, landmarks1, landmarks2, drop_features=[], no_calc=False):
    #     for drop_feature in drop_features:
    #         if drop_feature not in self.landmarks_dict.keys():
    #             raise ValueError("Can not drop feature {0}. Feature that can be dropped are {1}."
    #                              .format(drop_feature, self.landmarks_dict.keys()))

    #     if self.align:  # project landmarks2 onto landmarks 1
    #         landmarks2 = self._project_landmarks(landmarks1, landmarks2)
    #     else:  # just center
    #         # landmarks1 -= landmarks1[self.center]
    #         landmarks2 -= landmarks2[self.center]
    #         landmarks2 += landmarks1[self.center]

    #     if len(drop_features) > 0:
    #         landmarks1 = self._drop_features(landmarks1, drop_features)
    #         landmarks2 = self._drop_features(landmarks2, drop_features)

    #     if no_calc:
    #         return landmarks1, landmarks2

    #     return np.sum(np.sqrt((landmarks1 - landmarks2)**2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--func", help="crop_folder|crop_img")
    parser.add_argument("--video_path", help="display a square of a given number")
    parser.add_argument("--outdir", help="display a square of a given number")
    parser.add_argument("--res", type=int, default=512, help="display a square of a given number")
    parser.add_argument("--device", default='cuda', help='cpu|cuda')

    args = parser.parse_args()

    if args.func == 'crop_folder':
        FLE = FacialLandmarksExtractor(device=args.device)
        FLE.crop_folder(args.video_path, args.res, args.outdir)
    elif args.func == 'crop_img':
        raise NotImplementedError('Crop image functionality not implemented yet.')
    else:
        raise ValueError('func has to be crop_folder or crop_img')
