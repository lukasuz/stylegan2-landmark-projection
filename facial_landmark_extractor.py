import cv2
import numpy as np
import warnings
import face_alignment
import torch
# from torchvision.transforms import Resize

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

    def read_and_extract(self, path):
        img = cv2.imread(path)
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
        img = 255 * img

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

    def save_landmarks_img(self, img, landmarks, save_path="output.png"):
        landmarks_img = self._draw_landmarks_on_img(img, landmarks)

        cv2.imwrite(save_path, landmarks_img)

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
    path1 = "./1.png"
    path2 = "2.jpg"

    FLE = FacialLandmarksExtractor(device='cpu')
    img1, landmarks1 = FLE.read_and_extract(path1)
    img2, landmarks2 = FLE.read_and_extract(path2)
    batch_test = np.concatenate([np.expand_dims(img1, 0), np.expand_dims(img1, 0)], axis=0)
    heatmaps = FLE.get_heat_map(batch_test)
    heatmap1 = heatmaps[0].detach().cpu().numpy()
    heatmap1 = np.sum(heatmap1, axis=0, keepdims=True)
    heatmap1 = np.repeat(heatmap1, 3, axis=0)
    heatmap1 = heatmap1.transpose((1,2,0))
    # heatmap1 = np.resize(heatmap1, img1.shape[:2])
    heatmap1 = cv2.resize(heatmap1, img1.shape[:2], interpolation=cv2.INTER_CUBIC)
    # heat
    heatmap1 -= heatmap1.min()
    heatmap1 /= heatmap1.max()
    heatmap1 *= 255
    heatmap1 = heatmap1.astype('uint8')

    FLE.display_landmarks_img(img1, landmarks1)
    FLE.display_landmarks_img(heatmap1, landmarks1)