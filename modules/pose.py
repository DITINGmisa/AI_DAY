import cv2
import numpy as np

from modules.keypoints import BODY_PARTS_KPT_IDS, BODY_PARTS_PAF_IDS
from modules.one_euro_filter import OneEuroFilter

def draw_Bingo(x, y, image):
    cv2.line(image, (int(x - 5), int(y - 5)), (int(x - 10), int(y - 10)), (0, 255, 0), 1)
    cv2.line(image, (int(x - 5), int(y - 5)), (int(x + 2), int(y - 20)), (0, 255, 0), 1)


class Pose:
    num_kpts = 18
    kpt_names = ['nose', 'neck',
                 'r_sho', 'r_elb', 'r_wri', 'l_sho', 'l_elb', 'l_wri',
                 'r_hip', 'r_knee', 'r_ank', 'l_hip', 'l_knee', 'l_ank',
                 'r_eye', 'l_eye',
                 'r_ear', 'l_ear']
    sigmas = np.array([.26, .79, .79, .72, .62, .79, .72, .62, 1.07, .87, .89, 1.07, .87, .89, .25, .25, .35, .35],
                      dtype=np.float32) / 10.0
    vars = (sigmas * 2) ** 2
    last_id = -1
    colors = [[255, 0, 255],  [0, 0, 255],   [18, 153, 255], [0, 255, 255], [42, 42, 128],  [201, 252, 18],
              [50, 205, 50],  [32, 139, 34], [87, 201, 0],   [0, 255, 12],  [208, 224, 64], [250, 46, 10],
              [235, 206, 13], [205, 90, 10], [225, 105, 65], [255, 0, 255], [200, 0, 255],  [255, 0, 200]]

    def __init__(self, keypoints, confidence):
        super().__init__()
        self.keypoints = keypoints
        self.confidence = confidence
        self.bbox = Pose.get_bbox(self.keypoints)
        self.id = None
        self.filters = [[OneEuroFilter(), OneEuroFilter()] for _ in range(Pose.num_kpts)]
        self.similar_array = []

    @staticmethod
    def get_bbox(keypoints):
        found_keypoints = np.zeros((np.count_nonzero(keypoints[:, 0] != -1), 2), dtype=np.int32)
        found_kpt_id = 0
        for kpt_id in range(Pose.num_kpts):
            if keypoints[kpt_id, 0] == -1:
                continue
            found_keypoints[found_kpt_id] = keypoints[kpt_id]
            found_kpt_id += 1
        bbox = cv2.boundingRect(found_keypoints)  # 返回四个值，分别是x，y，w，h；x，y是矩阵左上点的坐标，w，h是矩阵的宽和高
        return bbox

    def update_id(self, id=None):
        self.id = id
        if self.id is None:
            self.id = Pose.last_id + 1
            Pose.last_id += 1

    def draw(self, img):
        assert self.keypoints.shape == (Pose.num_kpts, 2)
        for id in range(len(self.keypoints)):
            if (not self.similar_array == []) and self.similar_array[id] == 1:
                draw_Bingo(int(self.keypoints[id, 0]), int(self.keypoints[id, 1]), img)
        for part_id in range(len(BODY_PARTS_PAF_IDS) - 2):
            kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]
            global_kpt_a_id = self.keypoints[kpt_a_id, 0]
            if global_kpt_a_id != -1:
                x_a, y_a = self.keypoints[kpt_a_id]
                cv2.circle(img, (int(x_a), int(y_a)), 3, Pose.colors[part_id], -1)
            kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]
            global_kpt_b_id = self.keypoints[kpt_b_id, 0]
            if global_kpt_b_id != -1:
                x_b, y_b = self.keypoints[kpt_b_id]
                cv2.circle(img, (int(x_b), int(y_b)), 3, Pose.colors[part_id], -1)
                # if (not self.similar_array == []) and self.similar_array[part_id] == 1:
                #     draw_Bingo(int(x_b), int(y_b), img)
            if global_kpt_a_id != -1 and global_kpt_b_id != -1:
                cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)), Pose.colors[part_id], 2)


def get_similarity(a, b, threshold=0.75):
    num_similar_kpt = 0
    mean_similarity = 0
    for kpt_id in range(Pose.num_kpts):
        if a.keypoints[kpt_id, 0] != -1 and b.keypoints[kpt_id, 0] != -1:
            # distance = np.sum((a.keypoints[kpt_id] - b.keypoints[kpt_id]) ** 2)
            # area = max(a.bbox[2] * a.bbox[3], b.bbox[2] * b.bbox[3])
            # similarity = np.exp(-distance / (2 * (area + np.spacing(1)) * Pose.vars[kpt_id]))  # np.spacing(1):产生一个无穷小量,防止除0
            area = max(a.bbox[2] * a.bbox[3], b.bbox[2] * b.bbox[3])
            distance = (((a.keypoints[kpt_id, 0] - a.bbox[0])/a.bbox[2] - (b.keypoints[kpt_id, 0] - b.bbox[0])/b.bbox[2]) ** 2 +
                        ((a.keypoints[kpt_id, 1] - a.bbox[1])/a.bbox[3] - (b.keypoints[kpt_id, 1] - b.bbox[1])/b.bbox[3]) ** 2) * area  # 归一化距离
            similarity = np.exp(-distance / (2 * (area + np.spacing(1)) * Pose.vars[kpt_id]))  # np.spacing(1):产生一个无穷小量,防止除0
            print(a.keypoints[kpt_id], b.keypoints[kpt_id], distance, area, similarity)
            if similarity > threshold:
                num_similar_kpt += 1
                b.similar_array.append(1)  # 相似点队列存到b的similar_array中
            else:
                b.similar_array.append(0)
            mean_similarity = (mean_similarity * kpt_id + similarity)/(kpt_id + 1)
        else:
            b.similar_array.append(0)
    return 100 * mean_similarity, num_similar_kpt


def track_poses(previous_poses, current_poses, threshold=3, smooth=False):
    """Propagate poses ids from previous frame results. Id is propagated,
    if there are at least `threshold` similar keypoints between pose from previous frame and current.
    If correspondence between pose on previous and current frame was established, pose keypoints are smoothed.

    :param previous_poses: poses from previous frame with ids
    :param current_poses: poses from current frame to assign ids
    :param threshold: minimal number of similar keypoints between poses
    :param smooth: smooth pose keypoints between frames
    :return: None
    """
    current_poses = sorted(current_poses, key=lambda pose: pose.confidence, reverse=True)  # match confident poses first
    mask = np.ones(len(previous_poses), dtype=np.int32)
    for current_pose in current_poses:
        best_matched_id = None
        best_matched_pose_id = None
        best_matched_iou = 0
        for id, previous_pose in enumerate(previous_poses):
            if not mask[id]:
                continue
            _, iou = get_similarity(current_pose, previous_pose)
            if iou > best_matched_iou:
                best_matched_iou = iou
                best_matched_pose_id = previous_pose.id
                best_matched_id = id
        if best_matched_iou >= threshold:
            mask[best_matched_id] = 0
        else:  # pose not similar to any previous
            best_matched_pose_id = None
        current_pose.update_id(best_matched_pose_id)

        if smooth:
            for kpt_id in range(Pose.num_kpts):
                if current_pose.keypoints[kpt_id, 0] == -1:
                    continue
                # reuse filter if previous pose has valid filter
                if (best_matched_pose_id is not None
                        and previous_poses[best_matched_id].keypoints[kpt_id, 0] != -1):
                    current_pose.filters[kpt_id] = previous_poses[best_matched_id].filters[kpt_id]
                current_pose.keypoints[kpt_id, 0] = current_pose.filters[kpt_id][0](current_pose.keypoints[kpt_id, 0])
                current_pose.keypoints[kpt_id, 1] = current_pose.filters[kpt_id][1](current_pose.keypoints[kpt_id, 1])
            current_pose.bbox = Pose.get_bbox(current_pose.keypoints)
    return best_matched_iou