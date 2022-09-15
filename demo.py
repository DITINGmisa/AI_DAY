import argparse

import cv2
import numpy as np
import torch

from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_state
from modules.pose import Pose, track_poses, get_similarity
from val import normalize, pad_width


class ImageReader(object):
    def __init__(self, file_names):
        self.file_names = file_names
        self.max_idx = len(file_names)

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx == self.max_idx:
            raise StopIteration
        img = cv2.imread(self.file_names[self.idx], cv2.IMREAD_COLOR)
        if img.size == 0:
            raise IOError('Image {} cannot be read'.format(self.file_names[self.idx]))
        self.idx = self.idx + 1
        return img


class VideoReader(object):
    def __init__(self, file_name):
        self.file_name = file_name
        try:  # OpenCV needs int to read from webcam
            self.file_name = int(file_name)
        except ValueError:
            pass

    def __iter__(self):
        self.cap = cv2.VideoCapture(self.file_name)
        if not self.cap.isOpened():
            raise IOError('Video {} cannot be opened'.format(self.file_name))
        return self

    def __next__(self):
        was_read, img = self.cap.read()
        if not was_read:
            raise StopIteration
        dst = cv2.flip(img, 1)  # 镜像翻转
        return dst


def infer_fast(net, img, net_input_height_size, stride, upsample_ratio, cpu,
               pad_value=(0, 0, 0), img_mean=np.array([128, 128, 128], np.float32), img_scale=np.float32(1 / 256)):
    height, width, _ = img.shape
    scale = net_input_height_size / height

    scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    scaled_img = normalize(scaled_img, img_mean, img_scale)
    min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
    padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

    tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
    if not cpu:
        tensor_img = tensor_img.cuda()

    stages_output = net(tensor_img)

    stage2_heatmaps = stages_output[-2]
    heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
    heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    stage2_pafs = stages_output[-1]
    pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
    pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    return heatmaps, pafs, scale, pad


def run_demo(net, image_provider, height_size, cpu, track, smooth, demoimage, _mode):
    net = net.eval()
    if not cpu:
        net = net.cuda()

    stride = 8
    upsample_ratio = 4
    num_keypoints = Pose.num_kpts
    previous_poses = []
    delay = 1
    for img in image_provider:
        orig_img = img.copy()
        heatmaps, pafs, scale, pad = infer_fast(net, img, height_size, stride, upsample_ratio, cpu)

        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(num_keypoints):  # 19th for bg
            total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type,
                                                     total_keypoints_num)

        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
            all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale
        current_poses = []
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
            pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
            for kpt_id in range(num_keypoints):
                if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                    pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
            pose = Pose(pose_keypoints, pose_entries[n][18])
            if current_poses == [] or pose.bbox[2] * pose.bbox[3] > current_poses[-1].bbox[2] * current_poses[-1].bbox[3]:
                current_poses.append(pose)

        # if track:
        #     track_poses(previous_poses, current_poses, smooth=smooth)
        #     previous_poses = current_poses
        if not current_poses:
            print("[ERROR]:没有检测到人像！")
            continue
        pose = current_poses[-1]
        if not demoimage:
            a, b = get_similarity(poseList[_mode], pose)  # pose1:previous_poses[-1]
        pose.draw(img)  # 获取相似度后才知道要不要画对号
        cv2.rectangle(img, (pose.bbox[0], int(pose.bbox[1] - 0.5 / 9 * pose.bbox[3])),  # 假设九头身，向上延伸半张脸
                      (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), (0, 255, 0))
        # img = cv2.addWeighted(orig_img, 0.4, img, 0.4, 0)
        if demoimage:
            cv2.imwrite(("./result"+str(_mode)+".jpg"),img)
            return pose, img
        # if track:
        #     cv2.putText(img, 'id: {}'.format(pose.id), (pose.bbox[0], pose.bbox[1] - 16),
        #                 cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
        resized_image = cv2.resize(img, dsize=(1920, 1080))
        resized_image_t = cv2.resize(imgList[_mode], dsize=(480, 360))
        resized_image[0:360, 1440:1920] = resized_image_t
        TruePoint = 0
        for kpt_id in range(len(poseList[_mode].keypoints)):
            if poseList[_mode].keypoints[kpt_id, 0] != -1:
                TruePoint = TruePoint + 1
        cv2.putText(resized_image, 'similarity:{:.2f}%, {}/{}'.format(a, b, TruePoint), (0, 45), cv2.FONT_HERSHEY_DUPLEX, 2,
                    (0, 0, 255), 2)
        if b == TruePoint or a >= 80:
            cv2.putText(resized_image, 'Succeeded!', (800, 580), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 2)
            delay = 0
        cv2.namedWindow('Lightweight Human Pose Estimation Python Demo', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('Lightweight Human Pose Estimation Python Demo', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('Lightweight Human Pose Estimation Python Demo', resized_image)
        key = cv2.waitKey(delay)
        if key == 27:  # esc
            return
        elif key == 32:  # space
            if delay == 1:
                delay = 0
            else:
                delay = 1
        elif key == 13:  # enter
            _mode = (_mode + 1) % 3


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Lightweight human pose estimation python demo.
                       This is just for quick results preview.
                       Please, consider c++ demo for the best performance.''')
    parser.add_argument('--checkpoint-path', type=str, default='checkpoint_iter_370000.pth',
                        help='path to the checkpoint')  # required=True,
    parser.add_argument('--height-size', type=int, default=256, help='network input layer height size')
    parser.add_argument('--video', type=str, default='0', help='path to video file or camera id')
    parser.add_argument('--images', nargs='+', default=[''], help='path to input image(s)')
    parser.add_argument('--cpu', action='store_true', help='run network inference on cpu')
    parser.add_argument('--track', type=int, default=1, help='track pose id in video')
    parser.add_argument('--smooth', type=int, default=1, help='smooth pose keypoints')
    args = parser.parse_args()

    if args.video == '' and args.images == ['']:
        raise ValueError('Either --video or --image has to be provided')

    net = PoseEstimationWithMobileNet()
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    load_state(net, checkpoint)
    frame_provider = ImageReader(args.images)
    if args.video != '':
        frame_provider = VideoReader(args.video)
    else:
        args.track = 0
    mode = 0
    poseList = []
    imgList = []
    for i in range(3):
        pose_t, img_t = run_demo(net, ImageReader([str(i) + '.jpg']), args.height_size, args.cpu, args.track, args.smooth, True, i)
        poseList.append(pose_t)
        imgList.append(img_t)
    run_demo(net, frame_provider, args.height_size, args.cpu, args.track, args.smooth, False, mode)
