"""Data loader for unlabeled upload-style videos or frame folders."""

import glob
import os

import cv2
import numpy as np

from dataset.data_loader.BaseLoader import BaseLoader


class UploadLoader(BaseLoader):
    """Loads a single uploaded video or frame directory without ground truth labels."""

    IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
    VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")

    def __init__(self, name, data_path, config_data, device=None):
        super().__init__(name, data_path, config_data, device)

    def get_raw_data(self, data_path):
        if os.path.isdir(data_path):
            index = os.path.basename(os.path.normpath(data_path)) or "upload"
            return [{"index": index, "path": data_path, "subject": "upload"}]

        if os.path.isfile(data_path):
            index = os.path.splitext(os.path.basename(data_path))[0] or "upload"
            return [{"index": index, "path": data_path, "subject": "upload"}]

        raise ValueError(f"{self.dataset_name} data path does not exist: {data_path}")

    def split_raw_data(self, data_dirs, begin, end):
        return data_dirs

    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        source_path = data_dirs[i]["path"]
        saved_filename = data_dirs[i]["index"]

        if os.path.isdir(source_path):
            frames = self.read_frame_directory(source_path)
        else:
            frames = self.read_video_file(source_path)

        if frames.size == 0:
            raise ValueError(f"No readable frames found at {source_path}")

        # Extraction mode has no ground-truth BVP. Save zero labels with the
        # correct temporal length so the existing preprocessing pipeline can run.
        bvps = np.zeros(frames.shape[0], dtype=np.float32)

        frames_clips, bvps_clips = self.preprocess(frames, bvps, config_preprocess)
        input_name_list, label_name_list = self.save_multi_process(
            frames_clips, bvps_clips, saved_filename
        )
        file_list_dict[i] = input_name_list

    @classmethod
    def read_frame_directory(cls, directory_path):
        frame_paths = sorted(
            frame_path
            for frame_path in glob.glob(os.path.join(directory_path, "*"))
            if os.path.isfile(frame_path)
            and frame_path.lower().endswith(cls.IMAGE_EXTENSIONS)
        )

        frames = []
        for frame_path in frame_paths:
            frame = cv2.imread(frame_path)
            if frame is None:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        return np.asarray(frames)

    @classmethod
    def read_video_file(cls, video_path):
        if not video_path.lower().endswith(cls.VIDEO_EXTENSIONS):
            raise ValueError(f"Unsupported upload video format: {video_path}")

        capture = cv2.VideoCapture(video_path)
        frames = []
        success, frame = capture.read()
        while success:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            success, frame = capture.read()
        capture.release()
        return np.asarray(frames)
