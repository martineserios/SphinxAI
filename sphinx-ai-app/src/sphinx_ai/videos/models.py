from abc import ABC, abstractmethod

import cv2

from sphinx_ai.utilities import VideoProgressBar
from sphinx_ai.utils.logging_config import logger
from sphinx_ai.videos.io import VideoCapture, VideoWriterFromCapture


class VideoModelAppInterface(ABC):
    def __init__(
        self,
        model,
        video_capture: VideoCapture,
        video_writer: VideoWriterFromCapture,
        progress_bar: VideoProgressBar,
    ):
        self.video_capture = video_capture
        self.video_writer = video_writer
        self.model = model
        self.progress_bar = progress_bar

    @abstractmethod
    def pre_process_frame(self, frame):
        return frame

    @abstractmethod
    def instanciate_model(self):
        pass

    @abstractmethod
    def detect(self, frame):
        return frame

    @abstractmethod
    def write_results_on_frame(self, frame):
        return frame

    @abstractmethod
    def post_process_frame(self, frame):
        return frame

    @abstractmethod
    def process_video(self, save_processed_video=True):
        while self.video_capture.cap.isOpened():
            ret, frame = self.video_capture.cap.read()
            if not ret:
                break

            pre_proc_frame = self.pre_process_frame(frame)
            self.detect(pre_proc_frame)
            proc_video = self.write_results_on_frame(frame)
            post_proc_frame = self.post_process_frame(proc_video)

            if save_processed_video:
                self.video_writer.writer.write(post_proc_frame)

            self.progress_bar.update()

            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

        self.video_capture.cap.release()
