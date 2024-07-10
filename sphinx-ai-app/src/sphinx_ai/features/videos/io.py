# import packages
import cv2
from sphinx_ai.utils.logging_config import logger


class VideoCapture:
    def __init__(self, input_filepath: str, name="Live"):
        # Read input video
        self.input_filepath = input_filepath
        self.cap = cv2.VideoCapture(
            input_filepath,
        )

        # Get video properties
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.spf = 1 / self.fps
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # while self.cap.isOpened():
        ret, self.frame = self.cap.read()
        self.frame_shape = self.frame.shape
        self.channels = self.frame.shape[2]
        # self.channels = 3
        # self.cap.release()


class VideoWriterFromCapture:
    def __init__(self, output_filepath: str, video_capture: VideoCapture):
        self.output_filepath = output_filepath
        self.writer = cv2.VideoWriter(
            filename=output_filepath,
            fourcc=video_capture.fourcc,
            fps=video_capture.fps,
            frameSize=(video_capture.width, video_capture.height),
        )
