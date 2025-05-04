from collections import deque

import cv2
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

from sphinx_ai.utilities import VideoProgressBar
from sphinx_ai.utils.logging_config import logger
from sphinx_ai.videos.io import VideoCapture, VideoWriterFromCapture
from sphinx_ai.videos.models import VideoModelAppInterface


def get_optimal_font_scale(text, width, font, thickness=1):
    for scale in reversed(range(0, 60, 1)):
        textSize = cv2.getTextSize(
            text, fontFace=font, fontScale=scale / 10, thickness=thickness
        )
        new_width = textSize[0][0]
        if new_width <= width:
            return scale / 10
    return 1


def optimize_text_size(cv2_func_put_text_func):
    def decor(*args, **kwargs):
        scale = get_optimal_font_scale(
            kwargs["text"], kwargs["width_of_screen"], kwargs["font"]
        )
        kwargs["size"] = scale
        return cv2_func_put_text_func(*args, **kwargs)

    return decor


class EyeAnalyzer(VideoModelAppInterface):
    def __init__(
        self,
        video_capture,
        video_writer,
        progress_bar,
        models_params,
        st_output_streaming,
    ):
        self.num_faces = models_params["Blink Detector"]["num_faces"]
        self.blink_detection_threshold = models_params["Blink Detector"][
            "blink_detection_threshold"
        ]
        self.instanciate_model()
        super().__init__(self.model, video_capture, video_writer, progress_bar)
        self.capture = video_capture
        self.writer = video_writer
        self.st_output_streaming = st_output_streaming

        # # Find OpenCV version
        # (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split(".")
        self.detection_result = None
        self.total_blinks = -1
        self.prev_frame_blink = False
        self.blink = False
        self.counterTime = 0
        self.blink_duration = 0
        self.frames_in_ten_secs = self.capture.fps * 10
        self.blinks_acc = deque(maxlen=self.frames_in_ten_secs)
        self.ten_secs_frames = 100  # self.capture.fps * 10
        # variables
        self.frame_counter = 0
        self.CEF_COUNTER = 0
        self.TOTAL_BLINKS = 0
        # constants
        self.CLOSED_EYES_FRAME = 3
        self.FONTS = cv2.FONT_HERSHEY_COMPLEX

        self.frame_counter = 0
        self.prev_tick = cv2.getTickCount()
        logger.info("EyeAnalyzer Model initialized")

    color = (0, 255, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX

    @optimize_text_size
    def write_text_to_frame(
        self,
        frame,
        text,
        position,
        width_of_screen,
        font=font,
        size=1,
        color=color,
        thickness=1,
    ):
        cv2.putText(
            frame,
            text,
            position,
            font,
            size,
            color,
            thickness,
        )

    def instanciate_model(self):
        self.BaseOptions = mp.tasks.BaseOptions
        self.FaceLandmarker = mp.tasks.vision.FaceLandmarker
        self.FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        self.VisionRunningMode = mp.tasks.vision.RunningMode

        # Create a face landmarker instance with the video mode:
        self.options = self.FaceLandmarkerOptions(
            base_options=self.BaseOptions(
                model_asset_path="resources/models/face_landmarker.task"
            ),
            running_mode=self.VisionRunningMode.VIDEO,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=self.num_faces,
        )
        logger.info(f"Num faces: {self.num_faces}")

        self.model = self.FaceLandmarker.create_from_options(self.options)

    def detect(self, frame):
        self.detection_result = self.model.detect_for_video(
            frame, self.time_tick_manager()
        )

    def write_results_on_frame(self, frame):
        face_landmarks_list = self.detection_result.face_landmarks
        # annotated_image = np.copy(frame)

        # Loop through the detected faces to visualize.
        for idx in range(len(face_landmarks_list)):
            face_landmarks = face_landmarks_list[idx]

            # Draw the face landmarks.
            face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            face_landmarks_proto.landmark.extend(
                [
                    landmark_pb2.NormalizedLandmark(
                        x=landmark.x, y=landmark.y, z=landmark.z
                    )
                    for landmark in face_landmarks
                ]
            )

            solutions.drawing_utils.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style(),
            )
            solutions.drawing_utils.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style(),
            )
            solutions.drawing_utils.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style(),
            )

        color = (200, 255, 155)
        font = cv2.FONT_HERSHEY_SIMPLEX

        # added +1 because it starts in -1. can be improved
        self.write_text_to_frame(
            frame=frame,
            text=f"Blink counter: {self.total_blinks + 1}",
            position=(10, 40),
            width_of_screen=self.frame_width / 3,
            font=font,
            size=1,
            color=color,
            thickness=1,
        )

        self.write_text_to_frame(
            frame=frame,
            text=f"Last blink duration: {round(self.blink_duration, 3)} seg.",
            position=(10, 70),
            width_of_screen=self.frame_height / 3,
            font=font,
            size=1,
            color=color,
            thickness=1,
        )

        self.write_text_to_frame(
            frame=frame,
            text=f"Blinks last 10sec: {sum(self.blinks_acc)} blinks/10seg.",
            position=(10, 100),
            width_of_screen=self.frame_height / 3,
            font=font,
            size=1,
            color=color,
            thickness=1,
        )

        self.write_text_to_frame(
            frame=frame,
            text=self.eyes_looking_direction(),
            position=(10, 130),
            width_of_screen=self.frame_height / 3,
            font=font,
            size=1,
            color=color,
            thickness=1,
        )
        if self.blink:
            self.write_text_to_frame(
                frame=frame,
                text="Blink!",
                position=(10, 10),
                width_of_screen=self.frame_height / 3,
                font=font,
                size=1,
                color=color,
                thickness=1,
            )

        return frame

    def update_blink_counter(self):
        if (self.prev_frame_blink == False) and (self.is_blinking() == True):
            self.total_blinks += 1
            self.update_blinks_ten_secs(blink=True)
            return True
        else:
            self.update_blinks_ten_secs(blink=False)
            return False

    def update_prev_blink(self):
        if (self.prev_frame_blink == False) and (self.is_blinking() == True):
            self.prev_frame_blink = True
        elif self.is_blinking() == False:
            self.prev_frame_blink = False

    def update_frame_counter(self):
        self.frame_counter += 1

    def update_blink_duration(self):
        if self.blink_duration == -1:
            self.blink_duration = 0
        elif self.is_blinking() and (self.prev_frame_blink == True):
            self.blink_duration += self.capture.spf
        elif self.is_blinking() and (self.prev_frame_blink == False):
            self.blink_duration = 0
            self.blink_duration += self.capture.spf

        return self.blink_duration

    def update_blinks_ten_secs(self, blink: bool):
        if blink is True:
            self.blinks_acc.append(True)
        elif blink is False:
            self.blinks_acc.append(False)

        return self.blinks_acc

    def time_tick_manager(self):
        timestamps = [self.video_capture.cap.get(cv2.CAP_PROP_POS_MSEC)]
        calc_timestamps = [0.0]
        timestamps.append(self.video_capture.cap.get(cv2.CAP_PROP_POS_MSEC))
        calc_timestamps.append(calc_timestamps[-1] + 1000 / self.video_capture.fps)
        return int(timestamps[-1] - calc_timestamps[-1])

    def pre_process_frame(self, frame):
        frame = cv2.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        self.frame_height, self.frame_width = frame.shape[:2]

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        return mp_image

    def post_process_frame(self, frame):
        # To improve performance
        # frame.flags.writeable = True
        # Convert the frame from BGR to RGB format
        # frame = cv2.flip(frame, 1)  # cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def process_video(self):  # video_placeholder):
        # Process video frames
        with self.model as landmarker:
            while self.video_capture.cap.isOpened():
                ret, frame = self.video_capture.cap.read()
                if not ret:
                    break

                logger.info("Reading video")
                # Apply pre_processing to the frame
                pre_proc_frame = self.pre_process_frame(frame)
                self.detection_result = landmarker.detect_for_video(
                    pre_proc_frame, self.time_tick_manager()
                )
                self.update_blink_counter()
                self.update_blink_duration()
                self.update_prev_blink()
                proc_frame = self.write_results_on_frame(frame)
                post_proc_frame = self.post_process_frame(proc_frame)
                self.video_writer.writer.write(post_proc_frame)

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.st_output_streaming.image(frame, channels="RGB")

                self.progress_bar.update()

                if cv2.waitKey(10) & 0xFF == ord("q"):
                    break
                # else:
                #     break

        self.video_capture.cap.release()

    def is_blinking(self, blink_detection_threshold=None):
        if blink_detection_threshold is not None:
            self.blink_detection_threshold = blink_detection_threshold
        logger.info(f"Detection threshold used: {self.blink_detection_threshold}")
        try:
            if (
                self.detection_result.face_blendshapes[0][9].score
                > self.blink_detection_threshold
                or self.detection_result.face_blendshapes[0][10].score
                > self.blink_detection_threshold
            ):
                self.blink = True
                return self.blink
            else:
                self.blink = False
                return False
        except Exception as e:
            logger.exception(f"An error occurred: {e}")
            logger.info("No face has been detected")
            self.blink = False
            return False

    def eyes_looking_direction(self, score_threshold=None):
        if score_threshold is not None:
            self.score_threshold = score_threshold
        try:
            if score_threshold is not None:
                self.score_threshold = score_threshold
            if (
                self.detection_result.face_blendshapes[0][9].score
                > self.score_threshold
                or self.detection_result.face_blendshapes[0][10].score
                > self.score_threshold
            ):
                self.looking_direction_text = "Looking down"
                return self.looking_direction_text
            elif (
                self.detection_result.face_blendshapes[0][11].score
                > self.score_threshold
                or self.detection_result.face_blendshapes[0][12].score
                > self.score_threshold
            ):
                self.looking_direction_text = "Looking right"
                return self.looking_direction_text
            elif (
                self.detection_result.face_blendshapes[0][14].score
                > self.score_threshold
                or self.detection_result.face_blendshapes[0][15].score
                > self.score_threshold
            ):
                self.looking_direction_text = "Looking left"
                return self.looking_direction_text
            elif (
                self.detection_result.face_blendshapes[0][17].score
                > self.score_threshold
                or self.detection_result.face_blendshapes[0][18].score
                > self.score_threshold
            ):
                self.looking_direction_text = "Looking up"
                return self.looking_direction_text
            else:
                self.looking_direction_text = "Looking to the front"
                return self.looking_direction_text
        except Exception as error:
            logger.info("An error occurred:", type(error).__name__)
            self.looking_direction_text = "No face detected!"


if __name__ == "__main__":
    video_source = "/home/martin/Downloads/sphinxai/Juan ojo derecho.mp4"
    video_cap = VideoCapture(video_source)
    video_writer = VideoWriterFromCapture(
        "/home/martin/Downloads/sphinxai/video_proc.mpp4", video_cap
    )
    progress_bar = VideoProgressBar(video_cap)
    # # eye_manager = EyeAnalyzer(video_cap, video_writer, progress_bar)
    # eye_manager.process_video()
    # eye_manager.process_video()
