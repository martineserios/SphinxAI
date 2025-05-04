import cv2
import mediapipe as mp
import numpy as np

from sphinx_ai.utilities import calculate_angle
from sphinx_ai.utils.logging_config import logger
from sphinx_ai.videos.models import VideoModelAppInterface


class PoseEstimator(VideoModelAppInterface):
    def __init__(
        self,
        video_capture,
        video_writer,
        progress_bar,
        models_params,
        st_output_streaming,
    ) -> None:
        self.min_detection_confidence = models_params["Headpose Estimator"][
            "min_detection_confidence"
        ]
        self.min_tracking_confidence = models_params["Headpose Estimator"][
            "min_tracking_confidence"
        ]

        logger.info(
            f"Headpose Estimator initialized with min_detection_confidence: {self.min_detection_confidence} and min_tracking_confidence: {self.min_tracking_confidence}."
        )
        self.instanciate_model()
        super().__init__(self.model, video_capture, video_writer, progress_bar)
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.video_capture = video_capture
        self.video_writer = video_writer
        self.st_output_streaming = st_output_streaming

        # maybe these should go in another class or part
        self.left_hip_coords = []
        self.right_hip_coords = []
        self.rotation_text = ""

    def instanciate_model(self):
        mp_pose = mp.solutions.pose
        self.model = mp_pose.Pose(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )

    def detect(self, frame):
        pass

    def pre_process_frame(self, frame):
        # Flip the image horizontally for a later selfie-view display
        # Also convert the color space from BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # To improve performance
        frame.flags.writeable = False
        return frame

    @staticmethod
    def post_process_frame(frame):
        # Convert the color space from RGB to BGR
        # frame = cv2.flip(frame, 1)  # cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # To improve performance
        frame.flags.writeable = True
        return frame

    def write_results_on_frame(self, frame, results):
        cv2.putText(
            frame,
            text=self.rotation_text,
            org=(200, 450),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(255, 255, 255),
            thickness=1,
            lineType=cv2.LINE_AA,
        )

        # Extract landmarks
        # try:
        if results.pose_landmarks is not None:
            landmarks = results.pose_landmarks.landmark

            # Get coordinates
            hip = [
                landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
            ]
            knee = [
                landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y,
            ]
            ankle = [
                landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y,
            ]

            # # Get coordinates
            # shoulder = [landmarks[mp_pose.PoseLandmark.Ri.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            # elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            # wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            # Calculate angle
            angle = calculate_angle(hip, knee, ankle)

            # Visualize angle
            cv2.putText(
                frame,
                str(angle),
                tuple(np.multiply(hip, [640, 480]).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # Render detections
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(
                    color=(245, 117, 66),
                    thickness=1,
                    circle_radius=1,
                ),
                self.mp_drawing.DrawingSpec(
                    color=(245, 66, 230),
                    thickness=1,
                    circle_radius=1,
                ),
            )
            # except:
            #     logger.info("An error occurred: No Landmarks")

            return frame

    def get_hip_rotation(self, results):
        if self.rotation_text == "":
            diff_z = 0
            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                landmarks_world = results.pose_world_landmarks.landmark
            except:
                pass

            # process results
            self.landmarks_pairs = [
                (landmark_pose.value, landmark_pose.name)
                for landmark_pose in [i for i in self.mp_pose.PoseLandmark.mro()[0]]
            ]

            self.left_hip_coords.append(
                (
                    landmarks_world[23].x,
                    landmarks_world[23].y,
                    landmarks_world[23].z,
                )
            )
            self.right_hip_coords.append(
                (
                    landmarks_world[24].x,
                    landmarks_world[24].y,
                    landmarks_world[24].z,
                )
            )

            diff_x = abs(self.left_hip_coords[-1][0] - self.right_hip_coords[-1][0])
            diff_z = self.left_hip_coords[-1][2] - self.right_hip_coords[-1][2]

            if diff_x < 0.03:
                if diff_z < 0:
                    self.rotation_text = "Rotated Left"
                    # return self.rotation_text

                elif diff_z > 0:
                    self.rotation_text = "Rotated Right"
                    # return self.rotation_text
            else:
                self.rotation_text = "No rotation detected"
                logger.info("No rotation detected")

    def process_video(self):
        with self.model as pose:
            while self.video_capture.cap.isOpened():
                ret, frame = self.video_capture.cap.read()
                if not ret:
                    break

                pre_proc_frame = self.pre_process_frame(frame)
                # Make detection
                results = pose.process(pre_proc_frame)
                self.get_hip_rotation(results)
                proc_frame = self.write_results_on_frame(frame, results)
                logger.info(self.rotation_text)

                post_proc_frame = self.post_process_frame(proc_frame)
                # Write the frame to the output video
                self.video_writer.writer.write(post_proc_frame)

                # to ouput stream video
                post_proc_frame = cv2.cvtColor(post_proc_frame, cv2.COLOR_BGR2RGB)
                self.st_output_streaming.image(post_proc_frame, channels="RGB")

                self.progress_bar.update()

                if cv2.waitKey(10) & 0xFF == ord("q"):
                    break

        self.video_capture.cap.release()


# if __name__ == "__main__":
#     video_cap = VideoCapture("/home/martin/Downloads/sphinxai/Juan ojo derecho.mp4")
#     video_writer = VideoWriterFromCapture(
#         "/home/martin/Downloads/sphinxai/video_proc.mpp4", video_cap
#     )
#     progress_bar = VideoProgressBar(video_cap)
#     eye_manager = PoseEstimator(
#         video_cap,
#         video_writer,
#         progress_bar,
#         models_params,
#     )
