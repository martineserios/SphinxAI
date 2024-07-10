# import packages
from collections import Counter

import cv2
import mediapipe as mp
import numpy as np

from sphinx_ai.utilities import VideoProgressBar
from sphinx_ai.utils.logging_config import logger
from sphinx_ai.videos.io import VideoCapture, VideoWriterFromCapture
from sphinx_ai.videos.models import VideoModelAppInterface


class HeadAnalyzer(VideoModelAppInterface):
    def __init__(
        self,
        video_capture,
        video_writer,
        progress_bar,
        models_params,
        st_output_streaming,
    ) -> None:
        """
        HeadAnalyzer class constructor.

        :param face_mesh: face mesh detector object.
        :type face_mesh: face_mesh.FaceMeshDetector.
        :param blink_counter: counter for counting blinks occurence.
        :type blink_counter: int.
        :param prev_blink: previous blink state.
        :type prev_blink: int.
        :param blink_counter_duration: counter for counting blink duration.
        :type blink_counter_duration: int.
        :param blink_duration: blink duration.
        :type blink_duration: int.
        :param frame_counter: counter for counting frames.
        :type frame_counter: int.
        :param bf: blinks frequency.
        :type bf: float.
        :param blinks_bag: blinks frequency bag.
        :type blinks_bag: deque.
        :param yaw_mean: yaw angles mean.
        :type yaw_mean: list.
        :param roll_mean: roll angles mean.
        :type roll_mean: list.
        """
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
        self.video_capture = video_capture
        self.st_output_streaming = st_output_streaming

        self._face_landamarks = None

        self.face_direction = ""

        # angles
        self.pitch = 0.0
        self.roll = 0.0
        self.yaw = 0.0
        # angles mean
        self.yaw_mean = []
        self.roll_mean = []
        self.pitch_mean = []
        # angles accumulator
        self.yaw_acc = []
        self.roll_acc = []
        self.pitch_acc = []
        # angles category accumulator
        self.yaw_categ_acc = {"Pro": 0, "AR": 0, "Amateur": 0, "Beginner": 0}
        self.roll_categ_acc = {"Pro": 0, "AR": 0, "Amateur": 0, "Beginner": 0}
        self.pitch_categ_acc = {"Pro": 0, "AR": 0, "Amateur": 0, "Beginner": 0}

        self.yaw_categ_acc = Counter()
        self.roll_categ_acc = Counter()
        self.pitch_categ_acc = Counter()

    def instanciate_model(self):
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        self.model = face_mesh

    def pre_process_frame(self, frame):
        # Flip the image horizontally for a later selfie-view display
        # Also convert the color space from BGR to RGB
        frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        # To improve performance
        frame.flags.writeable = False
        self.img_h, self.img_w, self.img_c = frame.shape
        self.face_3d = []
        self.face_2d = []
        return frame

    @staticmethod
    def post_process_frame(frame):
        # Convert the color space from RGB to BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # To improve performance
        frame.flags.writeable = True
        return frame

    def detect(self, frame):
        self._face_landamarks = self.model.process(frame)
        # return frame

    def get_head_angles(self, frame):
        self.img_w = frame.shape[1]
        self.img_h = frame.shape[0]
        self.face_2d = []
        self.face_3d = []
        self.nose_2d = tuple
        self.nose_3d = tuple
        self.angles = []

        if self._face_landamarks.multi_face_landmarks:
            for face_landmarks in self._face_landamarks.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if (
                        idx == 33
                        or idx == 263
                        or idx == 1
                        or idx == 61
                        or idx == 291
                        or idx == 199
                    ):
                        if idx == 1:
                            self.nose_2d = (lm.x * self.img_w, lm.y * self.img_h)
                            self.nose_3d = (
                                lm.x * self.img_w,
                                lm.y * self.img_h,
                                lm.z * 3000,
                            )

                        self.x, self.y = int(lm.x * self.img_w), int(lm.y * self.img_h)

                        # Get the 2D Coordinates
                        self.face_2d.append([self.x, self.y])

                        # Get the 3D Coordinates
                        self.face_3d.append([self.x, self.y, lm.z])

                # Convert it to the NumPy array
                self.face_2d = np.array(self.face_2d, dtype=np.float64)

                # Convert it to the NumPy array
                self.face_3d = np.array(self.face_3d, dtype=np.float64)

                # The camera matrix
                self.focal_length = 1 * self.img_w

                self.cam_matrix = np.array(
                    [
                        [self.focal_length, 0, self.img_h / 2],
                        [0, self.focal_length, self.img_w / 2],
                        [0, 0, 1],
                    ]
                )

                # The distortion parameters
                self.dist_matrix = np.zeros((4, 1), dtype=np.float64)

                # Solve PnP
                success, self.rot_vec, self.trans_vec = cv2.solvePnP(
                    self.face_3d, self.face_2d, self.cam_matrix, self.dist_matrix
                )
                # Get rotational matrix
                self.rmat, self.jac = cv2.Rodrigues(self.rot_vec)

                # Get angles
                self.angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(self.rmat)

                # Get the y rotation degree
                self.x_rot = self.angles[0] * 360
                self.y_rot = self.angles[1] * 360
                self.z_rot = self.angles[2] * 360

                self.yaw = self.y_rot
                self.pitch = self.x_rot
                self.roll = self.z_rot

                # # Display the nose direction
                # nose_3d_projection, jacobian = cv2.projectPoints(
                #     nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix
                # )

                self.p1 = (int(self.nose_2d[0]), int(self.nose_2d[1]))
                self.p2 = (
                    int(self.nose_2d[0] + self.y * 10),
                    int(self.nose_2d[1] - self.x * 10),
                )
        else:
            self.p1 = None
            self.p2 = None

        return self.yaw, self.pitch, self.roll

    def get_face_direction(self, frame, angles_threshold=8):
        self.get_head_angles(frame)
        # See where the user's head tilting
        if self.y_rot < -angles_threshold:
            self.face_direction = "Head looking Left"
        elif self.y_rot > angles_threshold:
            self.face_direction = "Head looking Right"
        elif self.x_rot < -angles_threshold:
            self.face_direction = "Head looking Down"
        elif self.x_rot > angles_threshold:
            self.face_direction = "Head Looking Up"
        else:
            self.face_direction = "Head looking Forward"
        return self.face_direction

    def write_head_axis_angles(
        self,
        frame,
        thickness: int = 2,
    ):
        # Add the text on the image
        cv2.putText(
            frame,
            self.get_face_direction(frame),
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            "x: " + str(np.round(self.x_rot, 2)),
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )
        cv2.putText(
            frame,
            "y: " + str(np.round(self.y_rot, 2)),
            (20, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )
        cv2.putText(
            frame,
            "z: " + str(np.round(self.z_rot, 2)),
            (20, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )
        return frame

    def write_results_on_frame(self, frame, axis_angles=True):
        """
        Function used to draw y (headpose label) on Input Image x.
        Implemented by: shamangary
        https://github.com/shamangary/FSA-Net/blob/master/demo/demo_FSANET.py
        Modified by: Omar Hassan
        """
        self.pitch = self.pitch * np.pi / 180
        self.yaw = self.yaw * np.pi / 180
        self.roll = self.roll * np.pi / 180

        size = 50
        thickness = (2, 2, 2)

        if (self.p1 != None) and (self.p2 != None):
            tdx = self.p1[0]
            tdy = self.p1[1]
        else:
            tdx = None
            tdy = None

        if tdx != None and tdy != None:
            tdx = tdx
            tdy = tdy
            # else:
            #     self.height, self.width = frame.shape[:2]
            #     tdx = self.width / 2
            #     tdy = self.height / 2

            # X-Axis pointing to right. drawn in red
            x1 = size * (np.cos(self.yaw) * np.cos(self.roll)) + tdx
            y1 = (
                size
                * (
                    np.cos(self.pitch) * np.sin(self.roll)
                    + np.cos(self.roll) * np.sin(self.pitch) * np.sin(self.yaw)
                )
                + tdy
            )

            # Y-Axis | drawn in green
            #        v
            x2 = size * (-np.cos(self.yaw) * np.sin(self.roll)) + tdx
            y2 = (
                size
                * (
                    np.cos(self.pitch) * np.cos(self.roll)
                    - np.sin(self.pitch) * np.sin(self.yaw) * np.sin(self.roll)
                )
                + tdy
            )

            # Z-Axis (out of the screen) drawn in blue
            x3 = size * (np.sin(self.yaw)) + tdx
            y3 = size * (-np.cos(self.yaw) * np.sin(self.pitch)) + tdy

            cv2.line(
                frame,
                (int(tdx), int(tdy)),
                (int(x1), int(y1)),
                (0, 0, 255),
                thickness[0],
            )
            cv2.line(
                frame,
                (int(tdx), int(tdy)),
                (int(x2), int(y2)),
                (0, 255, 0),
                thickness[1],
            )
            cv2.line(
                frame,
                (int(tdx), int(tdy)),
                (int(x3), int(y3)),
                (255, 0, 0),
                thickness[2],
            )

            if axis_angles:
                frame = self.write_head_axis_angles(frame)

        else:
            cv2.putText(
                frame,
                "No face detected!",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                3,
                (255, 0, 0),
                3,
            )

        return frame

    def process_video(self):
        self.instanciate_model()
        with self.model as landamrker:
            # Process video frames
            while self.video_capture.cap.isOpened():
                ret, frame = self.video_capture.cap.read()
                if not ret:
                    break

                logger.info("Reading video")
                # try:
                # Apply pre_processing to the frame
                pre_proc_frame = self.pre_process_frame(frame)
                self._face_landamarks = landamrker.process(pre_proc_frame)
                self.get_face_direction(pre_proc_frame)
                # Draw results
                proc_frame = self.write_results_on_frame(pre_proc_frame)
                # Apply post_processing to the frame

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


if __name__ == "__main__":
    video_cap = VideoCapture("/home/martin/Downloads/sphinxai/Juan ojo derecho.mp4")
    video_writer = VideoWriterFromCapture(
        "/home/martin/Downloads/sphinxai/video_proc.mpp4", video_cap
    )
    progress_bar = VideoProgressBar(video_cap)
    eye_manager = HeadAnalyzer(video_cap, video_writer, progress_bar)
