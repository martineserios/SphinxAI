# ref: https://storage.googleapis.com/mediapipe-assets/documentation/mediapipe_face_landmark_fullsize.png


import cv2
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

from sphinx_ai.utilities import VideoProgressBar
from sphinx_ai.utils.logging_config import logger
from sphinx_ai.videos.io import VideoWriterFromCapture
from sphinx_ai.videos.models import VideoCapture, VideoModelAppInterface


def get_optimal_font_scale(text, width, font, thickness=1):
    for scale in reversed(range(0, 60, 1)):
        textSize = cv2.getTextSize(
            text[1:4], fontFace=font, fontScale=scale / 10, thickness=thickness
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
        logger.info(kwargs["position"])
        y_position = kwargs["position"][1]
        kwargs["position"] = (10, 15 + int(y_position * (scale)))
        logger.info(kwargs["position"])
        return cv2_func_put_text_func(*args, **kwargs)

    return decor


class GestureAnalyzer(VideoModelAppInterface):
    def __init__(
        self,
        video_capture,
        video_writer,
        progress_bar,
        models_params,
        st_output_streaming,
    ):
        self.num_faces = models_params["Gesture Detector"]["num_faces"]
        self.gesture_detection_threshold = models_params["Gesture Detector"][
            "gesture_detection_threshold"
        ]
        self.instanciate_model()
        super().__init__(self.model, video_capture, video_writer, progress_bar)
        self.capture = video_capture
        self.writer = video_writer
        self.st_output_streaming = st_output_streaming

        # variables
        self.frame_counter = 0
        self.CEF_COUNTER = 0
        self.TOTAL_BLINKS = 0
        # constants
        self.CLOSED_EYES_FRAME = 3
        self.FONTS = cv2.FONT_HERSHEY_COMPLEX

        self.frame_counter = 0
        self.prev_tick = cv2.getTickCount()
        logger.info("Model initialized")

    color = (0, 255, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # dict for translating gestures name
    blendshapes_dict = {
        "browDownLeft": "CejaAbajoIzq",
        "browDownRight": "CejaAbajoDer",
        "browInnerUp": "CejaInteriorArriba",
        "browOuterUpLeft": "CejaExteriorArribaIzq",
        "browOuterUpRight": "CejaExteriorArribaDer",
        "cheekPuff": "InflarMejilla",
        "cheekSquintLeft": "FruncirMejillaIzq",
        "cheekSquintRight": "FruncirMejillaDer",
        "eyeBlinkLeft": "ParpadeoIzq",
        "eyeBlinkRight": "ParpadeoDer",
        "eyeLookDownLeft": "MirarAbajoIzq",
        "eyeLookDownRight": "MirarAbajoDer",
        "eyeLookInLeft": "MirarInteriorIzq",
        "eyeLookInRight": "MirarInteriorDer",
        "eyeLookOutLeft": "MirarExteriorIzq",
        "eyeLookOutRight": "MirarExteriorDer",
        "eyeLookUpLeft": "MirarArribaIzq",
        "eyeLookUpRight": "MirarArribaDer",
        "eyeSquintLeft": "EntrecerrarOjosIzq",
        "eyeSquintRight": "EntrecerrarOjosDer",
        "eyeWideLeft": "OjosAbiertosIzq",
        "eyeWideRight": "OjosAbiertosDer",
        "jawForward": "MandíbulaAdelante",
        "jawLeft": "MandíbulaIzq",
        "jawOpen": "MandíbulaAbierta",
        "jawRight": "MandíbulaDer",
        "mouthClose": "CerrarBoca",
        "mouthDimpleLeft": "HoyueloIzq",
        "mouthDimpleRight": "HoyueloDer",
        "mouthFrownLeft": "FruncirBocaIzq",
        "mouthFrownRight": "FruncirBocaDer",
        "mouthFunnel": "EmbudoBoca",
        "mouthLeft": "BocaIzq",
        "mouthLowerDownLeft": "BajarBocaIzq",
        "mouthLowerDownRight": "BajarBocaDer",
        "mouthPressLeft": "PresionarBocaIzq",
        "mouthPressRight": "PresionarBocaDer",
        "mouthPucker": "FruncirBoca",
        "mouthRight": "BocaDer",
        "mouthRollLower": "RodarBocaInferior",
        "mouthRollUpper": "RodarBocaSuperior",
        "mouthShrugLower": "EncogerBocaInferior",
        "mouthShrugUpper": "EncogerBocaSuperior",
        "mouthSmileLeft": "SonrisaIzq",
        "mouthSmileRight": "SonrisaDer",
        "mouthStretchLeft": "EstirarBocaIzq",
        "mouthStretchRight": "EstirarBocaDer",
        "mouthUpperUpLeft": "LevantarBocaSuperiorIzq",
        "mouthUpperUpRight": "LevantarBocaSuperiorDer",
        "noseSneerLeft": "FruncirNarizIzq",
        "noseSneerRight": "FruncirNarizDer",
        "tongueOut": "LenguaAfuer",
    }

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

        gestures = self.detect_any_gestures()
        # CHECK x position is hardcoded
        y_position = 0
        if len(gestures) > 0:
            for gesture in gestures:
                y_position += 40
                self.write_text_to_frame(
                    frame=frame,
                    text=f"{self.blendshapes_dict[gesture]}",
                    position=(10, y_position),
                    width_of_screen=self.frame_width / 20,
                    font=font,
                    size=1,
                    color=color,
                    thickness=2,
                )
        return frame

    def detect_any_gestures(self, score_threshold=None):
        gestures = []
        try:
            if score_threshold is not None:
                self.gesture_detection_threshold = score_threshold

            logger.info(self.gesture_detection_threshold)
            for i, face_blendshape in enumerate(
                self.detection_result.face_blendshapes[0]
            ):
                if face_blendshape.score > self.gesture_detection_threshold:
                    gestures.append(face_blendshape.category_name)

        except Exception as error:
            logger.info("An error occurred:", type(error).__name__)
            self.looking_direction_text = "No face detected!"

        return gestures

    # def time_tick_manager(self):
    #     # if ret:
    #     self.frame_counter += 1  # frame counter
    #     self.curr_tick = cv2.getTickCount()
    #     self.time_ms = (self.curr_tick - self.prev_tick) / cv2.getTickFrequency() * 1000
    #     self.prev_tick = self.curr_tick

    #     logger.info(self.curr_tick)
    #     return self.curr_tick

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
                proc_frame = self.write_results_on_frame(frame)
                post_proc_frame = self.post_process_frame(proc_frame)

                # to ouput stream video
                post_proc_frame = cv2.cvtColor(post_proc_frame, cv2.COLOR_BGR2RGB)
                self.st_output_streaming.image(post_proc_frame, channels="RGB")

                if isinstance(self.video_capture.input_filepath, str):
                    self.video_writer.writer.write(post_proc_frame)
                    self.progress_bar.update()
                else:
                    cv2.imshow("Video", post_proc_frame)

                if cv2.waitKey(10) & 0xFF == ord("q"):
                    break
                # else:
                #     break

        self.video_capture.cap.release()


if __name__ == "__main__":
    models_params = {
        "Gesture Detector": {"gesture_detection_threshold": 0.5, "num_faces": 1},
    }

    video_source = 0  # "/home/martin/Downloads/sphinxai/Juan ojo derecho.mp4"
    video_cap = VideoCapture(video_source)
    video_writer = VideoWriterFromCapture(
        "/home/martin/Downloads/sphinxai/video_proc.mpp4", video_cap
    )
    progress_bar = VideoProgressBar(video_cap)
    eye_manager = GestureAnalyzer(video_cap, video_writer, progress_bar, models_params)
    eye_manager.process_video()
