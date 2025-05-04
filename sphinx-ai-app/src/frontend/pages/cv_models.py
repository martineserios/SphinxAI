import os
from typing import Dict

import streamlit as st

from sphinx_ai.config import INPUT_DIR, OUTPUT_DIR
from sphinx_ai.eye import EyeAnalyzer
from sphinx_ai.gestures import GestureAnalyzer
from sphinx_ai.head import HeadAnalyzer
from sphinx_ai.pose import PoseEstimator
from sphinx_ai.utilities import VideoProgressBar
from sphinx_ai.videos.io import VideoCapture, VideoWriterFromCapture

st.header("🗃 CV Models")
progress_bar = st.empty()


def get_video_paths(input_video: str):
    # create input and proc output video filename and path
    input_filename = input_video
    input_filepath = os.path.join(INPUT_DIR, input_filename)
    processed_filename = os.path.splitext(input_video)[0] + "_proc.mp4"
    processed_filepath = os.path.join(OUTPUT_DIR, processed_filename)
    out = None

    return input_filename, input_filepath, processed_filename, processed_filepath, out


def get_video_paths(input_video: str) -> Dict[str, str]:
    input_filename = input_video
    input_filepath = os.path.join(INPUT_DIR, input_filename)
    processed_filename = f"{os.path.splitext(input_video)[0]}_proc.mp4"
    processed_filepath = os.path.join(OUTPUT_DIR, processed_filename)

    return {
        "input_filename": input_filename,
        "input_filepath": input_filepath,
        "processed_filename": processed_filename,
        "processed_filepath": processed_filepath,
    }


models_params = {
    "Blink Detector": {"blink_detection_threshold": None, "num_faces": None},
    "Headpose Estimator": {
        "min_detection_confidence": None,
        "min_tracking_confidence": None,
    },
    "Pose Estimator": {
        "min_detection_confidence": None,
        "min_tracking_confidence": None,
    },
    "Gesture Detector": {"gesture_detection_threshold": None, "num_faces": None},
}

with st.sidebar:
    input_video = st.file_uploader("Upload video", type=["mp4", "avi"])

    # if input_video:
    # initialize processing options
    options = {
        "Headpose Estimator": HeadAnalyzer,
        "Pose Estimator": PoseEstimator,
        "Blink Detector": EyeAnalyzer,
        "Gesture Detector": GestureAnalyzer,
    }
    selected_option = st.sidebar.selectbox("Select an option", list(options.keys()))

    # expose parameters depending on the model chosen
    if selected_option == "Blink Detector":
        models_params["Blink Detector"][
            "blink_detection_threshold"
        ] = st.sidebar.slider(
            "Blink Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            key="blink_detection_threshold",
        )
        models_params["Blink Detector"]["num_faces"] = st.number_input(
            "Number of faces to detect", value=1
        )

    elif selected_option == "Headpose Estimator":
        models_params["Headpose Estimator"][
            "min_detection_confidence"
        ] = st.sidebar.slider(
            "Detection Confidence",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            key="head_min_detection_confidence",
        )
        models_params["Headpose Estimator"][
            "min_tracking_confidence"
        ] = st.sidebar.slider(
            "Tracking Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            key="head_min_tracking_confidence",
        )

    elif selected_option == "Pose Estimator":
        models_params["Pose Estimator"][
            "min_detection_confidence"
        ] = st.sidebar.slider(
            "Detection Confidence",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            key="pose_min_detection_confidence",
        )
        models_params["Pose Estimator"][
            "min_tracking_confidence"
        ] = st.sidebar.slider(
            "Tracking Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            key="pose_min_tracking_confidence",
        )

    elif selected_option == "Gesture Detector":
        models_params["Gesture Detector"][
            "gesture_detection_threshold"
        ] = st.sidebar.slider(
            "Gesture Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            key="gesture_detection_threshold",
        )
        models_params["Gesture Detector"]["num_faces"] = st.number_input(
            "Number of faces to detect", value=1
        )
    process_button = st.sidebar.button("Process", key="download-button")



tests_types = {
    "Test Inicial Vision": {
        "Sacádicos": [
            "Izquiera a Derecha de Arriba hacia Abajo",
            "Derecha a Izquierda de Arriba hacia Abajo",
            "Izquiera a Derecha de Abajo hacia Arriba",
            "Derecha a Izquierda de Abajo hacia Arriba"
        ],
        "Vision Periferica": [
            "Logical",
            "Creative"
        ],
        "Resistencia Ocular": ["Resistencia Ocular"]
    },
    "Test Médico": {
        "Smooth Pursuit": [
            "Horizontal",
            "Vertical",
            "Circular"
        ],
        "Object Tracking": ["Object Tracking"],
        "Fijación/Estbilidad": ["Fijación/Estbilidad"],
        "Reaction Time Sacádicos": [
            "Horizontal",
            "Vertical"
        ]

    } 
}

stream_placeholder = st.empty()

if input_video:
    # Save the input video file
    input_filename = input_video.name
    input_filepath = os.path.join(INPUT_DIR, input_filename)
    with open(input_filepath, "wb") as f:
        f.write(input_video.getbuffer())
    # read video
    captured_video = VideoCapture(input_filepath=input_filepath)

    # Show original video
    st.subheader("Original Video")
    st.video(input_video)
    (
        input_filename,
        input_filepath,
        processed_filename,
        processed_filepath,
        out,
    ) = get_video_paths(input_video.name)

if input_video and process_button:
    # Create a progress bar to show the progress of the transformation
    progress_bar = VideoProgressBar(captured_video, progress_bar)

    # instantiate output video writer
    writer_out = VideoWriterFromCapture(processed_filepath, captured_video)

    model = options[selected_option](
            captured_video,
            writer_out,
            progress_bar,
            models_params,
            stream_placeholder,
        )

    model.process_video()

    # if writer_out is not None:
    writer_out.writer.release()

    # show processed video
    st.subheader("Transformed Video")
    st.video(processed_filepath)

    # Download transformed video
    with open(processed_filepath, "rb") as f:
        video_bytes = f.read()
        st.download_button(
            label="Download",
            data=video_bytes,
            file_name=processed_filepath,
            mime="video/mp4",
        )


if __name__ == "__main__":
    # Create input and output directories if they don't exist
    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)