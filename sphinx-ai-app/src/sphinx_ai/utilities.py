# import libraries
import subprocess
from functools import reduce

import cv2
import ffmpeg
import numpy as np
import streamlit as st
from sphinx_ai.utils.logging_config import logger


class VideoProgressBar:
    def __init__(self, video_capture):
        self.total_frames = video_capture.total_frames
        self.progress_bar = st.progress(0)
        self.current_frame_number = 0

    def update(self):
        # update progress bar
        self.current_frame_number += 1
        logger.info(
            f"Frame: {self.current_frame_number}/ {self.total_frames} | {round(min([(self.current_frame_number/ self.total_frames) * 100, 100]), 2)}%"
        )
        self.progress_bar.progress(
            min([int((self.current_frame_number / self.total_frames) * 100), 100])
        )

    def reset(self):
        self.current_frame = 0
        self.progress_bar.progress(0)

    def empty(self):
        self.progress_bar.empty()


class ProgressBar:
    def __init__(self, iterable):
        self.total = len(iterable)
        self.progress_bar = st.progress(0)
        self.current_iteration = 0

    def update(self):
        # update progress bar
        self.current_iteration += 1
        logger.info(
            f"{round(min([(self.current_iteration/ self.total) * 100, 100]), 2)}%"
        )
        self.progress_bar.progress(
            min([int((self.current_iteration / self.total) * 100), 100])
        )

    def reset(self):
        self.current_iteration = 0
        self.progress_bar.progress(0)

    def empty(self):
        self.progress_bar.empty()


## frame/images utils
def extend_image_height(image: np.ndarray, value: int, color: str, dimension="height"):
    """
    Extends the size of the image.
    diensión can be either: height or width.
    """
    if dimension == "height":
        image_extended = np.ndarray(
            (image.shape[0] + value,) + image.shape[1:], dtype=image.dtype
        )
        image_extended[: image.shape[0], :] = image
        image_extended[image.shape[0] :, :] = color
    elif dimension == "width":
        image_extended = np.ndarray(
            image.shape[0] + (image.shape[1:] + value), dtype=image.dtype
        )
        image_extended[:, : image.shape[1]] = image
        image_extended[:, image.shape[1] :] = color

    return image_extended


def black_and_white(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray


## set the video and audio encoding options
def encode_h264(input_file, output_file, crf=23):
    cmd = [
        "ffmpeg",
        "-i",
        input_file,
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        str(crf),
        "-c:a",
        "copy",
        output_file,
        "-y",
    ]
    subprocess.run(cmd, check=True)


def preview_player(
    model,
    frame_writer,
    input_filepath,
    encoded_filepath,
    captured_video,
    progress_bar,
    place,
    counter,
):
    # Encoding the frame in h.264
    process1 = (
        ffmpeg.input(input_filepath)
        .output(
            "pipe:",
            format="rawvideo",
            pix_fmt="bgr24",
            s="{}x{}".format(captured_video.width, captured_video.height),
        )  # , vframes=8)
        .run_async(pipe_stdout=True)
    )

    process2 = (
        ffmpeg.input(
            "pipe:",
            format="rawvideo",
            pix_fmt="bgr24",
            s=f"{captured_video.width}x{captured_video.height}",
        )  # , vframes=8)
        .output(
            encoded_filepath,
            pix_fmt="yuv420p",
            vcodec="libx264",
            crf=23,
            s="{}x{}".format(
                captured_video.width,
                captured_video.height,
            ),
        )
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )

    while True:
        in_bytes = process1.stdout.read(
            (captured_video.width * captured_video.height * captured_video.channels)
        )
        if not in_bytes:
            break
        in_frame = np.fromstring(in_bytes, np.uint8).reshape(
            (
                captured_video.height,
                captured_video.width,
                captured_video.channels,
            )
        )
        logger.info(f"Channels: {captured_video.channels}")
        logger.info(in_frame.shape)
        # try:
        # Apply selected processing option
        proc_frame = model.process_frame(in_frame)
        annotated_frame = frame_writer.draw_head_axis(
            frame=proc_frame,
            yaw=model.y_rot,
            pitch=model.x_rot,
            roll=model.z_rot,
            tdx=int(model.nose_2d[0]),
            tdy=int(model.nose_2d[1]),
        )
        # except:
        #     annotated_frame = in_frame
        #     logger.error(f"Error processing frame: {i}")

        # except Exception as error:
        #     # handle the exception
        #     logger.info("An exception occurred:", type(error).__name__, "–", error)

        process2.stdin.write(
            annotated_frame
            # .astype(np.uint8)
            # .tobytes()
        )

        # show transformed frames
        if counter == 0:
            preview_place = place.empty()
            preview_place = st.image(annotated_frame)

        # annontated_frame = in_frame
        # update progress bar
        counter = counter + 1
        progress_bar.progress(
            min([int((counter / captured_video.total_frames) * 100), 100])
        )

        if counter % 6 == 0:
            # while col2:
            preview_place.empty()
            preview_place.image(annotated_frame)

    process2.stdin.close()
    process1.wait()
    process2.wait()


## pose utils
def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(
        a[1] - b[1], a[0] - b[0]
    )
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


## head_analysis utils
def HEAD_COMPENSATION_MAP(value):
    value = abs(value)
    category = ""

    if value <= abs(4):
        category = "PRO"
    elif (value > abs(4)) & (value <= abs(8)):
        category = "AR"
    elif (value > abs(8)) & (value <= abs(16)):
        category = "AM"
    elif value > abs(16):
        category = "BEG"
    return category


## general
def percent_elements_in_dict(categs_dict):
    total = sum(categs_dict.values())
    percent = {
        key: f"{round(100 * (value/total))}%" for key, value in categs_dict.items()
    }

    return percent


def list_mean(lst):
    return reduce(lambda a, b: a + b, lst) / len(lst)


def stop_cv_reading():
    key = 256 * 27
    return key


def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size=50, thickness=(2, 2, 2)):
    """
    Function used to draw y (headpose label) on Input Image x.
    Implemented by: shamangary
    https://github.com/shamangary/FSA-Net/blob/master/demo/demo_FSANET.py
    Modified by: Omar Hassan
    """
    pitch = pitch * np.pi / 180
    yaw = yaw * np.pi / 180
    roll = roll * np.pi / 180

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (np.cos(yaw) * np.cos(roll)) + tdx
    y1 = (
        size
        * (np.cos(pitch) * np.sin(roll) + np.cos(roll) * np.sin(pitch) * np.sin(yaw))
        + tdy
    )

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-np.cos(yaw) * np.sin(roll)) + tdx
    y2 = (
        size
        * (np.cos(pitch) * np.cos(roll) - np.sin(pitch) * np.sin(yaw) * np.sin(roll))
        + tdy
    )

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (np.sin(yaw)) + tdx
    y3 = size * (-np.cos(yaw) * np.sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 0, 255), thickness[0])
    cv2.line(img, (int(tdx), int(tdy)), (int(x2), int(y2)), (0, 255, 0), thickness[1])
    cv2.line(img, (int(tdx), int(tdy)), (int(x3), int(y3)), (255, 0, 0), thickness[2])

    return img
