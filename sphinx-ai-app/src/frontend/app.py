# package imports
import asyncio
import datetime
import os
import tempfile
import uuid

import cv2
import duckdb as ddb
import numpy as np
import pandas as pd
import streamlit as st
from loguru import logger
from moviepy.editor import AudioFileClip

from sphinx_ai.audio import SpeechToText
from sphinx_ai.config import (DB_LOCATION, INPUT_DIR, OUTPUT_DIR,
                              OUTPUT_TRANSCRIPTIONS_BUCKET_NAME)
from sphinx_ai.eye import EyeAnalyzer
from sphinx_ai.gestures import GestureAnalyzer
from sphinx_ai.head import HeadAnalyzer
from sphinx_ai.pose import PoseEstimator
from sphinx_ai.pupils_glasses.data import upload_data
from sphinx_ai.utilities import ProgressBar, VideoProgressBar
from sphinx_ai.utils.logging_config import logger

if "file_uploader_key" not in st.session_state:
    st.session_state["file_uploader_key"] = 100

if "uploaded_files" not in st.session_state:
    st.session_state["uploaded_files"] = []

st.set_page_config(layout="wide")

@st.cache_resource
def connect_to_database(db_path: str, read_only: bool):
    return ddb.connect(database=db_path, read_only=False)

# @st.cache_data
# def query_db(db_conn, query):
#     return db_conn.execute(query).df()


def query(db_path:str, query: str, ttl: int = 3600, **kwargs) -> pd.DataFrame:
    @st.cache_data(ttl=ttl)
    def _query(db_path:str, query: str, read_only=bool, **kwargs) -> pd.DataFrame:
        cursor = connect_to_database(db_path, False)
        cursor.execute(query, **kwargs)

        logger.info("db usage")
        return cursor.df()

    return _query(db_path, query, **kwargs)

db_conn = connect_to_database(DB_LOCATION, False)


detection_threshold = None


def get_video_paths(input_video: str):
    # create input and proc output video filename and path
    input_filename = input_video
    input_filepath = os.path.join(INPUT_DIR, input_filename)
    processed_filename = os.path.splitext(input_video)[0] + "_proc.mp4"
    processed_filepath = os.path.join(OUTPUT_DIR, processed_filename)
    out = None

    return input_filename, input_filepath, processed_filename, processed_filepath, out


def create_semi_random_names(initial_string):
    # The generated bucket name must be between 3 and 63 chars long
    return "".join([initial_string, str(uuid.uuid4())])

def configure_sidebar() -> None:
    """
    Setup and display the sidebar elements.

    This function configures the sidebar of the Streamlit application,
    including the form for user inputs and the resources section.
    """
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

        return input_video, options, selected_option, process_button, models_params
    # else:
    #     return


def main_page(
    input_video, options, selected_option, process_button, models_params
) -> None:
    st.title("EyeTracker")

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

    tab1, tab2, tab3, tab4 = st.tabs(["🗃 Load Video", "🔴 Transcription", "⏫ Upload data", "📈 Results"])
    # tab1, tab2, tab3, tab4 = st.tabs(["🔴 Transcription", "⏫ Upload data", "📈 Results", "🗃 Load Video"])
    with tab1:
        tab1.subheader("🗃 Load Video")
        stream_placeholder = tab1.empty()

        if input_video:
            # Save the input video file
            input_filename = input_video.name
            input_filepath = os.path.join(INPUT_DIR, input_filename)
            with open(input_filepath, "wb") as f:
                f.write(input_video.getbuffer())
            # read video
            captured_video = VideoCapture(input_filepath=input_filepath)

           # Show original video
            tab1.subheader("Original Video")
            tab1.video(input_video)
            (
                input_filename,
                input_filepath,
                processed_filename,
                processed_filepath,
                out,
            ) = get_video_paths(input_video.name)
 
        if input_video and process_button:
            # Create a progress bar to show the progress of the transformation
            progress_bar = VideoProgressBar(captured_video)

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
            tab1.subheader("Transformed Video")
            tab1.video(processed_filepath)

           # Download transformed video
            with open(processed_filepath, "rb") as f:
                video_bytes = f.read()
                st.download_button(
                    label="Download",
                    data=video_bytes,
                    file_name=processed_filepath,
                    mime="video/mp4",
               )
    with tab2:
        audio_bytes = None
        audio_file_path = None

        def extract_audio(mp4_path):
            """
            Extracts the audio from an MP4 file as an MP3 and saves it in the same directory.

            Args:
                mp4_path: The path to the input MP4 file.

            Returns:
                The path to the extracted MP3 file, or None if an error occurs.
            """

            if not os.path.isfile(mp4_path):
                return None  # MP4 file not found

            mp3_path = os.path.splitext(mp4_path)[0] + ".mp3"

            try:
                # Load the MP4 file
                audio_clip = AudioFileClip(mp4_path)

                # Extract and write the audio as MP3
                audio_clip.write_audiofile(mp3_path)

                # Close the clip (important to avoid potential errors)
                audio_clip.close()
            except Exception as e:
                logger.info(f"Error extracting audio: {e}")
                return None

            return mp3_path



        tab1.header("🔴 Transcription")
        upload_option = st.radio(
            "Choose input method:",
            ("Upload MP3 or MP4 file.", "Record Audio")
        )

        if upload_option == "Upload MP3 or MP4 file.":
            transcribed_text = None
            input_file = st.file_uploader("Upload Video or Audio File", type=["mp4", "mp3"])
            if input_file:
                logger.info(input_file.name)
                if input_file.name.split(".")[-1] in {"mp3"}:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
                        if temp_file:
                            temp_file.write(input_file.read())
                            audio_file_path = temp_file.name
                    
                elif input_file.name.split(".")[-1] in {"mp4"}:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
                        if temp_file:
                            temp_file.write(input_file.read())
                            temp_file_path = temp_file.name

                            st.video(input_file)
                            audio_file_path = extract_audio(temp_file_path)
                            logger.info('Pass')


        elif upload_option == "Record Audio":

            st.title("Audio Recorder")


            wav_audio_data = st_audiorec()
            if wav_audio_data is not None:
                # st.audio(wav_audio_data)

                # Save to file (optional)
                audio_file_path = "/tmp/recorded_audio.wav"
                with open(audio_file_path, "wb") as f:
                    f.write(wav_audio_data)


        transc_lang = st.selectbox(
            "Idioma de la transcripción:",
            ("es-ES", "es-US", 'en-US')
            )

        if st.button("Transcribe. Un click y esperar. Hay que tener paciencia como dice Pablo José."):
            with st.spinner('Wait for it...'):
                transcriber = SpeechToText(OUTPUT_TRANSCRIPTIONS_BUCKET_NAME, transc_lang)
                st.session_state.transcribed_text = asyncio.run(transcriber.transcribe_file(audio_file_path))
            st.text_area(
                'Transcription'
                , st.session_state.transcribed_text
            )
    with tab3:
        tab2.subheader("Upload Data")
        stream_placeholder = tab2.empty()

        athlete_name = None
        test_category = None
        test_name = None
        test_variation = None

        def reset_upload_filters():
            st.session_state.athlete_name = None
            st.session_state.test_category = None
            st.session_state.test_name = None
            st.session_state.test_variation = None
            st.session_state["file_uploader_key"] += 1
            


        athlets_query = """
                SELECT DISTINCT athlets.id, athlets.name 
                FROM main.athlets AS athlets
                ORDER BY athlets.name ASC
        """
        athlets = query(db_path=DB_LOCATION, query=athlets_query)
        athlete_name = st.selectbox(
            label="Athlete name 👇",
            options=list(athlets['name'].values) + ["New Athlete"],
            # label_visibility=st.session_state.visibility,
            # disabled=st.session_state.disabled,
            # placeholder=st.session_state.placeholder,
            index=None,
            key='athlete_name'
        )
        # Create text input for user entry
        if athlete_name == "New Athlete": 
            athlete_name = st.text_input("Enter new athlete here...")

        if athlete_name is not None:
            test_category = st.selectbox(
                "Test Category 👇",
                options=tests_types.keys(),
                # label_visibility=st.session_state.visibility,
                # disabled=st.session_state.disabled,
                # placeholder=st.session_state.placeholder,
                index=None,
                key='test_category'
            )
        if test_category is not None:
            test_name = st.selectbox(
                "Test Name 👇",
                options=tests_types[test_category].keys(),#tests_types[test_category].keys()
                # label_visibility=st.session_state.visibility,
                # disabled=st.session_state.disabled,
                # placeholder=st.session_state.placeholder,
                # placeholder=st.session_state.placeholder,
                index=None,
                key="test_name"
            )
        if test_name is not None:
            test_variation = st.selectbox(
                "Test Variation 👇",
                options=tests_types[test_category][test_name],
                # label_visibility=st.session_state.visibility,
                # disabled=st.session_state.disabled,
                index=None,
                key="test_variation"
            )
        if test_name is not None:
            aim = st.number_input(
                "AIM Socre 👇",
                key="aim_score"
            )

        # with st.form(key="pupil_form", clear_on_submit=True):

        uploaded_files = st.file_uploader(
            "Choose Pupil'sexport files",
            accept_multiple_files=True,
            type={"csv"},
            key=st.session_state["file_uploader_key"]
        )
        if uploaded_files:
            if tab3.button(label="Process"):
                data_file_check = any(
                    [
                        file
                        for file in uploaded_files
                        if file.name == "iMotions_info.csv"
                    ]
                )
                if data_file_check is True:
                    st.write("Cool!")
                    upload_check = asyncio.run(
                        upload_data(
                            uploaded_files,
                            db_conn,
                            athlete_name,
                            # gender,
                            # sport,
                            # position,
                            # birth_date,
                            test_category,
                            test_name,
                            test_variation,
                            aim
                        )
                    )

                    if upload_check:
                        st.write("Cool and Done!")
                        st.button('Reset', on_click=reset_upload_filters)

                        
                else:
                    st.write("Not cool!")

            # submit_button = st.form_submit_button(label="Submit")
    with tab4:
        st.header("Filters")
        col_a1, col_a2, col_a3, col_a4 = st.columns(4)

        athlete = None
        test_category = None
        test_name = None
        test_variation = None
        test_id = None

        # with col_a1:
        athlets = query(
            db_path=DB_LOCATION,
            query="""
            SELECT DISTINCT athlets.id, athlets.name 
            FROM main.athlets AS athlets
            """
        )
        athlete = st.selectbox("Choose athlete", athlets["name"], key=5, index=None)
        if athlete is not None:
            athlete_id = athlets[athlets["name"] == athlete]["id"].values[0]

        if athlete is not None:
            logger.info(athlete, athlete_id, type(athlete_id))
            # with col_a2:
            test_category = st.selectbox("Category", tests_types.keys(), key=6, index=None)

        if test_category is not None:
            # with col_a3:
            test_name = st.selectbox("Test", tests_types[test_category].keys(), key=7, index=None)

        if test_name is not None:
            # with col_a4:
            test_variation = st.selectbox("Variation", tests_types[test_category][test_name], key=8, index=None)

        if test_variation is not None:
            test_datetime_df = db_conn.execute(
                f"""
                    SELECT DISTINCT tests.id, tests.test_date, tests.test_time
                    FROM main.tests AS tests
                    WHERE tests.athlete_id = '{athlete_id}'
                        AND tests.category = '{test_category}'
                        AND tests.test_name = '{test_name}'
                        AND tests.variation = '{test_variation}'
                """
            ).df()
            test_datetime_df['merged_datetime'] = test_datetime_df.apply(
                lambda row: datetime.datetime.combine(row['test_date'].date(), row['test_time'].time()), axis=1
            )
            test_id_list = test_datetime_df['merged_datetime'].astype(str).tolist()

            logger.info(test_id_list)


            # with col_a4:
            test_datetime = st.selectbox("Date & Time of Test", test_id_list, key=9, index=None)


        st.header("")
        st.header("Stats")
        # get data 
        if st.button("See Stats"):
            # try:
            st.subheader("Blinking")
            col_b1, col_b2, col_b3 = st.columns(3)

            ## duration
            with col_b1:

                def get_duration_score_category(score:float) -> str:
                    if (score >= 0) & (score < 20):
                        return "Pro"
                    elif (score >= 20) & (score < 40):
                        return "AR"
                    elif (score >= 40) & (score < 60):
                        return "AM"
                    elif (score >= 60):
                        return "Beginner"
                    else:
                        return "Score error"


                test_id = test_datetime_df[test_datetime_df['merged_datetime'] == test_datetime]['id'].values[0]
                logger.info(test_id)
                duration = db_conn.execute(
                        f"""
                            SELECT tests.id, duration
                            FROM main.tests AS tests
                            WHERE tests.id  = '{test_id}'
                        """
                    ).df()['duration'].values[0]
                duration_timestamp = pd.Timestamp(duration)
                st.write('Test Duration')
                seconds = f"{duration_timestamp.second} s"
                st.metric(label=get_duration_score_category(duration_timestamp.second),  value=seconds)


            ## blinks
            with col_b2:
                total_blinks = db_conn.execute(
                        f"""
                            SELECT COUNT(DISTINCT blinks.id) as blinks
                            FROM main.blinks as blinks
                            WHERE blinks.test_id  = '{test_id}'
                        """
                    ).df()['blinks'].values[0]
                st.metric(label="Total Blinks",  value=total_blinks)
                
            with col_b3:
                ## blinks every 10 seconds
                blinks_ten_secs = db_conn.execute(
                        f"""
                            SELECT test_id, AVG(blinks_ten_secs) AS blinks_ten_sec
                            FROM (
                                SELECT blinks.test_id, blinks.end_timestamp, COUNT(DISTINCT(id)) OVER (PARTITION BY blinks.test_id ORDER BY blinks.end_timestamp RANGE BETWEEN INTERVAL 10 SECOND PRECEDING AND INTERVAL 0 SECOND FOLLOWING) AS blinks_ten_secs
                                FROM main.blinks as blinks
                                WHERE blinks.test_id  = '{test_id}'
                                )
                            GROUP BY test_id
                        """
                    ).df()['blinks_ten_sec'].values[0]
                st.metric(label="Blinks / 10s",  value=round(blinks_ten_secs, 1))


            st.subheader("Head Angles")

            head_angles = db_conn.execute(
                    f"""
                        WITH yaw_pitch_roll_vals as (
                                SELECT 
                                    hpt.test_id, 
                                    HOUR(hpt.timestamp), 
                                    MINUTE(hpt.timestamp), 
                                    SECOND(hpt.timestamp),
                                    AVG(hpt.yaw) AS yaw,
                                    AVG(hpt.pitch) AS pitch,
                                    AVG(hpt.roll) AS roll
                                FROM main.head_poses_tracker as hpt
                                WHERE hpt.test_id  = '{test_id}'
                                GROUP BY
                                    hpt.test_id, 
                                    HOUR(hpt.timestamp), 
                                    MINUTE(hpt.timestamp), 
                                    SECOND(hpt.timestamp)
                        )
                        SELECT
                            yprv.test_id,
                            AVG(yprv.yaw) AS yaw_val, 
                            CASE 
                                WHEN AVG(ABS(yprv.yaw)) <= 2 THEN 'Pro'
                                WHEN AVG(ABS(yprv.yaw)) > 2 AND AVG(ABS(yprv.yaw)) <= 5 THEN 'AR'
                                WHEN AVG(ABS(yprv.yaw)) > 5 AND AVG(ABS(yprv.yaw)) <= 10 THEN 'AM'
                                ELSE 'Beginner'
                        END AS yaw_categ,
                        AVG(yprv.pitch) AS pitch_val,
                        CASE 
                                WHEN AVG(ABS(yprv.pitch)) <= 2 THEN 'Pro'
                                WHEN AVG(ABS(yprv.pitch)) > 2 AND AVG(ABS(yprv.pitch)) <= 5 THEN 'AR' 
                                WHEN AVG(ABS(yprv.pitch)) > 5 AND AVG(ABS(yprv.pitch)) <= 10 THEN 'AM'
                            ELSE 'Beginner'
                            END AS pitch_categ,
                            AVG(yprv.roll) AS roll_val, 
                            CASE 
                                WHEN AVG(ABS(yprv.roll)) <= 2 THEN 'Pro'
                                WHEN AVG(ABS(yprv.roll)) > 2 AND AVG(ABS(yprv.roll)) <= 5 THEN 'AR'
                                WHEN AVG(ABS(yprv.roll)) > 5 AND AVG(ABS(yprv.roll)) <= 10 THEN 'AM'
                                ELSE 'Beginner'
                            END AS roll_categ
                        FROM yaw_pitch_roll_vals AS yprv
                        GROUP BY yprv.test_id
                    """
                ).df()

            col_c1, col_c2, col_c3 = st.columns(3)
            with col_c1:
                st.write("Yaw")
                yaw = round(head_angles['yaw_val'].values[0], 1)
                yaw_categ = head_angles['yaw_categ'].values[0]
                st.metric(label=yaw_categ,  value=yaw)

            with col_c2:
                st.write("Pitch")
                pitch = round(head_angles['pitch_val'].values[0], 1)
                pitch_categ = label=head_angles['pitch_categ'].values[0]
                st.metric(pitch_categ,  value=pitch)

            with col_c3:
                st.write("Roll")
                roll = round(head_angles['roll_val'].values[0], 1)
                roll_categ = head_angles['roll_categ'].values[0]
                st.metric(label=roll_categ,  value=roll)
            # except:
            #     st.write("There are no registered tests.")


            # col_c1, col_c2 = st.columns(2)
            # with col_c1:

            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.subheader("Fixations per Surface")

                fixations_p_surface = db_conn.execute(
                        f"""
                            SELECT gaze_s.surface, COUNT(DISTINCT gaze_s.world_index) 
                            FROM main.gaze_positions_on_surfaces as gaze_s
                            JOIN main.blinks AS blinks
                                ON gaze_s.world_index  = blinks.index 
                            WHERE gaze_s.test_id  = '{test_id}'
                            GROUP BY gaze_s.surface, gaze_s.on_surf
                            HAVING gaze_s.on_surf = 1

                        """
                    ).df()
                st.pyplot(fixations_p_surface.plot.barh().figure)

            with col_d2:
                st.subheader("Gaze Heatmap")

                fixations_list = db_conn.execute(
                    f"""
                        SELECT fixations.norm_pos_x, fixations.norm_pos_y 
                        FROM main.fixations AS fixations
                        WHERE fixations.test_id = '{test_id}'
                    """
                ).df().values.tolist()

                def plot_heatmap(fixations:list, backgorund_image:str):
                    # Your Data
                    # points = [(100, 150), (250, 80), (300, 300), ...]  # Your list of (x, y) points

                    # Load Image
                    img = cv2.imread(backgorund_image)
                    height, width, _ = img.shape

                    # Prepare Heatmap Data
                    heatmap_data = np.zeros((height, width), dtype=np.float32)

                    # Convert float points to integer coordinates for indexing
                    for x, y in fixations:
                        # x_int = int(round((1-x) * width))
                        x_int = int(round((1-x) * width))
                        y_int = int(round(y * height))

                        # Check if coordinates are within the image boundaries
                        if 0 <= x_int < width and 0 <= y_int < height:
                            heatmap_data[y_int, x_int] += 1  

                    # Apply Gaussian Blur
                    heatmap_data = cv2.GaussianBlur(heatmap_data, (51, 51), 10)

                    # Normalize Heatmap
                    heatmap_data = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min())

                    # Apply Colormap
                    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap_data), cv2.COLORMAP_JET)

                    # Overlay Heatmap on Image
                    overlay = cv2.addWeighted(heatmap, 0.5, img, 0.5, 0)

                    # Display or Save
                    # plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
                    # plt.show()
                    
                    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

                if test_name == 'Vision Periferica':
                    backgorund_image = 'src/frontend/resources/laminas/Vision periferica.jpeg'
                    st.image(plot_heatmap(fixations_list, backgorund_image))
                elif test_name == 'Sacádicos':
                    backgorund_image = 'src/frontend/resources/laminas/Sacadicos.jpg'
                    st.image(plot_heatmap(fixations_list, backgorund_image))

            col_d1, col_d2 = st.columns(2)
            with col_d1:
                aim_score = db_conn.execute(
                        f"""
                            SELECT test_id, AVG(score) AS score
                            FROM main.aim_tracker as aim
                            GROUP BY test_id
                        """
                    ).df()['score'].values[0]
                st.metric(label="AIM",  value=round(aim_score, 1))



                st.subheader(" ")   
                st.subheader(" ")
                st.subheader("PUNTAJE TOTAL")

                to_sum = [
                    abs(yaw), abs(pitch), abs(yaw),
                    float(duration_timestamp.second),
                    float(blinks_ten_secs),
                    float(aim_score)

                ]

                def get_total_score_category(score:float) -> str:
                    if (score >= 0) & (score < 14):
                        return "Básico"
                    elif (score >= 14) & (score < 33):
                        return "Normal"
                    elif (score >= 33) & (score < 54):
                        return "Intermedio"
                    elif (score >= 54) & (score < 69):
                        return "Avanzado"
                    elif (score >= 69) & (score < 84):
                        return "Pro"
                    elif (score >= 84):
                        return "Elite"
                    else:
                        return "Score error"


                st.metric(
                    label=get_total_score_category(round(sum(to_sum), 0)), 
                    value=round(sum(to_sum), 0)
                )


if __name__ == "__main__":
    # Create input and output directories if they don't exist
    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    (
        input_video,
        options,
        selected_option,
        process_button,
        models_params,
    ) = configure_sidebar()
    main_page(input_video, options, selected_option, process_button, models_params)
    # main_page()
