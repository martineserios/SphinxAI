# load packages
import hashlib
from dataclasses import astuple
from datetime import datetime
from io import StringIO

import duckdb as ddb
import pandas as pd
from polars import duration

from sphinx_ai.pupils.pupil_classes import AimTracker, Athlete, Test
from sphinx_ai.utilities import ProgressBar


def generate_id(input_string):
    # Convert input string to bytes
    input_bytes = input_string.encode("utf-8")

    # Generate hash value using SHA-256 hash function
    hash_object = hashlib.sha256()
    hash_object.update(input_bytes)
    hash_value = hash_object.hexdigest()

    # Return the first 8 characters of the hash value as the ID
    return hash_value[:8]


def get_test_player_data(data_file) -> dict:
    pupil_data = pd.read_csv(
        StringIO(data_file.getvalue().decode("utf-8")),
        on_bad_lines="skip",
    )

    # relevant_keys = [
    #     "Start Date",
    #     "Start Time (System)",
    #     "Start Time (Synced)",
    #     "Recording UUID",
    # ]
    
    # logger.info("pupil_data_df: ", pupil_data     )
    # pupil_data = pupil_data.set_index("key")
    # logger.info(pupil_data)
    pupil_data = pupil_data.set_index("key")['value']
    pupil_data_dict = pupil_data.to_dict()

    return pupil_data_dict

def get_pupils_test_player_data_keys(data_file_path:str, keys: list, date_format={}):
    test_player_data = get_test_player_data(data_file_path)

    if date_format != {}:
        for key, date_format in date_format.items():
            test_player_data[key] = datetime.strptime(test_player_data[key], str(date_format))
            

    return test_player_data    


def transform_unix_to_datetime(unix_time: float, data_file):
    test_player_data = get_pupils_test_player_data_keys(data_file, keys=["Start Time (System)", "Start Time (Synced)"])
    offset = float(test_player_data["Start Time (System)"])\
             - float(test_player_data["Start Time (Synced)"])
    wall_time = datetime.fromtimestamp(offset + unix_time).strftime(
        "%Y-%m-%d %H:%M:%S.%f"
    )
    return wall_time


# def get_test_duration(data_file_path:str, keys: list, date_format={}):
#     test_player_data = get_test_player_data(data_file_path)
#     duration_time = test_player_data['Duration Time']
#     logger.info(duration_time)
#     return duration_time.strftime("%H:%M:%S.%f")




# def get_pupils_recording_id(data_file_path: str):
#     get_test_player_data(data_file_path)["Recording UUID"]

# def get_pupils_start_date(data_file: str):
#     start_date = get_test_player_data(data_file)["Start Date"]
#     return datetime.strptime(start_date, "%d.%m.%y")


# def get_pupils_start_time(data_file: str):
#     start_time = get_test_player_data(data_file)["Start Time"]
#     return datetime.strptime(start_time, "%H:%M:%S")




# def get_test_date(data_file: str):
#     return get_test_player_data(data_file)["Start Date"]


async def upload_data(
    pupil_files,
    db_conn,
    athlete_name,
    # gender,
    # discipline,
    # main_position,
    # birth_date,
    category,
    test_name,
    test_variation,
    aim_score
):
    data_file = [file for file in pupil_files if file.name == "iMotions_info.csv"][0]
    pupil_files = [file for file in pupil_files if file.name != "iMotions_info.csv"]

    progress_bar = ProgressBar(pupil_files)


    # athlete_data = {key:None for key, field in Athlete.__dataclass_fields__.items() if field.default == ''}

    athlete_id = generate_id(athlete_name)
    test_player_data = get_pupils_test_player_data_keys(data_file,
                                     keys=["Recording UUID","Start Time", "Start Date", "Duration Time"],
                                     date_format={
                                         "Start Time": "%H:%M:%S",
                                         "Start Date": "%d.%m.%Y",
                                         "Duration Time": "%H:%M:%S",
                                    })
    test_id = test_player_data['Recording UUID']
    start_date = test_player_data['Start Date'].strftime("%Y-%m-%d %H:%M:%S")
    start_time = test_player_data['Start Time'].strftime("%Y-%m-%d %H:%M:%S")
    duration = test_player_data['Start Time'].strftime("%Y-%m-%d %H:%M:%S")

    # test_id, start_time, start_date = tuple(get_pupils_test_player_data_keys(data_file,
    #                                  keys=["Recording UUID","Start Time", "Start Date"],
    #                                  date_format={
    #                                      "Start Time": "%H:%M:%S",
    #                                      "Start Date": "%d.%m.%y",
    #                                 }).iloc[0,:].values)
    # test_id = get_pupils_recording_id(data_file)
    # start_time = get_pupils_start_time(data_file)
    # start_date = get_pupils_start_date(data_file)

    athlete = Athlete(
        id=athlete_id,
        name=athlete_name,
    )

    test = Test(
        id=test_id,
        athlete_id=athlete_id,
        category=category,
        test_name=test_name,
        variation=test_variation,
        test_date=start_date,
        test_time=start_time,
        upload_datetime=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        duration=duration
    )

    aim = AimTracker(
        test_id=test_id,
        score=aim_score
    )

    logger.info(test_id)


    logger.info(athlete)
    logger.info(test_id)

    db_conn.sql(
        f"""
        INSERT OR REPLACE INTO athlets VALUES {astuple(athlete)}
    """
    )
    logger.info(test)

    db_conn.sql(
        f"""
        INSERT OR REPLACE INTO tests VALUES {astuple(test)}
    """
    )

    db_conn.sql(
        f"""
        INSERT OR REPLACE INTO aim_tracker VALUES {astuple(aim)}
    """
    )


    dfs = []

    for file in pupil_files:
        logger.info(file.name)
        progress_bar.update()
        df = pd.read_csv(file, on_bad_lines="skip")
        new_df = df.copy()

        for col_label, col_values in df.items():
            if str(col_label).find("timestamp") != -1:
                logger.info(col_label)
                df[col_label] = new_df[col_label].apply(
                    lambda x: transform_unix_to_datetime(x, data_file)
                )
                logger.info("transform_unix")

        logger.info("paso transform")

        if "on_surface" in file.name:
            surface = file.name.split(".")[0].split(" ")[-1]
            logger.info(file.name, "surface: ", surface)

            table_name = file.name.split("_Surface")[0] + "s"
            logger.info(table_name)

            df.insert(0, "surface", surface)

        elif "gaze_positions" in file.name:
            table_name = file.name.split(".")[0]
            logger.info(table_name)

        elif "pose_tracker" in file.name:
            table_name = "head_poses_tracker"#file.name.split("_poses")[0]
            logger.info(table_name)

        elif "pupil_positions" in file.name:
            table_name = "pupils_positions"
            logger.info(table_name)

        else:
            table_name = file.name.split(".")[0]
            logger.info(table_name)

        df.insert(0, "test_id", test_id)
        dfs.append((table_name, df))

        ddb.register(f"{table_name}_view", df)
        insert_query = f"""
            INSERT OR REPLACE INTO {table_name} SELECT * FROM df        
        """
        logger.info(insert_query)
        logger.info(df.shape)
        db_conn.execute(insert_query)

    return True
