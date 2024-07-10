# load packages
from dataclasses import dataclass
from datetime import date, datetime


# BUILDER
@dataclass
class User:
    id: str
    name: str
    sport: str
    position: str
    birth_date: str


@dataclass
class Test:
    id: str
    user_id: str
    test_name: str
    variation: str
    # test_date: date


@dataclass
class Blink:
    id: str
    test_id: str
    start_timestamp: datetime
    duration: float
    end_timestamp: datetime
    start_frame_index: int
    index: int
    end_frame_index: int
    confidence: float
    filter_response: str
    base_data: str


@dataclass
class GazePosition:
    gaze_timestamp: datetime
    test_id: str
    world_index: int
    confidence: float
    norm_pos_x: float
    norm_pos_y: float
    base_data: str
    gaze_point_3d_x: float
    gaze_point_3d_y: float
    gaze_point_3d_z: float
    eye_center0_3d_x: float
    eye_center0_3d_y: float
    eye_center0_3d_z: float
    gaze_normal0_x: float
    gaze_normal0_y: float
    gaze_normal0_z: float
    eye_center1_3d_x: float
    eye_center1_3d_y: float
    eye_center1_3d_z: float
    gaze_normal1_x: float
    gaze_normal1_y: float
    gaze_normal1_z: float


@dataclass
class PupilPosition:
    pupil_timestamp: datetime
    test_id: str
    world_index: int
    eye_id: int
    confidence: float
    norm_pos_x: float
    norm_pos_y: float
    diameter: float
    method: str
    ellipse_center_x: float
    ellipse_center_y: float
    ellipse_axis_a: float
    ellipse_axis_b: float
    ellipse_angle: float
    diameter_3d: float
    model_confidence: float
    model_id: float
    sphere_center_x: float
    sphere_center_y: float
    sphere_center_z: float
    sphere_radius: float
    circle_3d_center_x: float
    circle_3d_center_y: float
    circle_3d_center_z: float
    circle_3d_normal_x: float
    circle_3d_normal_y: float
    circle_3d_normal_z: float
    circle_3d_radius: float
    theta: float
    phi: float
    projected_sphere_center_x: float
    projected_sphere_center_y: float
    projected_sphere_axis_a: float
    projected_sphere_axis_b: float
    projected_sphere_angle: float


@dataclass
class Fixation:
    id: str
    test_id: str
    start_timestamp: datetime
    duration: float
    start_frame_index: int
    end_frame_index: int
    norm_pos_x: float
    norm_pos_y: float
    dispersion: float
    confidence: float
    method: str
    gaze_point_3d_x: float
    gaze_point_3d_y: float
    gaze_point_3d_z: float
    base_data: str


@dataclass
class FixationOnSurface:
    test_id: str
    surface: str
    world_timestamp: datetime
    world_index: int
    fixation_id: int
    start_timestamp: datetime
    duration: float
    dispersion: float
    norm_pos_x: float
    norm_pos_y: float
    x_scaled: float
    y_scaled: float
    on_surf: bool


@dataclass
class GazePositionOnSurface:
    test_id: str
    surface: str
    world_timestamp: datetime
    world_index: float
    gaze_timestamp: datetime
    x_norm: float
    y_norm: float
    x_scaled: float
    y_scaled: float
    on_surf: float
    confidence: float


@dataclass
class HeadPoseTracker:
    test_id: str
    timestamp: datetime
    rotation_x: float
    rotation_y: float
    rotation_z: float
    translation_x: float
    translation_y: float
    translation_z: float
    pitch: float
    yaw: float
    roll: float


from dataclasses import dataclass, fields
from datetime import date, datetime


def generate_table_creation_sql() -> str:
    table_class_name_map = {
        "User": "users",
        "Test": "tests",
        "Blink": "blinks",
        "GazePosition": "gaze_positions",
        "PupilPosition": "pupil_positions",
        "Fixation": "fixations",
        "FixationOnSurface": "fixations_on_surface",
        "GazePositionOnSurface": "gaze_positions_on_surface",
        "HeadPoseTracker": "head_pose_tracker",
    }

    data_classes = [
        User,
        Test,
        Blink,
        GazePosition,
        PupilPosition,
        Fixation,
        FixationOnSurface,
        GazePositionOnSurface,
        HeadPoseTracker,
    ]

    sql_script = ""

    for data_class in data_classes:
        table_name = table_class_name_map[data_class.__name__]
        # Get the name of the dataclass
        class_name = data_class.__name__

        # Get the fields of the dataclass
        data_class_fields = fields(data_class)

        # Initialize SQL statement
        sql_statement = f"CREATE TABLE {table_name} ("

        # Loop through fields
        for field in data_class_fields:
            field_name = field.name
            field_type = field.type

            # Determine SQL type based on Python type
            sql_type = "VARCHAR"
            if field_type == str:
                sql_type = "VARCHAR"
            elif field_type == int:
                sql_type = "INTEGER"
            elif field_type == float:
                sql_type = "FLOAT"
            elif field_type == date:
                sql_type = "DATE"
            elif field_type == datetime:
                sql_type = "TIMESTAMP"

            # Add field to SQL statement
            sql_statement += f"{field_name} {sql_type}, "

        # Define primary key
        logger.info(table_name)
        if table_name == "pupil_positions":
            sql_statement = f"{sql_statement[:-2]})"
        elif table_name == "head_pose_tracker":
            sql_statement += "PRIMARY KEY (test_id, timestamp))"
        elif table_name == "gaze_positions":
            sql_statement += "PRIMARY KEY (test_id, gaze_timestamp))"
        elif "fixations_on_surface" in table_name:
            sql_statement = f"{sql_statement[:-2]})"
        elif "gaze_positions_on_surface" in table_name:
            sql_statement = f"{sql_statement[:-2]})"
        elif (table_name == "fixations") | (table_name == "blinks"):
            sql_statement += "PRIMARY KEY (id, test_id, start_timestamp))"
        elif table_name == "users":
            sql_statement += "PRIMARY KEY (id))"
        elif table_name == "tests":
            sql_statement += "PRIMARY KEY (id, user_id))"

        sql_script += f"{sql_statement};\n"

    return sql_script
