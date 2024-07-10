from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Athlete:
    id: str
    name: str
    gender: str = field(init=False, default="")
    birth_date: str = field(init=False, default="")
    side: str = field(init=False, default="")
    height: float = field(init=False, default=-1.1)
    weight: float = field(init=False, default=-1.1)
    discipline: str = field(init=False, default="")
    discipline_category: str = field(init=False, default="")
    main_position: str = field(init=False, default="")
    second_position: str = field(init=False, default="")
    club: str = field(init=False, default="")


@dataclass
class Test:
    id: str
    athlete_id: str
    category: str
    test_name: str
    variation: str
    test_date: datetime
    test_time: datetime
    upload_datetime: datetime
    duration: datetime


@dataclass
class Blinks:
    test_id: str
    id: str
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
class GazePositions:
    test_id: str
    gaze_timestamp: datetime
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
class PupilsPositions:
    test_id: str
    pupil_timestamp: datetime
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
class Fixations:
    test_id: str
    id: str
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
class FixationsOnSurfaces:
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
class GazePositionsOnSurfaces:
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
class HeadPosesTracker:
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


@dataclass
class AimTracker:
    test_id: str
    score: float