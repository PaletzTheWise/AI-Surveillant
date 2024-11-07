import dataclasses
import datetime
import numpy
import typing

class SupervisionDetections: # actually supervision.Detections but import of supervision takes time, so delaying it until after UI is shown
    
    def __init__(self):
        raise NotImplementedError()
    
    def __iter__(self):
        """Suppress errors about this not being iterable."""
        raise NotImplementedError()
    
    def __getitem__(self, parameter):
        """Suppress errors about this not having [] operator."""
        raise NotImplementedError()

@dataclasses.dataclass
class CamDefinition:
    url : str
    id : int
    label : str
    sound_alert_path : str

@dataclasses.dataclass
class Interest:
    coco_class_id : int
    label : str
    enabled_by_default : bool
    sound_alert_path : str

class DetectionLogic(typing.Protocol):
    def configure( self, coco_class_ids : list[int], confidence : float ) -> None:
        """
        Configure detection parameters.

        Parameters:
            coco_class_ids - list of coco class ids to detect.
            confidence - minimum confidence level of detection on the scale from 0.0 = none to 1.0 = absolute.
        """

    def detect( self, image : numpy.ndarray ) -> SupervisionDetections:
        """
        Detect objects.

        The results shall conform to settings provided by configure().
        
        Parameters:
           image - Image in QImage.Format.Format_RGB888, do not modify
        Returns detections.
        """

@dataclasses.dataclass
class Configuration:
    cam_definitions : list[CamDefinition]
    interests : list[Interest]
    max_history_entries : int
    detection_logic : DetectionLogic
    grid_column_count : int
    initial_confidence : float = 0.65
    minimum_detection_area : int = 1500
    redetection_delay : datetime.timedelta = datetime.timedelta( seconds=15 )
    camera_feed_timeout : datetime.timedelta = datetime.timedelta( seconds=3 )
    disconnect_indicator_additional_delay : datetime.timedelta = datetime.timedelta( seconds=2 ) # additional to camera_feed_timeout
    use_tcp_transport : bool = True
    max_delay : datetime.timedelta = datetime.timedelta( seconds=3 )

    def get_cam_definition( self, cam_id : int ) -> CamDefinition:
        for cam_definition in self.cam_definitions:
            if cam_definition.id == cam_id:
                return cam_definition
        raise ValueError("Unknown cam ID.")
    
    def get_interest( self, coco_class_id : int ) -> Interest:
        for interest in self.interests:
            if interest.coco_class_id == coco_class_id:
                return interest
        raise ValueError("Unknown coco class ID.")
    
    def get_disconnect_indicator_delay(self) -> datetime.timedelta:
        '''
        Get disconnect indicator delay.

        When camera feed times out, it may produce a partial frame. Disconnector indicator should thus not engage prior to that otherwise
        the partial frame could refresh disconnect indicator logic and make the UI look hesitant.
        '''
        return self.camera_feed_timeout + self.disconnect_indicator_additional_delay

@dataclasses.dataclass
class _FrameInfo:
    image : numpy.ndarray
    cam_id : int

@dataclasses.dataclass
class _AudioChunk:
    chunk : bytes
    cam_id : int

@dataclasses.dataclass
class _SvDetection:
    """Helper to access SV detection properties"""
    xyxy_coords : list[numpy.float32]
    confidence : float
    coco_class_id : int
    mask : typing.Any = None
    tracker_id : int | None = None
    data : dict[str, typing.Any] = dataclasses.field( default_factory=lambda: dict() )

    @staticmethod
    def from_sv_detection( supervision_detection_values : list ) -> "_SvDetection":
        xyxy_coords, mask, confidence, coco_class_id, tracker_id, data = supervision_detection_values
        return _SvDetection(
            xyxy_coords=[float(value) for value in xyxy_coords],
            mask=mask,
            confidence=float(confidence),
            coco_class_id=int(coco_class_id),
            tracker_id=None if tracker_id is None else int(tracker_id),
            data=data
        )
    
    @staticmethod
    def list_from_sv_detections( detections : SupervisionDetections ) -> list["_SvDetection"]:
        return [_SvDetection.from_sv_detection(detection) for detection in detections]

@dataclasses.dataclass
class _ImageDetectionsInfo:
    frame_info : _FrameInfo
    detections : list[_SvDetection]
    when : datetime.datetime

_T = typing.TypeVar('T')

@dataclasses.dataclass
class Point2D(typing.Generic[_T]):
    x : _T
    y : _T

@dataclasses.dataclass
class _ObjectDetectionInfo:
    cam_id : int
    supervision : _SvDetection
    when : datetime.datetime
    frame_size : Point2D[int]

@dataclasses.dataclass
class _IgnorePoint:
    coco_class_id : int
    at : Point2D[float] # 0.0-1.1 values
    cam_id : int