import dataclasses
import datetime
import numpy
import typing
import supervision

@dataclasses.dataclass
class SvDetection:
    """Helper to access SV detection properties"""
    xyxy_coords : list[numpy.float32]
    confidence : float
    coco_class_id : int
    mask : typing.Any = None
    tracker_id : int | None = None
    data : dict[str, typing.Any] = dataclasses.field( default_factory=lambda: dict() )

    @staticmethod
    def from_sv_detection( supervision_detection_values : list ) -> "SvDetection":
        xyxy_coords, mask, confidence, coco_class_id, tracker_id, data = supervision_detection_values
        return SvDetection(
            xyxy_coords=[float(value) for value in xyxy_coords],
            mask=mask,
            confidence=float(confidence),
            coco_class_id=int(coco_class_id),
            tracker_id=None if tracker_id is None else int(tracker_id),
            data=data
        )
    
    @staticmethod
    def list_from_sv_detections( detections : supervision.Detections ) -> list["SvDetection"]:
        return [SvDetection.from_sv_detection(detection) for detection in detections]

_T = typing.TypeVar('T')

@dataclasses.dataclass
class Point2D(typing.Generic[_T]):
    x : _T
    y : _T

@dataclasses.dataclass
class ObjectDetectionInfo:
    cam_id : int
    supervision : SvDetection
    when : datetime.datetime
    frame_size : Point2D[int]

@dataclasses.dataclass
class IgnorePoint:
    coco_class_id : int
    at : Point2D[float] # 0.0-1.1 values
    cam_id : int