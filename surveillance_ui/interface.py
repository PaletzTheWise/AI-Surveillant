import typing
import dataclasses
import datetime

if typing.TYPE_CHECKING:
    import numpy
    import supervision

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

    def detect( self, image : 'numpy.ndarray' ) -> 'supervision.Detections':
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
