import typing
import dataclasses
import datetime
import PySide6.QtCore

if typing.TYPE_CHECKING:
    import numpy
    import supervision

@dataclasses.dataclass
class CamDefinition:
    url : str
    id : int
    label : str
    sound_alert_path : str
    discard_corrupted_frames : bool = False

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
    """
    params:
        grid_column_count - Grid column count in the overview tab used for automatic layout and even for manual layout through grid_widgets if it exceeds the implied column count.
        grid_widget_locs - Camera widget locations in the overview tab square grid, if none, cameras will be laid out automatically.
                           This is useful to arrange cameras with different aspect ratios.
                           The location count must match camera count (alerts widgets will repeat the same locations underneath cam widgets) 
                           or double the camera count (alert widgets will use the second half of the locations).
                           Top left corner square is (0,0). The first component is horizontal, the second component is vertical.
    """
    cam_definitions : list[CamDefinition]
    interests : list[Interest]
    max_history_entries : int
    detection_logic : DetectionLogic
    grid_column_count : int = 2 
    grid_widget_locs : list[PySide6.QtCore.QRect] | None = None
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

    def is_defined_cam( self, cam_id : int ) -> bool:
        return any( [cd.id == cam_id for cd in self.cam_definitions] )
    
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
