import pathlib
import pathlib
import json
import supervision
import typing
from .interface import (
    Configuration,
    CamDefinition,
)
from ._common import (
    Point2D,
    SvDetection,
    IgnorePoint,
) 
from .utility import (
    EventDispatcher,
)
from .synchronized import (
    Synchronized,
)
import PySide6.QtWidgets
import PySide6.QtGui
import PySide6.QtCore

class IgnoreList(PySide6.QtWidgets.QFrame):
    _IGNORE_FILE = pathlib.Path("ignore_list.json")
    _IGNORE_FILE_NEW = pathlib.Path("ignore_list.new.json")

    _configuration : Configuration

    _synchronized_ignore_list : Synchronized[list[IgnorePoint]]
    _added_dispatcher : EventDispatcher[IgnorePoint]
    _removed_dispatcher : EventDispatcher[IgnorePoint]

    def __init__( self,
                  configuration : Configuration ):
        super().__init__()
        self._configuration = configuration

        self._synchronized_ignore_list = Synchronized( list() )
        self._added_dispatcher = EventDispatcher()
        self._removed_dispatcher = EventDispatcher()

        with self._synchronized_ignore_list.lock() as ignore_list:
            for ignore_point in self._load_ignore_list():
                ignore_list.append( ignore_point )

    def get_ignore_points(self):
        with self._synchronized_ignore_list.lock() as ignore_list:
            return ignore_list.copy()
    
    def add(self, ignore_point : IgnorePoint ):
        with self._synchronized_ignore_list.lock() as ignore_list:
            ignore_list.append( ignore_point )
        self._added_dispatcher.fire(ignore_point)
        self._save_ignore_list()
    
    def remove(self, ignore_point : IgnorePoint ):
        with self._synchronized_ignore_list.lock() as ignore_list:
            ignore_list.remove( ignore_point )
        self._removed_dispatcher.fire(ignore_point)
        self._save_ignore_list()

    def filter_ignored( self, detections : supervision.Detections, cam_definition : CamDefinition, frame_size : Point2D[int] ) -> supervision.Detections:
        """
        Filter detections to remove ignored detections.
        
        Thread-safe.
        """
        valid_detection_indices : list[int] = []
        for index, detection in enumerate( SvDetection.list_from_sv_detections( detections ) ):
            xyxy_coords=detection.xyxy_coords
            area = float((xyxy_coords[2]-xyxy_coords[0])*(xyxy_coords[3]-xyxy_coords[1]))
            if ( area < self._configuration.minimum_detection_area
                 or
                 self._is_ignored( detection.interest_id, cam_definition, xyxy_coords=xyxy_coords, frame_size=frame_size) ):
                continue
            valid_detection_indices.append( index )
        
        return detections[valid_detection_indices]

    def _is_ignored( self, interest_id : int, cam_definition : CamDefinition, xyxy_coords : list[float], frame_size : Point2D[int] ) -> bool:
        """
        Tell whether a detection should be ignored.

        Thread-safe.
        """
        min_x, min_y, max_x, max_y = [float(value) for value in xyxy_coords]
        min_x /= frame_size.x
        max_x /= frame_size.x
        min_y /= frame_size.y
        max_y /= frame_size.y

        with self._synchronized_ignore_list.lock() as ignore_list:
            for ignore_point in ignore_list:
                if ( ignore_point.interest_id == interest_id 
                    and
                    ignore_point.cam_id == cam_definition.id
                    and
                    min_x <= ignore_point.at.x and ignore_point.at.x <= max_x
                    and
                    min_y <= ignore_point.at.y and ignore_point.at.y <= max_y ):
                    return True
        
        return False

    def _load_ignore_list(self) -> list[IgnorePoint]:
        ignore_points : list[IgnorePoint] = []
        try:
            with open( self._IGNORE_FILE, "r") as file:
                for item in json.load( file ):
                    item = typing.cast( dict, item)
                    ignore_points.append(
                        IgnorePoint(
                            interest_id = int(item.get( "interest_id", None) or item.get("coco_class_id", None)),
                            at = Point2D( float(item["x"]), float(item["y"]) ),
                            cam_id = int(item["cam_id"])
                        )
                    )
        except OSError:
            pass # just start with empty list
        
        return ignore_points
    
    def _ignore_point_to_dict( self, ignore_point : IgnorePoint ) -> dict:
        return {
            "interest_id" : ignore_point.interest_id,
            "x" : ignore_point.at.x,
            "y" : ignore_point.at.y,
            "cam_id" : ignore_point.cam_id
        }

    def _save_ignore_list(self):
        with open( self._IGNORE_FILE_NEW, "w") as file, self._synchronized_ignore_list.lock() as ignore_list:
            json.dump( [self._ignore_point_to_dict( item ) for item in ignore_list] , file )
        try:
            self._IGNORE_FILE.unlink()
        except FileNotFoundError:
            pass
        self._IGNORE_FILE_NEW.rename( self._IGNORE_FILE )