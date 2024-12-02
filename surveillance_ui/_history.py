import os
import os.path
import pathlib
import datetime
import numpy
import re
import pathlib
from .interface import (
    Configuration,
)
from ._common import (
    Point2D,
    SvDetection,
    ObjectDetectionInfo,
)
from .utility import (
    EventDispatcher,
)
import PIL.Image
import PIL.ImageQt

class DetectionHistory:
    _FOLDER = pathlib.Path("detections/")

    _configuration : Configuration

    _detection_list : list[ObjectDetectionInfo]
    added_dispatcher : EventDispatcher[ObjectDetectionInfo]
    removed_dispatcher : EventDispatcher[ObjectDetectionInfo]

    def __init__( self,
                  configuration : Configuration ):
        super().__init__()
        self._configuration = configuration

        self.added_dispatcher = EventDispatcher()
        self.removed_dispatcher = EventDispatcher()

        self._detection_list = []
        self._load_saved_detections()

    def add(self, detection : ObjectDetectionInfo, image : numpy.ndarray ) -> bool:
        self._FOLDER.mkdir( exist_ok=True )

        pil_image = PIL.Image.fromarray(image, "RGB")
        pil_image.save( self._FOLDER / self._detection_info_to_filename( detection ) )
        
        self._detection_list.append( detection )
        self.added_dispatcher.fire( detection )
        self._control_length()
        
        return True

    def is_fresh_detection(self, detection : ObjectDetectionInfo) -> bool:
        for existing_detection in self._detection_list:
            if ( detection.cam_id == existing_detection.cam_id
                 and
                 detection.supervision.coco_class_id == existing_detection.supervision.coco_class_id
                 and
                 detection.when <= existing_detection.when + self._configuration.redetection_delay ):
                return False
        return True
    
    def get_detections(self):
        return self._detection_list.copy()

    def _control_length(self):
        while len(self._detection_list) > self._configuration.max_history_entries:
            detection = self._detection_list.pop(0)
            self.removed_dispatcher.fire(detection)
            path = self._FOLDER / self._detection_info_to_filename( detection )
            path.unlink()

    def get_detection_image_data(self, detection : ObjectDetectionInfo ) -> numpy.ndarray:
        filename = self._detection_info_to_filename( detection )
        with PIL.Image.open( self._FOLDER / filename ) as pil_image:
            return numpy.array(pil_image)

    def _load_saved_detections(self) -> None:
        detections : list[ObjectDetectionInfo] = []
        try:
            for filename in os.listdir(self._FOLDER):
                path = self._FOLDER / filename
                if not os.path.isfile(path):
                    continue
                detection = self.detection_info_from_file( filename )
                if detection is None:
                    continue

                detections.append( detection )
        except FileNotFoundError:
            pass # the folder does not exist yet so continue with no saved detections
        
        detections.sort( key = lambda x: x.when )

        for detection in detections:
            self._detection_list.append( detection )
        self._control_length()            

    def detection_info_from_file( self, filename : str  ) -> "ObjectDetectionInfo | None":
        match : re.Match = re.fullmatch(
            r"(?P<datetime>\d+-\d+-\d+ \d+-\d+-\d+) coco(?P<coco_class_id>\d+) cam(?P<cam_id>\d+) rect(?P<rectangle_x1>\d+)-(?P<rectangle_y1>\d+)-(?P<rectangle_x2>\d+)-(?P<rectangle_y2>\d+) frame(?P<frame_x>\d+)-(?P<frame_y>\d+) conf(?P<conf>\d+).jpg",
            filename
        )
        if match is None:
            return None
        
        try:
            when = datetime.datetime.strptime( match.group("datetime"), r"%Y-%m-%d %H-%M-%S" )
        except ValueError:
            return None
        
        def parse_int(group_name):
            try:
                return int(match.group(group_name))
            except ValueError:
                return None
        def parse_float(group_name):
            try:
                return float(match.group(group_name))
            except ValueError:
                return None

        coco_class_id = parse_int("coco_class_id")
        cam_id = parse_int("cam_id")
        confidence_percentage = parse_int("conf")
        xyxy_coords = [parse_int("rectangle_x1"),parse_int("rectangle_y1"),parse_int("rectangle_x2"),parse_int("rectangle_y2")]
        frame_size = Point2D( x=parse_float("frame_x"), y=parse_float("frame_y"))
        if any( [item is None for item in [coco_class_id, cam_id, confidence_percentage] + xyxy_coords] ):
            return None

        sv_detection  = SvDetection( xyxy_coords=xyxy_coords, confidence=confidence_percentage/100, coco_class_id=coco_class_id )
        
        return ObjectDetectionInfo( cam_id=cam_id, supervision=sv_detection, when=when, frame_size=frame_size )

    @staticmethod
    def _detection_info_to_filename(detection : ObjectDetectionInfo) -> pathlib.Path:
        sv_detection = detection.supervision
        frame_size = detection.frame_size
        return pathlib.Path(f"{detection.when.strftime(r"%Y-%m-%d %H-%M-%S")} coco{sv_detection.coco_class_id} cam{detection.cam_id} rect{'-'.join([str(int(coord)) for coord in sv_detection.xyxy_coords])} frame{'-'.join([str(int(coord)) for coord in [frame_size.x, frame_size.y]])} conf{'%.0f' % (sv_detection.confidence*100)}.jpg")