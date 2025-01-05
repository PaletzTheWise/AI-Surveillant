import os
import os.path
import pathlib
import datetime
import numpy
import re
import pathlib
import dataclasses
import uuid
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

@dataclasses.dataclass
class DetectionUpdate:
    old : ObjectDetectionInfo
    new : ObjectDetectionInfo

@dataclasses.dataclass
class _UpdatableObjectDetectionInfo(ObjectDetectionInfo):
    original_time : datetime.datetime = dataclasses.field( default_factory=lambda: datetime.datetime.min ) # default is required because the parent class has a default value field

    def __init__( self, detection : ObjectDetectionInfo, original_time : datetime.datetime ):
        super().__init__( *[getattr(detection,field.name) for field in dataclasses.fields(detection)] )
        self.original_time = original_time

class DetectionHistory:
    _FOLDER = pathlib.Path("detections/")

    _configuration : Configuration

    _detection_list : list[_UpdatableObjectDetectionInfo]
    added_dispatcher : EventDispatcher[ObjectDetectionInfo]
    removed_dispatcher : EventDispatcher[ObjectDetectionInfo]
    updated_dispatcher : EventDispatcher[DetectionUpdate]

    def __init__( self,
                  configuration : Configuration ):
        super().__init__()
        self._configuration = configuration

        self.added_dispatcher = EventDispatcher()
        self.removed_dispatcher = EventDispatcher()
        self.updated_dispatcher = EventDispatcher()

        self._detection_list = []
        self._load_saved_detections()

    def add(self, detection : ObjectDetectionInfo, image : numpy.ndarray, update_of : _UpdatableObjectDetectionInfo = None ) -> bool:
        self._FOLDER.mkdir( exist_ok=True )

        pil_image = PIL.Image.fromarray(image, "RGB")
        pil_image.save( self._FOLDER / self._detection_info_to_filename( detection ) )
        
        if update_of is not None:
            old_path = self._FOLDER / self._detection_info_to_filename( update_of )
            old_path.unlink()
            updatable_detection = _UpdatableObjectDetectionInfo(detection, original_time=update_of.original_time)
            self._detection_list[self._detection_list.index(update_of)] = updatable_detection
            self.updated_dispatcher.fire( DetectionUpdate( old=update_of, new=updatable_detection ) )
        else:
            updatable_detection = _UpdatableObjectDetectionInfo( detection, datetime.datetime.now() )
            self._detection_list.append( updatable_detection )
            self.added_dispatcher.fire( updatable_detection )
        
        self._control_length()
        
        return True

    def process_detection(self, detection : ObjectDetectionInfo, image : numpy.ndarray) -> bool:
        for existing_detection in self._detection_list:
            if ( detection.cam_id == existing_detection.cam_id
                 and
                 detection.supervision.interest_id == existing_detection.supervision.interest_id
                 and
                 detection.when <= existing_detection.original_time + self._configuration.redetection_delay ):
                if existing_detection.supervision.confidence < detection.supervision.confidence:
                    self.add( detection=detection, image=image, update_of=existing_detection )
                return False
        self.add( detection, image )
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
            self._detection_list.append( _UpdatableObjectDetectionInfo( detection=detection, original_time=datetime.datetime.min ) )
        self._control_length()            

    def detection_info_from_file( self, filename : str  ) -> "ObjectDetectionInfo | None":
        match : re.Match = re.fullmatch(
            r"(?P<datetime>\d+-\d+-\d+ \d+-\d+-\d+) interest(?P<interest_id>\d+) cam(?P<cam_id>\d+) rect(?P<rectangle_x1>\d+)-(?P<rectangle_y1>\d+)-(?P<rectangle_x2>\d+)-(?P<rectangle_y2>\d+) conf(?P<conf>\d+) (?P<guid>[0-9a-f-]+).jpg",
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

        interest_id = parse_int("interest_id")
        cam_id = parse_int("cam_id")
        confidence_percentage = parse_int("conf")
        xyxy_coords = [parse_int("rectangle_x1"),parse_int("rectangle_y1"),parse_int("rectangle_x2"),parse_int("rectangle_y2")]
      
        if any( [item is None for item in [interest_id, cam_id, confidence_percentage] + xyxy_coords] ):
            return None

        with PIL.Image.open( self._FOLDER / filename ) as pil_image:
            frame_size = Point2D( x=pil_image.size[0], y=pil_image.size[1] )

        sv_detection = SvDetection( xyxy_coords=xyxy_coords, confidence=confidence_percentage/100, interest_id=interest_id )

        try:
            guid = uuid.UUID(hex=match.group("guid"))
        except ValueError:
            guid = None

        return ObjectDetectionInfo( cam_id=cam_id, supervision=sv_detection, when=when, frame_size=frame_size, guid=guid )

    @staticmethod
    def _detection_info_to_filename(detection : ObjectDetectionInfo) -> pathlib.Path:
        sv_detection = detection.supervision
        return pathlib.Path(f"{detection.when.strftime(r"%Y-%m-%d %H-%M-%S")} interest{sv_detection.interest_id} cam{detection.cam_id} rect{'-'.join([str(int(coord)) for coord in sv_detection.xyxy_coords])} conf{'%.0f' % (sv_detection.confidence*100)} {detection.guid}.jpg")