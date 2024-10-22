import os
import os.path
import pathlib
import datetime
import numpy
import re
import pathlib
import typing
import statistics
import functools
from .common import (
    Configuration,
    Point2D,
    _SvDetection,
    _ObjectDetectionInfo,
    _IgnorePoint,
) 
from .utility import (
    FittingImage,
    ErrorHandler,
)
import PySide6.QtWidgets
import PySide6.QtGui
import PySide6.QtCore
import PIL.Image
import PIL.ImageQt

class SurveillanceHistoryView(PySide6.QtWidgets.QFrame):
    _FOLDER = pathlib.Path("detections/")

    _configuration : Configuration
    _error_handler : ErrorHandler
    _add_to_ignore : typing.Callable[[_IgnorePoint],None]

    _detection_list : list[_ObjectDetectionInfo]
    _detection_list_widget : PySide6.QtWidgets.QTreeWidget
    _detection_display : FittingImage

    def __init__( self,
                  configuration : Configuration,
                  error_handler : ErrorHandler,
                  add_to_ignore : typing.Callable[[_IgnorePoint],None] ):
        super().__init__()
        self._configuration = configuration
        self._error_handler = error_handler
        self._add_to_ignore = add_to_ignore

        self._detection_list = []
        detection_list_layout = PySide6.QtWidgets.QHBoxLayout()
        self.setLayout(detection_list_layout)
        self._detection_list_widget = PySide6.QtWidgets.QTreeWidget()
        self._detection_list_widget.setSizePolicy( PySide6.QtWidgets.QSizePolicy.Policy.Expanding, PySide6.QtWidgets.QSizePolicy.Policy.Expanding )
        self._detection_list_widget.setColumnCount( 6 )
        self._detection_list_widget.setHeaderHidden(  True )
        self._detection_list_widget.setSelectionMode( PySide6.QtWidgets.QListWidget.SelectionMode.SingleSelection )
        self._detection_list_widget.setSelectionBehavior( PySide6.QtWidgets.QListWidget.SelectionBehavior.SelectRows )
        detection_list_layout.addWidget( self._detection_list_widget )
        detection_list_layout.setStretch( 0, 100)
        self._detection_display = FittingImage( 50, 50, self._error_handler )
        detection_list_layout.addWidget( self._detection_display )
        detection_list_layout.setStretch( 1, 1)
        self._detection_list_widget.currentItemChanged.connect( lambda: self._on_current_item_change() )
        self._detection_list_widget.model().rowsInserted.connect( lambda: self._adjust_list_view_width() )       
        self._detection_list_widget.model().rowsRemoved.connect( lambda: self._adjust_list_view_width() )
        self._enumerate_saved_detections()

    def graceful_handler( handler ):
        @functools.wraps( handler )
        def wrapped_handler( self : 'SurveillanceHistoryView', *args, **kwargs ):
            self._error_handler.handle_gracefully( handler, "Internal error.", self, *args, **kwargs )
        return wrapped_handler

    def _append(self, detection : _ObjectDetectionInfo ):
        interest = self._configuration.get_interest( detection.supervision.coco_class_id )
        cam_definition = self._configuration.get_cam_definition( detection.cam_id )
        strings = [
            detection.when.strftime(r"%Y-%m-%d %H:%M:%S"),
            interest.label,
            cam_definition.label,
            str([int(coord) for coord in detection.supervision.xyxy_coords]),
            '%.0f %%' % (detection.supervision.confidence*100)
        ]
        self._detection_list.append( detection )
        item = PySide6.QtWidgets.QTreeWidgetItem( None, strings )
        self._detection_list_widget.addTopLevelItem(item)
        button = PySide6.QtWidgets.QPushButton(" 🚫 ")
        def ignore():
            xyxy_coords = detection.supervision.xyxy_coords
            x = float( statistics.mean( [xyxy_coords[0], xyxy_coords[2]] ) / detection.frame_size.x )
            y = float( statistics.mean( [xyxy_coords[1], xyxy_coords[3]] ) / detection.frame_size.y )
            self._add_to_ignore( _IgnorePoint( coco_class_id=interest.coco_class_id, at=Point2D(x,y), cam_id=cam_definition.id ) )
        button.pressed.connect( ignore )
        self._detection_list_widget.setItemWidget(item, 5, button)
        self._control_length()

    def persist_if_fresh(self, detection : _ObjectDetectionInfo, image : numpy.ndarray ) -> bool:
        if not self.is_fresh_detection( detection ):
            return False
        
        self._FOLDER.mkdir( exist_ok=True )

        pil_image = PIL.Image.fromarray(image, "RGB")
        pil_image.save( self._FOLDER / self.detection_info_to_filename( detection ) )
        self._append(detection)
        return True

    def is_fresh_detection(self, detection : _ObjectDetectionInfo) -> bool:
        for existing_detection in self._detection_list:
            if ( detection.cam_id == existing_detection.cam_id
                 and
                 detection.supervision.coco_class_id == existing_detection.supervision.coco_class_id
                 and
                 detection.when <= (existing_detection.when + datetime.timedelta(seconds=15)) ):
                return False
        return True

    def _control_length(self):
        if len(self._detection_list) > self._configuration.max_history_entries:
            detection = self._detection_list.pop(0)
            self._detection_list_widget.takeTopLevelItem(0)
            path = self._FOLDER / self.detection_info_to_filename( detection )
            path.unlink()
 
    @graceful_handler
    def _on_current_item_change(self) -> None:
        if self._detection_list_widget.currentIndex() is None:
            self._detection_display.clear()
            return
        
        row = self._detection_list_widget.currentIndex().row()
        self._detection_display.setPixmap( self._load_detection_image( self._detection_list[row] ) )
    
    def _load_detection_image(self, detection : _ObjectDetectionInfo ) -> PySide6.QtGui.QPixmap:
        filename = self.detection_info_to_filename( detection )
        with PIL.Image.open( self._FOLDER / filename ) as pil_image:
            pil_qt_image = PIL.ImageQt.ImageQt(pil_image)
            return PySide6.QtGui.QPixmap.fromImage( pil_qt_image )

    def _enumerate_saved_detections(self) -> None:
        detections : list[_ObjectDetectionInfo] = []
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
            self._append( detection )

    @graceful_handler
    def _adjust_list_view_width(self):
        for i in range( 0, self._detection_list_widget.columnCount()):
            self._detection_list_widget.resizeColumnToContents(i)
        
        total_width = sum([self._detection_list_widget.sizeHintForColumn(i) for i in range(0, self._detection_list_widget.columnCount())])
        self._detection_list_widget.setMaximumWidth( total_width + 50)

    def detection_info_from_file( self, filename : str  ) -> "_ObjectDetectionInfo | None":
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

        sv_detection  = _SvDetection( xyxy_coords=xyxy_coords, confidence=confidence_percentage/100, coco_class_id=coco_class_id )
        
        return _ObjectDetectionInfo( cam_id=cam_id, supervision=sv_detection, when=when, frame_size=frame_size )

    @staticmethod
    def detection_info_to_filename(detection : _ObjectDetectionInfo) -> pathlib.Path:
        sv_detection = detection.supervision
        frame_size = detection.frame_size
        return pathlib.Path(f"{detection.when.strftime(r"%Y-%m-%d %H-%M-%S")} coco{sv_detection.coco_class_id} cam{detection.cam_id} rect{'-'.join([str(int(coord)) for coord in sv_detection.xyxy_coords])} frame{'-'.join([str(int(coord)) for coord in [frame_size.x, frame_size.y]])} conf{'%.0f' % (sv_detection.confidence*100)}.jpg")