import pathlib
import pathlib
import json
import functools
import typing
from .common import (
    Configuration,
    SupervisionDetections,
    CamDefinition,
    Point2D,
    _SvDetection,
    _ObjectDetectionInfo,
    _ImageDetectionsInfo,
    _IgnorePoint,
) 
from .utility import (
    FittingImage,
    Synchronized,
)
import PySide6.QtWidgets
import PySide6.QtGui
import PySide6.QtCore

class IgnoreListView(PySide6.QtWidgets.QFrame):
    _IGNORE_FILE = pathlib.Path("ignore_list.json")
    _IGNORE_FILE_NEW = pathlib.Path("ignore_list.new.json")
    _PREVIEW_SIZE = Point2D( 640, 360 )

    _configuration : Configuration
    _get_cam_image : typing.Callable[[int], PySide6.QtGui.QPixmap]

    _synchronized_ignore_list : Synchronized[list[_IgnorePoint]]
    _ignore_list_widget : PySide6.QtWidgets.QTreeWidget
    _ignore_item_display : FittingImage

    def __init__(self, configuration : Configuration, get_cam_image : typing.Callable[[int], PySide6.QtGui.QPixmap]):
        super().__init__()
        self._configuration = configuration
        self._get_cam_image = get_cam_image

        self._synchronized_ignore_list = Synchronized( list() )
        ignore_list_layout = PySide6.QtWidgets.QHBoxLayout()
        self.setLayout(ignore_list_layout)
        self._ignore_list_widget = PySide6.QtWidgets.QTreeWidget()
        self._ignore_list_widget.setSizePolicy( PySide6.QtWidgets.QSizePolicy.Policy.Expanding, PySide6.QtWidgets.QSizePolicy.Policy.Expanding )
        self._ignore_list_widget.setColumnCount( 4 )
        self._ignore_list_widget.setHeaderHidden(  True )
        self._ignore_list_widget.setSelectionMode( PySide6.QtWidgets.QListWidget.SelectionMode.SingleSelection )
        self._ignore_list_widget.setSelectionBehavior( PySide6.QtWidgets.QListWidget.SelectionBehavior.SelectRows )
        ignore_list_layout.addWidget( self._ignore_list_widget )
        ignore_list_layout.setStretch( 0, 100)
        self._ignore_item_display = FittingImage( 50, 50 )
        ignore_list_layout.addWidget( self._ignore_item_display )
        ignore_list_layout.setStretch( 1, 1)
        self._ignore_list_widget.currentItemChanged.connect( self._on_current_item_change )
        self._ignore_list_widget.model().rowsInserted.connect( self._adjust_list_view_width )       
        self._ignore_list_widget.model().rowsRemoved.connect( self._adjust_list_view_width )

        for ignore_point in self._load_ignore_list():
            self._append_without_saving( ignore_point )

    def append(self, ignore_point : _IgnorePoint ):
        self._append_without_saving(ignore_point)
        self._save_ignore_list()
    
    def _append_without_saving(self, ignore_point : _IgnorePoint ):
        strings = [
            self._configuration.get_cam_definition( ignore_point.cam_id ).label,
            self._configuration.get_interest( ignore_point.coco_class_id ).label,
            str( [ignore_point.at.x, ignore_point.at.y] )
        ]
        with self._synchronized_ignore_list.lock() as ignore_list:
            ignore_list.append( ignore_point )
        item = PySide6.QtWidgets.QTreeWidgetItem( None, strings )
        self._ignore_list_widget.addTopLevelItem(item)

        button = PySide6.QtWidgets.QPushButton(" ✖ ")
        def remove():
            with self._synchronized_ignore_list.lock() as ignore_list:
                index = ignore_list.index( ignore_point )
                del ignore_list[index]
                self._ignore_list_widget.takeTopLevelItem( index )
                self._save_ignore_list()
        
        button.pressed.connect( remove )
        self._ignore_list_widget.setItemWidget(item, 3, button)

    
    def filter_ignored( self, detections : SupervisionDetections, cam_definition : CamDefinition, frame_size : Point2D[int] ) -> SupervisionDetections:
        """
        Filter detections to remove ignored detections.
        
        Thread-safe.
        """
        valid_detection_indices : list[int] = []
        for index, detection in enumerate( _SvDetection.list_from_sv_detections( detections ) ):
            xyxy_coords=detection.xyxy_coords
            area = float((xyxy_coords[2]-xyxy_coords[0])*(xyxy_coords[3]-xyxy_coords[1]))
            if ( area < self._configuration.minimum_detection_area
                 or
                 self._is_ignored( detection.coco_class_id, cam_definition, xyxy_coords=xyxy_coords, frame_size=frame_size) ):
                continue
            valid_detection_indices.append( index )
        
        return detections[valid_detection_indices]

    def _is_ignored( self, coco_class_id : int, cam_definition : CamDefinition, xyxy_coords : list[float], frame_size : Point2D[int] ) -> bool:
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
                if ( ignore_point.coco_class_id == coco_class_id 
                    and
                    ignore_point.cam_id == cam_definition.id
                    and
                    min_x <= ignore_point.at.x and ignore_point.at.x <= max_x
                    and
                    min_y <= ignore_point.at.y and ignore_point.at.y <= max_y ):
                    return True
        
        return False

    def _on_current_item_change(self) -> None:
        if self._ignore_list_widget.currentIndex() is None:
            self._ignore_item_display.clear()
            return
        
        row = self._ignore_list_widget.currentIndex().row()
        with self._synchronized_ignore_list.lock() as ignore_list:
            item = ignore_list[row]
        image = self._get_cam_image( item.cam_id )
        if image.isNull():
            image = PySide6.QtGui.QPixmap( self._PREVIEW_SIZE.x , self._PREVIEW_SIZE.y )
            image.fill( PySide6.QtGui.QColorConstants.Gray )
        with PySide6.QtGui.QPainter( image ) as painter:
            scale = min( image.size().width(), image.size().height() ) / min( self._PREVIEW_SIZE.x, self._PREVIEW_SIZE.y )
            scale = max( 1, scale )
            def set_color(color):
                painter.setPen( PySide6.QtGui.QPen( color, scale ) )    
            at_screen_point = PySide6.QtCore.QPoint( int(item.at.x * image.size().width()), int(item.at.y * image.size().height()) )
            set_color( PySide6.QtGui.QColorConstants.White )
            painter.drawEllipse( at_screen_point, 3*scale, 3*scale )
            set_color( PySide6.QtGui.QColorConstants.Red )
            painter.drawEllipse( at_screen_point, 4*scale, 4*scale )
            set_color( PySide6.QtGui.QColorConstants.White )
            painter.drawEllipse( at_screen_point, 5*scale, 5*scale )
        self._ignore_item_display.setPixmap( image )
    
    def _adjust_list_view_width(self):
        for i in range( 0, self._ignore_list_widget.columnCount()):
            self._ignore_list_widget.resizeColumnToContents(i)
        
        total_width = sum([self._ignore_list_widget.sizeHintForColumn(i) for i in range(0, self._ignore_list_widget.columnCount())])
        self._ignore_list_widget.setMaximumWidth( total_width + 50)

    def _load_ignore_list(self) -> list[_IgnorePoint]:
        ignore_points : list[_IgnorePoint] = []
        try:
            with open( self._IGNORE_FILE, "r") as file:
                for item in json.load( file ):
                    ignore_points.append(
                        _IgnorePoint(
                            coco_class_id = int(item["coco_class_id"]),
                            at = Point2D( float(item["x"]), float(item["y"]) ),
                            cam_id = int(item["cam_id"])
                        )
                    )
        except OSError:
            pass # just start with empty list
        
        def cmp( x : _IgnorePoint, y : _IgnorePoint ) -> int:
            values = [
                x.cam_id - y.cam_id,
                x.coco_class_id - y.coco_class_id,
                x.at.y - y.at.y,
                x.at.x - y.at.x
            ]
            for value in values:
                if value != 0:
                    return value
            return 0
        
        ignore_points.sort( key = functools.cmp_to_key( cmp ) )

        return ignore_points
    
    def _ignore_point_to_dict( self, ignore_point : _IgnorePoint ) -> dict:
        return {
            "coco_class_id" : ignore_point.coco_class_id,
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