import pathlib
import typing
import statistics
import functools
from .interface import (
    Configuration,
)
from ._common import (
    Point2D,
    ObjectDetectionInfo,
    IgnorePoint,
) 
from .error_handler import (
    ErrorHandler,
)
from ._history import (
    DetectionHistory
)
from ._live_view import (
    LiveView
)
import PySide6.QtWidgets
import PySide6.QtGui
import PySide6.QtCore

class DetectionHistoryView(PySide6.QtWidgets.QFrame):
    _FOLDER = pathlib.Path("detections/")

    _configuration : Configuration
    _error_handler : ErrorHandler
    _add_to_ignore : typing.Callable[[IgnorePoint],None]

    _detection_history : DetectionHistory
    _detection_list_widget : PySide6.QtWidgets.QTreeWidget
    _detection_display : LiveView

    def __init__( self,
                  detection_history : DetectionHistory,
                  configuration : Configuration,
                  error_handler : ErrorHandler,
                  add_to_ignore : typing.Callable[[IgnorePoint],None] ):
        super().__init__()
        self._detection_history = detection_history
        self._configuration = configuration
        self._error_handler = error_handler
        self._add_to_ignore = add_to_ignore

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
        empty_image = PySide6.QtGui.QPixmap(16,9)
        empty_image.fill( PySide6.QtGui.QColorConstants.Gray )
        self._detection_display = LiveView(
            self._configuration,
            self._error_handler,
            empty_image,
        )
        detection_list_layout.addWidget( self._detection_display )
        detection_list_layout.setStretch( 1, 1)
        self._detection_list_widget.currentItemChanged.connect( lambda: self._on_current_item_change() )
        self._detection_list_widget.model().rowsInserted.connect( lambda: self._adjust_list_view_width() )       
        self._detection_list_widget.model().rowsRemoved.connect( lambda: self._adjust_list_view_width() )

        for detection in self._detection_history.get_detections():
            self._append( detection )
        
        self._detection_history.added_dispatcher.register( self._append )
        self._detection_history.removed_dispatcher.register( self._remove )
    
    def shut_down( self ) -> None:
        self._detection_display.shut_down()
    
    def graceful_handler( handler ):
        @functools.wraps( handler )
        def wrapped_handler( self : 'DetectionHistoryView', *args, **kwargs ):
            self._error_handler.handle_gracefully_internal( handler, self, *args, **kwargs )
        return wrapped_handler

    def _set_item_detection( self, item_widget : PySide6.QtWidgets.QTreeWidgetItem, detection : ObjectDetectionInfo ) -> None:
        item_widget.user_detection = detection
    def _get_item_detection( self, item_widget : PySide6.QtWidgets.QTreeWidgetItem ) -> ObjectDetectionInfo:
        return item_widget.user_detection
    
    def _append(self, detection : ObjectDetectionInfo ):
        if self._configuration.is_defined_interest( detection.supervision.coco_class_id ):
            interest_label = self._configuration.get_interest( detection.supervision.coco_class_id ).label
        else:
            interest_label = "Unknown"
        
        if self._configuration.is_defined_cam( detection.cam_id ):
            cam_label = self._configuration.get_cam_definition( detection.cam_id ).label
        else:
            cam_label = "Unknown"
        strings = [
            detection.when.strftime(r"%Y-%m-%d %H:%M:%S"),
            interest_label,
            cam_label,
            str([int(coord) for coord in detection.supervision.xyxy_coords]),
            '%.0f %%' % (detection.supervision.confidence*100)
        ]

        item = PySide6.QtWidgets.QTreeWidgetItem( None, strings )
        self._set_item_detection( item, detection )
        self._detection_list_widget.addTopLevelItem(item)
        button = PySide6.QtWidgets.QPushButton(" 🚫 ")
        button.pressed.connect( lambda: self._ignore(detection) )
        self._detection_list_widget.setItemWidget(item, 5, button)

    def _remove(self, removed_detection : ObjectDetectionInfo ):
        it = PySide6.QtWidgets.QTreeWidgetItemIterator(self._detection_list_widget)
        while it.value():
            item = it.value()
            if self._get_item_detection(item) is removed_detection:
                self._detection_list_widget.invisibleRootItem().removeChild( item )
                return
            it += 1
        raise ValueError("Detection not found.")
 
    @graceful_handler
    def _on_current_item_change(self) -> None:
        if self._detection_list_widget.currentIndex() is None:
            empty_image = PySide6.QtGui.QPixmap(16,9)
            empty_image.fill( PySide6.QtGui.QColorConstants.Gray )
            self._detection_display.setPixmap( empty_image )
            return
        
        item = self._detection_list_widget.currentItem()
        image_data = self._detection_history.get_detection_image_data( self._get_item_detection( item ) )
        image = PySide6.QtGui.QImage( image_data, image_data.shape[1], image_data.shape[0], image_data.strides[0], PySide6.QtGui.QImage.Format.Format_RGB888)
        pixmap = PySide6.QtGui.QPixmap.fromImage( image )
        self._detection_display.setPixmap( pixmap )
    
    @graceful_handler
    def _ignore(self, detection : ObjectDetectionInfo ):
        xyxy_coords = detection.supervision.xyxy_coords
        x = float( statistics.mean( [xyxy_coords[0], xyxy_coords[2]] ) / detection.frame_size.x )
        y = float( statistics.mean( [xyxy_coords[1], xyxy_coords[3]] ) / detection.frame_size.y )
        self._add_to_ignore( IgnorePoint( coco_class_id=detection.supervision.coco_class_id, at=Point2D(x,y), cam_id=detection.cam_id ) )

    @graceful_handler
    def _adjust_list_view_width(self):
        for i in range( 0, self._detection_list_widget.columnCount()):
            self._detection_list_widget.resizeColumnToContents(i)
        
        total_width = sum([self._detection_list_widget.sizeHintForColumn(i) for i in range(0, self._detection_list_widget.columnCount())])
        self._detection_list_widget.setMaximumWidth( total_width + 50)
