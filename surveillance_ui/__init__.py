import sys
import av.container
import numpy
import av
import math
import threading
import queue
import time
import datetime
import typing
from .common import (
    SupervisionDetections,
    CamDefinition,
    Interest,
    Configuration,
    DetectionLogic,
    Point2D,
    _SvDetection,
    _FrameInfo,
    _ImageDetectionsInfo,
    _ObjectDetectionInfo,
)
from .utility import (
    LastFrameVideoCapture,
    FittingImage,
)
from ._history import (
    SurveillanceHistoryView
)
from ._ignore_list import (
    IgnoreListView
)

import PySide6.QtCore
import PySide6.QtWidgets
import PySide6.QtGui

class _SurveillanceWindowSignals(PySide6.QtCore.QObject):
    frame = PySide6.QtCore.Signal( _FrameInfo )
    detection = PySide6.QtCore.Signal( _ImageDetectionsInfo )

class _Detector():
    _configuration : Configuration
    _model_update_queue : queue.Queue[tuple[list[int],float]]
    _filter_ignored : typing.Callable[[SupervisionDetections,CamDefinition,Point2D[int]],SupervisionDetections]
    _on_detection : typing.Callable[[_ImageDetectionsInfo],None] 
    _on_frame : typing.Callable[[_FrameInfo],None]
    _shut_down_pending : bool = False

    _thread : threading.Thread
    _last_frame_captures : list[LastFrameVideoCapture]

    def __init__( 
            self,
            configuration : Configuration,
            filter_ignored : typing.Callable[[SupervisionDetections,CamDefinition,Point2D[int]],SupervisionDetections],
            on_detection : typing.Callable[[_ImageDetectionsInfo],None],
            on_frame : typing.Callable[[_FrameInfo],None]
        ):
        super().__init__()
        self._configuration = configuration
        self._filter_ignored = filter_ignored
        self._on_detection = on_detection
        self._on_frame = on_frame
        self._model_update_queue = queue.Queue(1)

        self._last_frame_captures = [self._open_cam( cam_definition ) for cam_definition in self._configuration.cam_definitions]

        self._thread = threading.Thread(target=self._detector_process)
        self._thread.daemon = True
        self._thread.start()


    def update_model( self, coco_classes : list[int], confidence : float ):
        try:
            self._model_update_queue.get_nowait()
        except queue.Empty:
            pass
        update_values = [coco_classes, confidence]
        self._model_update_queue.put( update_values )
    
    def shut_down( self ) -> None:
        self._shut_down_pending = True
        capture_shutdown_threads = [threading.Thread(target=last_frame_capture.shut_down) for last_frame_capture in self._last_frame_captures]
        for capture_shutdown_thread in capture_shutdown_threads:
            capture_shutdown_thread.start()
        
        self._thread.join()
        for capture_shutdown_thread in capture_shutdown_threads:
            capture_shutdown_thread.join()
    
    def _open_cam( self, cam_definition : CamDefinition ) -> LastFrameVideoCapture:
        def input_container_constructor() -> av.container.InputContainer:
            input_container = av.open(cam_definition.url, 'r')
            input_container.streams.video[0].codec_context.low_delay = True
            return input_container
        
        def on_frame( frame : numpy.ndarray ):
            self._on_frame( _FrameInfo( image=frame, cam_id=cam_definition.id ) )
        
        return LastFrameVideoCapture( input_container_constructor, on_frame=on_frame )

    def _detector_process(self):
        import supervision
        annotator = supervision.BoundingBoxAnnotator(thickness=5)
        
        default_interests = filter( lambda i: i.enabled_by_default, self._configuration.interests)
        default_coco_class_ids = [interest.coco_class_id for interest in default_interests]
        self._configuration.detection_logic.configure( default_coco_class_ids, self._configuration.initial_confidence )

        while True:
            for cam_definition, last_frame_capture in zip( self._configuration.cam_definitions, self._last_frame_captures ):
                if self._shut_down_pending:
                    return
                
                try:
                    values = self._model_update_queue.get_nowait()
                    self._configuration.detection_logic.configure( *values )
                except queue.Empty:
                    pass

                frame = last_frame_capture.get_latest_frame(timeout=0.01)
                if frame is None:
                    continue
                
                detections = self._filter_ignored( self._configuration.detection_logic.detect( frame ), cam_definition, Point2D(frame.shape[1], frame.shape[0]) )
                
                if len(detections) > 0:
                    annotated_frame = annotator.annotate(scene=frame.copy(), detections=detections)
                    frame_info = _FrameInfo( image=annotated_frame, cam_id=cam_definition.id )
                    sv_detections = _SvDetection.list_from_sv_detections(detections)
                    self._on_detection( _ImageDetectionsInfo(frame_info, sv_detections, datetime.datetime.now() ) )

class _AlertPlayer:
    _sound_alert_queue : queue.Queue[tuple[_ImageDetectionsInfo,float]]
    _player_thread : threading.Thread
    _shut_down_pending : bool = False
    _configuration : Configuration

    def __init__(self, configuration : Configuration ):
        self._configuration = configuration
        self._sound_alert_queue = queue.Queue(1)
        self._player_thread = threading.Thread(target=self._play_sound_proc)
        self._player_thread.daemon = True
        self._player_thread.start()

    def _play_sound_proc(self):
        import pygame # pre-load pygame library
        while True:
            if self._shut_down_pending:
                return

            try:
                image_detections_info, sound_volume = self._sound_alert_queue.get(timeout=0.01)
            except queue.Empty:
                continue

            for detection in image_detections_info.detections[:5]:
                interest = self._configuration.get_interest( detection.coco_class_id  )
                _play_sound_file_blocking(interest.sound_alert_path, sound_volume)
            cam_definition = self._configuration.get_cam_definition( image_detections_info.frame_info.cam_id )
            _play_sound_file_blocking(cam_definition.sound_alert_path, sound_volume)
            time.sleep(0.5)
    
    def try_fire(self, image_detections_info : _ImageDetectionsInfo, sound_volume : float ):
        if sound_volume <= 0.001:
            return
        try:
            self._sound_alert_queue.put_nowait( [image_detections_info, sound_volume] )
        except queue.Full:
            pass
    
    def shut_down(self) -> None:
        self._shut_down_pending = True
        self._player_thread.join()

class QCamScrollArea(PySide6.QtWidgets.QScrollArea):

    def resizeEvent( self, event ) -> None:
        super().resizeEvent( event )
        self.adjustCamSizes()
    
    def showEvent( self, event ) -> None:
        super().showEvent( event )
        self.adjustCamSizes()

    def adjustCamSizes( self ):
        if self.widget().layout() is None:
            return
        
        layout : PySide6.QtWidgets.QGridLayout = self.widget().layout()

        def row_items(row):
            items = [layout.itemAtPosition(row, column) for column in range( 0, layout.columnCount())]
            return filter( lambda item: item is not None, items )
        def row_heights_matching_aspect(row):
            return [item.widget().heightMatchingAspect() for item in row_items(row)]
        
        for row in range( 0, layout.rowCount() ):
            ideal_height = max( row_heights_matching_aspect(row) )
            for item in row_items(row):
                item.widget().setFixedHeight( ideal_height )

class SurveillanceWindow(PySide6.QtWidgets.QMainWindow):
    
    _configuration : Configuration

    # components
    _alert_player : _AlertPlayer
    _detector : _Detector

    # widgets
    _coco_check_boxes : list[PySide6.QtWidgets.QCheckBox]
    _sensitivity_slider : PySide6.QtWidgets.QSlider
    _sound_volume_slider : PySide6.QtWidgets.QSlider
    _cams_tab : PySide6.QtWidgets.QTabWidget
    _live_view_widgets : list[FittingImage]
    _annotation_widgets : list[FittingImage]

    _ignore_list_view : IgnoreListView

    _history_view : SurveillanceHistoryView

    _multiview_scroll_area : QCamScrollArea
    _multiview_live_view_widgets : list[FittingImage]
    _multiview_annotation_widgets : list[FittingImage]

    _signals : _SurveillanceWindowSignals # receives events from worker threads
    
    def __init__(self, configuration : Configuration ):
        """
        Params:
            cam_definitions - cams to be surveilled
            interests - objects of interest to be detected on cams
            detection_logic - detection algorithm to be used, this object will be used by a worker thread
        """
        super().__init__()
        
        self._configuration = configuration

        self._alert_player = _AlertPlayer( self._configuration )
        self._signals = _SurveillanceWindowSignals()

        self.setWindowTitle("AI Surveillant")
        self.setMinimumSize( 500, 500 )
        self.showMaximized()

        layout = PySide6.QtWidgets.QVBoxLayout()

        controls_bar_layout = PySide6.QtWidgets.QHBoxLayout()
        layout.addLayout( controls_bar_layout )

        self._coco_check_boxes = []
        for interest in self._configuration.interests:
            check_box = PySide6.QtWidgets.QCheckBox(interest.label)
            check_box.setChecked( interest.enabled_by_default )
            check_box.stateChanged.connect( self._on_configuration_change )
            self._coco_check_boxes.append( check_box )
            controls_bar_layout.addWidget( check_box )
        
        controls_bar_layout.addWidget( self._make_vertical_line() )

        controls_bar_layout.addWidget( PySide6.QtWidgets.QLabel(text="👁 ") )
        self._sensitivity_slider, sensitivity_slider_percentage = self._make_percentage_slider( int( (1-configuration.initial_confidence) * 100) )
        self._sensitivity_slider.valueChanged.connect( self._on_configuration_change )
        controls_bar_layout.addWidget( self._sensitivity_slider )
        controls_bar_layout.addWidget( sensitivity_slider_percentage )
        controls_bar_layout.addWidget( PySide6.QtWidgets.QLabel(text="👁👁👁") )

        controls_bar_layout.addWidget( self._make_vertical_line() )

        controls_bar_layout.addWidget( PySide6.QtWidgets.QLabel(text="🔊 ") )
        self._sound_volume_slider, sound_volume_slider_percentage = self._make_percentage_slider( 50 )      
        controls_bar_layout.addWidget( self._sound_volume_slider )
        controls_bar_layout.addWidget( sound_volume_slider_percentage )
        controls_bar_layout.addStretch()

        self._cams_tab = PySide6.QtWidgets.QTabWidget()
        layout.addWidget( self._cams_tab )

        self._multiview_scroll_area = QCamScrollArea()
        self._multiview_scroll_area.setHorizontalScrollBarPolicy( PySide6.QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff )
        self._multiview_scroll_area.setVerticalScrollBarPolicy( PySide6.QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOn )
        self._multiview_scroll_area.setWidgetResizable(True)
        self._cams_tab.addTab( self._multiview_scroll_area, "  *  " )

        def get_cam_image( cam_id : int ) -> PySide6.QtGui.QImage:
            cam_index = self._configuration.cam_definitions.index( self._configuration.get_cam_definition( cam_id ) )
            return PySide6.QtGui.QPixmap( self._live_view_widgets[cam_index].pixmap() )

        self._ignore_list_view = IgnoreListView( self._configuration, get_cam_image )

        self._history_view = SurveillanceHistoryView( self._configuration, lambda ignore_point: self._ignore_list_view.append(ignore_point) )
        self._cams_tab.addTab( self._history_view, " !,!,... " )

        multiview_scroll_subwidget = PySide6.QtWidgets.QWidget()
        self._multiview_scroll_area.setWidget( multiview_scroll_subwidget )

        multiview_layout = PySide6.QtWidgets.QGridLayout()
        multiview_scroll_subwidget.setLayout( multiview_layout )

        self._live_view_widgets = []        
        self._annotation_widgets = []
        self._multiview_live_view_widgets = []
        self._multiview_annotation_widgets = []
        for index, cam_definition in enumerate( self._configuration.cam_definitions ):

            live_view_widget = FittingImage( 50, 50 )
            self._live_view_widgets.append( live_view_widget )
            self._cams_tab.addTab( live_view_widget, cam_definition.label )
     
            self._annotation_widgets.append( FittingImage( 50, 50 ) )
            self._cams_tab.addTab( self._annotation_widgets[-1], cam_definition.label + " ! " )

            multiview_cam_widget = FittingImage( 50, 50 )
            self._multiview_live_view_widgets.append(multiview_cam_widget)
            row = math.floor(index / self._configuration.grid_column_count)
            column = index - (row*self._configuration.grid_column_count)
            multiview_layout.addWidget( multiview_cam_widget, row, column, 1, 1 )

            row += math.ceil( len(self._configuration.cam_definitions) / self._configuration.grid_column_count ) # put annotations below live views
            multiview_annotation_widget = FittingImage( 50, 50 )
            self._multiview_annotation_widgets.append(multiview_annotation_widget)
            multiview_layout.addWidget( multiview_annotation_widget, row, column, 1, 1 )

        for i in range( 0, self._configuration.grid_column_count ):
            multiview_layout.setColumnStretch( i, 1 )
        for i in range( 0, math.ceil( (len(self._configuration.cam_definitions)+1) / self._configuration.grid_column_count ) ):
            multiview_layout.setRowStretch( i, 1 )
        
        self._cams_tab.addTab( self._ignore_list_view, " 👁🚫 " )

        central_widget = PySide6.QtWidgets.QWidget()
        central_widget.setLayout(layout)

        self.setCentralWidget(central_widget)

        self._signals.detection.connect(self._on_detection)
        self._signals.frame.connect(self._on_frame)

        self._detector = _Detector( 
            configuration=self._configuration,
            filter_ignored=self._ignore_list_view.filter_ignored,
            on_detection=self._signals.detection.emit,
            on_frame=self._signals.frame.emit
        )

    def closeEvent( self, event: PySide6.QtGui.QCloseEvent ) -> None:
        shutdown_alert_thread = threading.Thread(target=self._alert_player.shut_down)
        shutdown_alert_thread.start()
        shutdown_detector_thread = threading.Thread(target=self._detector.shut_down)
        shutdown_detector_thread.start()
        shutdown_alert_thread.join()
        shutdown_detector_thread.join()

        super().closeEvent(event)

    def _on_configuration_change(self):
        self._detector.update_model( self._get_selected_coco_classes(), self._get_selected_confidence() )
    
    def _get_selected_coco_classes(self) -> list[int]:
        classes = []
        for index, interest in enumerate( self._configuration.interests ):
            if self._coco_check_boxes[index].isChecked():
                classes.append(interest.coco_class_id)
        
        return classes
    
    def _get_selected_confidence(self) -> float:
        # the slider is 0-100 and inverse of confidence
        return (100 - self._sensitivity_slider.value()) / 100

    def _on_detection(self, image_detections_info : _ImageDetectionsInfo ) -> None:

        shape = image_detections_info.frame_info.image.shape
        frame_size = Point2D( shape[1], shape[0] )
        
        fresh_detections : list[_SvDetection] = []
        for detection in image_detections_info.detections:
            single_detection_info = _ObjectDetectionInfo( cam_id=image_detections_info.frame_info.cam_id, supervision=detection, when=image_detections_info.when, frame_size=frame_size )
            if self._history_view.persist_if_fresh( single_detection_info, image=image_detections_info.frame_info.image ):
                fresh_detections.append(detection)

        if len(fresh_detections) > 0:
            fresh_detection_info = _ImageDetectionsInfo(
                frame_info=image_detections_info.frame_info,
                detections=fresh_detections,
                when=image_detections_info.when
            )
            self._alert_player.try_fire( image_detections_info=fresh_detection_info, sound_volume=self._sound_volume_slider.value()/100 )
        
        pixmap = self._make_pixmap( image_detections_info.frame_info.image )
        index = self._configuration.cam_definitions.index( self._configuration.get_cam_definition(image_detections_info.frame_info.cam_id) )
        self._set_pixmap( self._annotation_widgets[index], pixmap )
        self._set_pixmap( self._multiview_annotation_widgets[index], pixmap )
    
    def _on_frame(self, frame_info : _FrameInfo ) -> None:
        for cam_definition, live_view_cam_widget, multiview_cam_widget in zip( self._configuration.cam_definitions, self._live_view_widgets, self._multiview_live_view_widgets ):
            if cam_definition.id == frame_info.cam_id:
                pixmap = self._make_pixmap( frame_info.image )
                self._set_pixmap( live_view_cam_widget, pixmap )
                self._set_pixmap( multiview_cam_widget, pixmap )
    
    def _make_pixmap(self, frame : numpy.ndarray ):
        q_image = PySide6.QtGui.QImage( frame, frame.shape[1], frame.shape[0], frame.strides[0], PySide6.QtGui.QImage.Format.Format_RGB888)
        return PySide6.QtGui.QPixmap.fromImage(q_image)
    
    def _set_pixmap(self, widget : FittingImage, pixmap : PySide6.QtGui.QPixmap ):
        previous = widget.pixmap()
        widget.setPixmap( pixmap )
        if previous is None or pixmap.size() != previous.size():
            self._multiview_scroll_area.adjustCamSizes()

    def _make_vertical_line(self) -> PySide6.QtWidgets.QFrame:
        retval = PySide6.QtWidgets.QFrame()
        retval.setFrameShape(PySide6.QtWidgets.QFrame.VLine)
        retval.setFrameShadow(PySide6.QtWidgets.QFrame.Sunken)
        return retval
    
    def _make_percentage_slider(self, initial_value : int ) -> tuple[PySide6.QtWidgets.QSlider, PySide6.QtWidgets.QLabel]:
        slider = PySide6.QtWidgets.QSlider( PySide6.QtCore.Qt.Orientation.Horizontal )
        slider.setMinimum(0)
        slider.setMaximum(100)
        slider.setSingleStep(1)
        slider.setPageStep(10)
        slider.setValue(initial_value)
        slider.setMaximumSize( 200, 50 )
                
        percentage = PySide6.QtWidgets.QLabel(f"{initial_value} %")
        percentage.setMinimumSize(30,1)
        percentage.setAlignment( PySide6.QtCore.Qt.AlignmentFlag.AlignRight )
        def update_percentage():
            percentage.setText(f"{slider.value()} %")
        slider.valueChanged.connect( update_percentage )
        
        return slider, percentage

def _play_sound_file_blocking( file : str, sound_volume : float ):
    if file == None:
        return
    import pygame
    pygame.mixer.init()    
    pygame.mixer.music.load(file)
    pygame.mixer.music.set_volume(sound_volume)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(0.01)

def run_surveillance_application( configuration : Configuration ) -> int:
    """
    Run surveillance application.

    Params:
        cam_definitions - cams to be surveilled
        interests - objects of interest to be detected on cams
        detection_logic - detection algorithm to be used, this object will be used by a worker thread
    
    Returns: Application exit code
    """
    app = PySide6.QtWidgets.QApplication(sys.argv)
    window = SurveillanceWindow( configuration )
    window.show()
    return app.exec()
