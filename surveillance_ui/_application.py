import av.container
import numpy
import av
import math
import threading
import queue
import datetime
import typing
import functools
import supervision
import dataclasses
from .interface import (
    CamDefinition,
    Configuration,
)
from ._common import (
    Point2D,
    SvDetection,
    ObjectDetectionInfo,
)
from .utility import (
    LastFrameVideoCapture,
    AudioStreamPlayer,
    make_percentage_slider,
)
from .error_handler import (
    ErrorHandler,
)
from ._history import (
    DetectionHistory
)
from ._history_view import (
    DetectionHistoryView
)
from ._ignore_list import (
    IgnoreList
)
from ._ignore_list_view import (
    IgnoreListView
)
from ._live_view import (
    LiveView
)

import PySide6.QtCore
import PySide6.QtWidgets
import PySide6.QtGui
import PySide6.QtMultimedia

@dataclasses.dataclass
class _FrameInfo:
    image : numpy.ndarray
    cam_id : int

@dataclasses.dataclass
class _ImageDetectionsInfo:
    frame_info : _FrameInfo
    detections : list[SvDetection]
    when : datetime.datetime

class _SurveillanceWindowSignals(PySide6.QtCore.QObject):
    frame = PySide6.QtCore.Signal( _FrameInfo )
    detection = PySide6.QtCore.Signal( _ImageDetectionsInfo )

@dataclasses.dataclass
class _AudioChunk:
    chunk : bytes
    cam_id : int

class _Detector():
    _configuration : Configuration
    _model_update_queue : queue.Queue[tuple[list[int],float]]
    _filter_ignored : typing.Callable[[supervision.Detections,CamDefinition,Point2D[int]],supervision.Detections]
    _on_detection : typing.Callable[[_ImageDetectionsInfo],None] 
    _on_frame : typing.Callable[[_FrameInfo],None]
    _on_audio_chunk : typing.Callable[[_AudioChunk],None]    
    _on_uncaught_exception : typing.Callable[[BaseException,str],None]
    _shutdown_pending : bool = False

    _thread : threading.Thread
    _last_frame_captures : list[LastFrameVideoCapture]

    def __init__( 
            self,
            configuration : Configuration,
            filter_ignored : typing.Callable[[supervision.Detections,CamDefinition,Point2D[int]],supervision.Detections],
            on_detection : typing.Callable[[_ImageDetectionsInfo],None],
            on_frame : typing.Callable[[_FrameInfo],None],
            on_audio_chunk : typing.Callable[[_AudioChunk],None] | None,            
            on_uncaught_exception : typing.Callable[[BaseException,str],None]
        ):
        super().__init__()
        self._configuration = configuration
        self._filter_ignored = filter_ignored
        self._on_detection = on_detection
        self._on_frame = on_frame
        self._on_audio_chunk = on_audio_chunk
        self._on_uncaught_exception = on_uncaught_exception
        self._model_update_queue = queue.Queue(1)

        self._last_frame_captures = [self._open_cam( cam_definition ) for cam_definition in self._configuration.cam_definitions]

        def graceful_detector_process():
            try:
                self._detector_process()
            except BaseException as e: # NOSONAR
                on_uncaught_exception( e, "The detection thread has crashed." )

        self._thread = threading.Thread( target=graceful_detector_process )
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
        self._shutdown_pending = True
        capture_shutdown_threads = [threading.Thread(target=last_frame_capture.shut_down) for last_frame_capture in self._last_frame_captures]
        for capture_shutdown_thread in capture_shutdown_threads:
            capture_shutdown_thread.start()
        
        self._thread.join()
        for capture_shutdown_thread in capture_shutdown_threads:
            capture_shutdown_thread.join()
    
    def _open_cam( self, cam_definition : CamDefinition ) -> LastFrameVideoCapture:
        def input_container_constructor() -> av.container.InputContainer:
            input_container = av.open(
                cam_definition.url,
                'r',
                timeout=self._configuration.camera_feed_timeout.total_seconds(),
                options= {
                    'rtsp_transport': 'tcp' if self._configuration.use_tcp_transport else 'udp',
                    'stimeout' : str(self._configuration.camera_feed_timeout.total_seconds()*pow(10,6)),
                    'max_delay': str(self._configuration.max_delay.total_seconds()*pow(10,6)),
                },
            )
            if cam_definition.discard_corrupted_frames:
                input_container.flags |= av.container.Flags.DISCARD_CORRUPT
            return input_container
        
        def on_frame( frame : numpy.ndarray ):
            self._on_frame( _FrameInfo( image=frame, cam_id=cam_definition.id ) )
        
        def on_audio_bytes( audio_bytes : bytes ):
            self._on_audio_chunk( _AudioChunk( chunk=audio_bytes, cam_id=cam_definition.id ) )

        def on_uncaught_cam_exception( exception : BaseException ):
            self._on_uncaught_exception( exception, f"{cam_definition.label} cam thread has crashed." )
        
        return LastFrameVideoCapture( input_container_constructor, on_frame=on_frame, on_audio_bytes=on_audio_bytes, on_uncaught_exception=on_uncaught_cam_exception )

    def _detector_process(self):
        annotator = supervision.BoundingBoxAnnotator(thickness=5)
        
        default_interests = filter( lambda i: i.enabled_by_default, self._configuration.interests)
        default_coco_class_ids = [interest.coco_class_id for interest in default_interests]
        self._configuration.detection_logic.configure( default_coco_class_ids, self._configuration.initial_confidence )

        while True:
            for cam_definition, last_frame_capture in zip( self._configuration.cam_definitions, self._last_frame_captures ):
                if self._shutdown_pending:
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
                    sv_detections = SvDetection.list_from_sv_detections(detections)
                    self._on_detection( _ImageDetectionsInfo(frame_info, sv_detections, datetime.datetime.now() ) )

class _AlertPlayer:
    _configuration : Configuration
    _error_handler : ErrorHandler
    _sound_path_to_media_player : dict[str,PySide6.QtMultimedia.QMediaPlayer]
    _audio_outputs : list[PySide6.QtMultimedia.QAudioOutput]
    _sound_queue : list[list[PySide6.QtMultimedia.QMediaPlayer]]
    _ready_to_play : bool

    def __init__(self, configuration : Configuration, error_handler : ErrorHandler ):
        self._configuration = configuration
        self._error_handler = error_handler

        self._ready_to_play = True
        self._sound_queue = []
        self._sound_path_to_media_player = dict()
        self._audio_outputs = []

    def graceful_handler( handler ):
        @functools.wraps( handler )
        def wrapped_handler( self : '_AlertPlayer', *args, **kwargs ):
            self._error_handler.handle_gracefully_internal( handler, self, *args, **kwargs )
        return wrapped_handler

    def try_alert(self, image_detections_info : _ImageDetectionsInfo ):     
        sounds = []
        for detection in image_detections_info.detections[:5]:
            interest = self._configuration.get_interest( detection.coco_class_id  )
            if interest.sound_alert_path is not None:
                sounds.append( self.get_sound( interest.sound_alert_path ) )
        cam_definition = self._configuration.get_cam_definition( image_detections_info.frame_info.cam_id )
        if cam_definition.sound_alert_path is not None:
            sounds.append( self.get_sound( cam_definition.sound_alert_path ) )
        
        if len(sounds) == 0:
            return
        
        self._sound_queue.append( sounds )

        self._try_play_next_sound()
    
    def set_volume(self, volume : float) -> None:
        for media_player in self._sound_path_to_media_player.values():
            media_player.audioOutput().setVolume(volume)

    def shut_down(self) -> None:
        for media_player in self._sound_path_to_media_player.values():
            media_player.stop()
    
    @graceful_handler
    def _on_media_status_change(self, status : PySide6.QtMultimedia.QMediaPlayer.MediaStatus ):
        if status == PySide6.QtMultimedia.QMediaPlayer.MediaStatus.EndOfMedia:
            self._ready_to_play = True
            self._try_play_next_sound()
    
    @graceful_handler
    def _try_play_next_sound(self) -> None:
        if self._ready_to_play and len(self._sound_queue) > 0:
            self._ready_to_play = False

            media_player = self._sound_queue[0].pop(0)
            if len( self._sound_queue[0] ) == 0:
                self._sound_queue.pop(0)
            
            media_player.play()

    def get_sound( self, sound_path : str ) -> PySide6.QtMultimedia.QMediaPlayer:
        if sound_path in self._sound_path_to_media_player:
            return self._sound_path_to_media_player[sound_path]
        
        media_player = PySide6.QtMultimedia.QMediaPlayer()
        audio_output = PySide6.QtMultimedia.QAudioOutput()
        self._sound_path_to_media_player[sound_path] = media_player
        self._audio_outputs.append( audio_output )
        media_player.setAudioOutput( audio_output )
        media_player.setSource( PySide6.QtCore.QUrl.fromLocalFile( sound_path ) )
        media_player.mediaStatusChanged.connect( self._on_media_status_change )
        return media_player

class _OverviewLayout(typing.Protocol):
    def adjust_cam_sizes(self) -> None:
        """Adjust cam sizes so they fit properly."""

class QCamScrollArea(PySide6.QtWidgets.QScrollArea):

    _error_handler : ErrorHandler

    def __init__( self, error_handler : ErrorHandler ):
        self._error_handler = error_handler
        super().__init__()
    
    def graceful_handler( handler ):
        @functools.wraps( handler )
        def wrapped_handler( self : 'QCamScrollArea', *args, **kwargs ):
            self._error_handler.handle_gracefully_internal( handler, self, *args, **kwargs )
        return wrapped_handler

    @graceful_handler
    def resizeEvent( self, event ) -> None:
        super().resizeEvent( event )
        self._get_layout().adjust_cam_sizes()
    
    @graceful_handler
    def showEvent( self, event ) -> None:
        super().showEvent( event )
        self._get_layout().adjust_cam_sizes()
  
    def _get_layout( self ) -> _OverviewLayout:
        return self.widget().layout()

class _OverviewAutoLayout(PySide6.QtWidgets.QGridLayout):
    _configuration : Configuration
    
    def __init__(self, configuration : Configuration):
        super().__init__()
        self._configuration = configuration

        self.setHorizontalSpacing(0)
        self.setVerticalSpacing(0)

    def addWidget( self, widget : PySide6.QtWidgets.QWidget ) -> None:
        index = self.count()

        group_row = 0

        if index >= len(self._configuration.cam_definitions):
            # annotations, lay them out separately below live views
            index -= len(self._configuration.cam_definitions)
            group_row = math.ceil( len(self._configuration.cam_definitions) / self._configuration.grid_column_count )

        row = group_row + math.floor(index / self._configuration.grid_column_count)
        column = index % self._configuration.grid_column_count
        
        self.setRowStretch( row, 1 )
        self.setColumnStretch( column, 1 )

        super().addWidget( widget, row, column, 1, 1 )
    
    def adjust_cam_sizes(self):
        def row_items(row):
            items = [self.itemAtPosition(row, column) for column in range( 0, self.columnCount())]
            return filter( lambda item: item is not None, items )
        def row_heights_matching_aspect(row):
            return [item.widget().heightMatchingAspect() for item in row_items(row)]
        
        for row in range( 0, self.rowCount() ):
            ideal_height = max( row_heights_matching_aspect(row) )
            for item in row_items(row):
                item.widget().setFixedHeight( ideal_height )

class _OverviewManualLayout(PySide6.QtWidgets.QLayout):
    _configuration : Configuration
    _items : list[PySide6.QtWidgets.QLayoutItem]
    _aspect_ratio : PySide6.QtCore.QSize
    _widget_locations : list[PySide6.QtCore.QRectF]

    def __init__(self, configuration : Configuration):
        super().__init__()
        self._configuration = configuration
        self._items = []

        loc_count = len( self._configuration.grid_widget_locs )
        cam_count = len( self._configuration.cam_definitions ) 
        if loc_count not in [cam_count, 2*cam_count]:
            raise ValueError(f"Manual layout requires widget location count ({loc_count} provided) matching or double of camera count ({cam_count} provided).")
        
        self._widget_locations = self._configuration.grid_widget_locs.copy()
        row_count = max( [w.y()+w.height() for w in self._widget_locations] )
        column_count = max( [w.x()+w.width() for w in self._widget_locations] + [self._configuration.grid_column_count] )

        if loc_count == cam_count:
            # add another set of widget locations under the existing ones for the annotation widgets
            self._widget_locations.extend( [PySide6.QtCore.QRectF( w.x(), w.y()+row_count, w.width(), w.height() ) for w in self._widget_locations] )
            row_count *= 2

        self._aspect_ratio = PySide6.QtCore.QSize( column_count, row_count )
    
    def count( self ) -> int:
        return len(self._items)

    def addItem( self, item : PySide6.QtWidgets.QLayoutItem ) -> None:
        self._items.append(item)
    
    def itemAt( self, index : int ) -> PySide6.QtWidgets.QLayoutItem | None:
        if index < 0 or len(self._items) <= index:
            return None
        return self._items[index]
    
    def takeAt( self, index : int ) -> PySide6.QtWidgets.QLayoutItem | None:
        if index < 0 or len(self._items) <= index:
            return None
        return self._items.pop(index)
    
    def sizeHint( self ) -> PySide6.QtCore.QSize:
        return self._aspect_ratio
    
    def setGeometry(self, rect: PySide6.QtCore.QRect) -> None:
        magnification = rect.width() / self._aspect_ratio.width()

        def magnify_rectangle( r : PySide6.QtCore.QRectF ) -> PySide6.QtCore.QRect:
            return PySide6.QtCore.QRect(
                PySide6.QtCore.QPoint(
                    math.floor( round( r.left() * magnification, 2 ) ), # the rounding is to counter floating point imprecision.
                    math.floor( round( r.top() * magnification, 2 ) )
                ),
                PySide6.QtCore.QPoint(
                    math.floor( round( r.right() * magnification, 2 ) ) - 1,
                    math.floor( round( r.bottom() * magnification,  2 ) ) - 1,
                )
            )
        
        for location, widget in zip( self._widget_locations, self._items ):
            widget.setGeometry( magnify_rectangle( location ) )
        
    def hasHeightForWidth( self ) -> bool:
        return True
    
    def heightForWidth( self, width : int ) -> int:
        magnification = width / self._aspect_ratio.width()
        return math.floor( self._aspect_ratio.height() * magnification )

    def adjust_cam_sizes(self):
        pass # let's not end up in an infinite recursion

class SurveillanceWidget(PySide6.QtWidgets.QWidget):
    
    _configuration : Configuration

    # components
    _alert_player : _AlertPlayer
    _detector : _Detector
    _history : DetectionHistory
    _ignore_list : IgnoreList
    
    # Qt
    _coco_check_boxes : list[PySide6.QtWidgets.QCheckBox]
    _sensitivity_slider : PySide6.QtWidgets.QSlider
    _sound_volume_slider : PySide6.QtWidgets.QSlider
    _cams_tab : PySide6.QtWidgets.QTabWidget
    _live_view_widgets : list[LiveView]
    _annotation_widgets : list[LiveView]

    _ignore_list_view : IgnoreListView
    _history_view : DetectionHistoryView

    _overview_scroll_area : QCamScrollArea
    _overview_live_view_widgets : list[LiveView]
    _overview_annotation_widgets : list[LiveView]

    _cam_id_to_audio_stream_player_dict : dict[int,AudioStreamPlayer]
    
    _error_handler : ErrorHandler

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
        
        self._signals = _SurveillanceWindowSignals()

        self._error_handler = ErrorHandler(self)

        self._alert_player = _AlertPlayer( self._configuration, self._error_handler )
        self._alert_player.set_volume( 0.5 )

        self.setMinimumSize( 500, 500 )

        layout = PySide6.QtWidgets.QVBoxLayout()

        controls_bar_layout = PySide6.QtWidgets.QHBoxLayout()
        layout.addLayout( controls_bar_layout )

        self._coco_check_boxes = []
        for interest in self._configuration.interests:
            check_box = PySide6.QtWidgets.QCheckBox(interest.label)
            check_box.setChecked( interest.enabled_by_default )
            check_box.stateChanged.connect( lambda: self._on_configuration_change() )
            self._coco_check_boxes.append( check_box )
            controls_bar_layout.addWidget( check_box )
        
        controls_bar_layout.addWidget( self._make_vertical_line() )

        controls_bar_layout.addWidget( PySide6.QtWidgets.QLabel(text="👁 ") )
        self._sensitivity_slider, sensitivity_slider_percentage = make_percentage_slider( self._error_handler, int( (1-configuration.initial_confidence) * 100) )
        self._sensitivity_slider.valueChanged.connect( lambda: self._on_configuration_change() )
        controls_bar_layout.addWidget( self._sensitivity_slider )
        controls_bar_layout.addWidget( sensitivity_slider_percentage )
        controls_bar_layout.addWidget( PySide6.QtWidgets.QLabel(text="👁👁👁") )

        controls_bar_layout.addWidget( self._make_vertical_line() )

        controls_bar_layout.addWidget( PySide6.QtWidgets.QLabel(text="🔊 ") )
        self._sound_volume_slider, sound_volume_slider_percentage = make_percentage_slider( self._error_handler, 50 )      
        controls_bar_layout.addWidget( self._sound_volume_slider )
        controls_bar_layout.addWidget( sound_volume_slider_percentage )
        controls_bar_layout.addStretch()
        self._sound_volume_slider.valueChanged.connect( self._on_alert_volume_change )

        self._cams_tab = PySide6.QtWidgets.QTabWidget()
        layout.addWidget( self._cams_tab )

        self._overview_scroll_area = QCamScrollArea( self._error_handler )
        self._overview_scroll_area.setHorizontalScrollBarPolicy( PySide6.QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded )
        self._overview_scroll_area.setVerticalScrollBarPolicy( PySide6.QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOn )
        self._overview_scroll_area.setWidgetResizable(True)
        self._cams_tab.addTab( self._overview_scroll_area, "  *  " )

        def get_cam_image( cam_id : int ) -> PySide6.QtGui.QImage:
            cam_index = self._configuration.cam_definitions.index( self._configuration.get_cam_definition( cam_id ) )
            return PySide6.QtGui.QPixmap( self._live_view_widgets[cam_index].pixmap() )

        self._ignore_list = IgnoreList( self._configuration )
        self._ignore_list_view = IgnoreListView( self._ignore_list, self._configuration, self._error_handler, get_cam_image )

        self._history = DetectionHistory( self._configuration )
        self._history_view = DetectionHistoryView(
            detection_history=self._history,
            configuration=self._configuration,
            error_handler=self._error_handler,
            add_to_ignore=lambda ignore_point: self._ignore_list.add(ignore_point)
        )
        self._cams_tab.addTab( self._history_view, " !,!,... " )

        overview_scroll_subwidget = PySide6.QtWidgets.QWidget()
        self._overview_scroll_area.setWidget( overview_scroll_subwidget )

        if self._configuration.grid_widget_locs is None:
            overview_layout = _OverviewAutoLayout(self._configuration)
        else:
            overview_layout = _OverviewManualLayout(self._configuration)

        overview_scroll_subwidget.setLayout( overview_layout )

        # matches LastFrameVideoCapture output format
        audio_format = PySide6.QtMultimedia.QAudioFormat()
        audio_format.setChannelCount(1)
        audio_format.setSampleRate(48000)
        audio_format.setSampleFormat( PySide6.QtMultimedia.QAudioFormat.SampleFormat.Int16 )

        self._live_view_widgets = []        
        self._annotation_widgets = []
        self._overview_live_view_widgets = []
        self._overview_annotation_widgets = []
        self._cam_id_to_audio_stream_player_dict = dict()

        def on_volume_slider_change( slider_value : int, player : AudioStreamPlayer, live_views : list[LiveView] ):
            volume = slider_value / 100.0
            if player.get_volume() != volume:
                player.set_volume( volume )
                for live_view in live_views:
                    live_view.set_volume( slider_value )
        
        for index, cam_definition in enumerate( self._configuration.cam_definitions ):
            
            audio_stream_player = AudioStreamPlayer(audio_format, self._error_handler)
            self._cam_id_to_audio_stream_player_dict[cam_definition.id] = audio_stream_player

            live_views = [] # will fill this after partial() but the resulting function still refers to the same instance of list
            on_local_volume_slider_change = functools.partial( on_volume_slider_change, player=audio_stream_player, live_views=live_views )

            live_view_widget = LiveView(
                self._configuration,
                self._error_handler,
                initial_pixmap=PySide6.QtGui.QPixmap("surveillance_ui/disconnected.png"),
                on_volume_change=on_local_volume_slider_change,
            )
            live_views.append( live_view_widget )
            self._live_view_widgets.append( live_view_widget )
            self._cams_tab.addTab( live_view_widget, cam_definition.label )

            self._annotation_widgets.append(
                LiveView( 
                    self._configuration,
                    self._error_handler,
                    initial_pixmap=PySide6.QtGui.QPixmap("surveillance_ui/empty.png"),
                ) 
            )
            self._cams_tab.addTab( self._annotation_widgets[-1], cam_definition.label + " ! " )

            def scale_initial_pixmap( initial : PySide6.QtGui.QPixmap ) -> PySide6.QtGui.QPixmap:
                if self._configuration.grid_widget_locs is not None:
                    widget_location = self._configuration.grid_widget_locs[index]
                    aspect_ratio = widget_location.width()/widget_location.height()
                    if aspect_ratio > 1:
                        size = PySide6.QtCore.QSize( initial.height()*aspect_ratio, initial.height() )
                    else:
                        size = PySide6.QtCore.QSize( initial.width(), initial.width()/aspect_ratio )

                    scaled = PySide6.QtGui.QPixmap( size )
                    scaled.fill( PySide6.QtGui.QColorConstants.Black )

                    with PySide6.QtGui.QPainter( scaled ) as painter:
                        painter.drawPixmap(
                            PySide6.QtCore.QPoint(
                                 scaled.width()/2 - initial.width()/2,
                                 scaled.height()/2 - initial.height()/2
                            ), 
                            initial
                        )
                    
                    return scaled
                else:
                    return initial

            overview_cam_widget = LiveView(
                self._configuration,
                self._error_handler,
                initial_pixmap=scale_initial_pixmap(PySide6.QtGui.QPixmap("surveillance_ui/disconnected.png")),
                on_volume_change=on_local_volume_slider_change,
            )
            live_views.append( overview_cam_widget )
            self._overview_live_view_widgets.append(overview_cam_widget)
            overview_annotation_widget = LiveView(
                self._configuration,
                self._error_handler,
                initial_pixmap=scale_initial_pixmap(PySide6.QtGui.QPixmap("surveillance_ui/empty.png")),
            )

            self._overview_annotation_widgets.append(overview_annotation_widget)

        for widget in self._overview_live_view_widgets + self._overview_annotation_widgets:
            overview_layout.addWidget( widget )
        
        self._cams_tab.addTab( self._ignore_list_view, " 👁🚫 " )

        self.setLayout(layout)

        self._signals.detection.connect( self._on_detection )
        self._signals.frame.connect( self._on_frame )

        def on_audio_chunk( chunk : _AudioChunk ):
            self._cam_id_to_audio_stream_player_dict[chunk.cam_id].push( chunk.chunk )
        
        self._detector = _Detector( 
            configuration=self._configuration,
            filter_ignored=self._ignore_list.filter_ignored,
            on_detection=self._signals.detection.emit,
            on_frame=self._signals.frame.emit,
            on_audio_chunk=on_audio_chunk,
            on_uncaught_exception=self._error_handler.report_and_log_error,
        )

        timer = PySide6.QtCore.QTimer(self)
        timer.timeout.connect( self._update_live_view_connection_status )
        timer.start(10)

    def _update_live_view_connection_status( self ):
        for live_view_widget in self._live_view_widgets + self._overview_live_view_widgets:
            live_view_widget.update_connection_status()

    def _on_alert_volume_change( self, value : int ) -> None:
        self._alert_player.set_volume( value/100 )

    def graceful_handler( handler ):
        @functools.wraps( handler )
        def wrapped_handler( self : 'SurveillanceWidget', *args, **kwargs ):
            self._error_handler.handle_gracefully_internal( handler, self, *args, **kwargs )
        return wrapped_handler
    
    def shutdown( self ) -> None:
        shutdown_actions = (
            [self._alert_player.shut_down, self._detector.shut_down] 
            +
            [audio_stream_player.shut_down for audio_stream_player in self._cam_id_to_audio_stream_player_dict.values()]
            +
            [live_view.shut_down for live_view in (self._live_view_widgets + self._annotation_widgets + self._overview_live_view_widgets + self._overview_annotation_widgets)]
            +
            [self._history_view.shut_down, self._ignore_list_view.shut_down]
        )
        shutdown_threads = [threading.Thread(target=action) for action in shutdown_actions]
        for thread in shutdown_threads:
            thread.start()
        for thread in shutdown_threads:
            thread.join()

    @graceful_handler
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

    @graceful_handler
    def _on_detection(self, image_detections_info : _ImageDetectionsInfo ) -> None:
        shape = image_detections_info.frame_info.image.shape
        frame_size = Point2D( shape[1], shape[0] )
        
        fresh_detections : list[SvDetection] = []
        for detection in image_detections_info.detections:
            single_detection_info = ObjectDetectionInfo( cam_id=image_detections_info.frame_info.cam_id, supervision=detection, when=image_detections_info.when, frame_size=frame_size )
            if self._history.is_fresh_detection( single_detection_info ):
                self._history.add( single_detection_info, image=image_detections_info.frame_info.image ) 
                fresh_detections.append(detection)

        if len(fresh_detections) > 0:
            fresh_detection_info = _ImageDetectionsInfo(
                frame_info=image_detections_info.frame_info,
                detections=fresh_detections,
                when=image_detections_info.when
            )
            self._alert_player.try_alert( image_detections_info=fresh_detection_info )
        
        pixmap = self._make_pixmap( image_detections_info.frame_info.image )
        index = self._configuration.cam_definitions.index( self._configuration.get_cam_definition(image_detections_info.frame_info.cam_id) )
        self._annotation_widgets[index].setPixmap( pixmap )
        self._overview_annotation_widgets[index].setPixmap( pixmap )
    
    @graceful_handler
    def _on_frame(self, frame_info : _FrameInfo ) -> None:
        for cam_definition, live_view_cam_widget, overview_cam_widget in zip( self._configuration.cam_definitions, self._live_view_widgets, self._overview_live_view_widgets ):
            if cam_definition.id == frame_info.cam_id:
                pixmap = self._make_pixmap( frame_info.image )
                live_view_cam_widget.setPixmap( pixmap )
                overview_cam_widget.setPixmap( pixmap )
    
    def _make_pixmap(self, frame : numpy.ndarray ):
        q_image = PySide6.QtGui.QImage( frame, frame.shape[1], frame.shape[0], frame.strides[0], PySide6.QtGui.QImage.Format.Format_RGB888)
        return PySide6.QtGui.QPixmap.fromImage(q_image)
    
    def _make_vertical_line(self) -> PySide6.QtWidgets.QFrame:
        retval = PySide6.QtWidgets.QFrame()
        retval.setFrameShape(PySide6.QtWidgets.QFrame.VLine)
        retval.setFrameShadow(PySide6.QtWidgets.QFrame.Sunken)
        return retval
