import PySide6.QtMultimedia
import av.audio
import av.video
import numpy
import threading
import queue
import typing
import sys
import dataclasses
import time
import datetime
import functools
import av.container.input
import PySide6.QtWidgets
import PySide6.QtGui
import PySide6.QtCore
from .error_handler import (
    ErrorHandler as _ErrorHandler
)

class LastFrameVideoCapture:
    _input_container_constructor : typing.Callable[[],av.container.input.InputContainer]
    _thread : threading.Thread
    _frame_queue : queue.Queue[numpy.ndarray]
    _on_frame : typing.Callable[[numpy.ndarray],None]
    _on_audio_bytes : typing.Callable[[bytes],None]
    _shutdown_pending : bool = False
    _resampler : av.AudioResampler

    def __init__(
            self,
            input_container_constructor : typing.Callable[[],av.container.input.InputContainer],
            on_uncaught_exception : typing.Callable[[BaseException],None],
            on_frame : typing.Callable[[numpy.ndarray],None] | None = None,
            on_audio_bytes : typing.Callable[[bytes],None] | None = None ):
        """
        Parameters:
            input_container_constructor - function that creates the video source
            on_frame - callback used by a worker thread to notify about a new frame being available as 24bit RGB
            on_audio_bytes - callback used by a worker thread to notify about new audio bytes in mono 48kHz 16-bit
        """
        self._input_container_constructor = input_container_constructor
        self._on_frame = on_frame
        self._on_audio_bytes = on_audio_bytes
        self._on_uncaught_exception = on_uncaught_exception
        self._frame_queue = queue.Queue(1)
        
        self._resampler = av.AudioResampler(
            format=av.AudioFormat("s16p"),
            layout='mono',
            rate=48000,
        )

        def graceful_frame_pulling_process():
            try:
                self._frame_pulling_process()
            except BaseException as e: # NOSONAR
                on_uncaught_exception(e)
        
        self._thread = threading.Thread( target=graceful_frame_pulling_process )
        self._thread.daemon = True
        self._thread.start()
    
    def _frame_pulling_process(self):

        while True:
            try:
                if self._shutdown_pending:
                    return                
                input_container = self._input_container_constructor()
                audio_channel_count = len( input_container.streams.audio )
                
                if audio_channel_count > 0:
                    frame_iterator = input_container.decode( audio=0, video=0 )
                else:
                    frame_iterator = input_container.decode( video=0 )
                
                for frame in frame_iterator:
                    if self._shutdown_pending:
                        return
                    self._process_frame(frame)
            except av.FFmpegError as e:
                print( f"Video capture exception: {e}", file=sys.stderr )
                time.sleep(0.5) # Limit the retry speed so that a misconfigured cam doesn't eat too many resources.
    
    def _process_frame( self, frame : av.video.frame.VideoFrame | av.audio.frame.AudioFrame ) -> None:
        if isinstance( frame, av.video.frame.VideoFrame ):
            image = frame.to_ndarray(format="rgb24")
            
            if self._on_frame is not None:
                self._on_frame(image)

            self._update_latest_frame(image)
        else:
            if self._on_audio_bytes is not None:
                for audio_frame in self._resampler.resample( frame ):
                    self._on_audio_bytes( audio_frame.to_ndarray().tobytes() )
    
    def get_latest_frame(self, timeout=float) -> numpy.ndarray:
        """
        Read the latest frame
        
        The on_frame callback will return the same frame instance, so modification of the frame data may be perilous.

        Returns: frame as 24bit RGB ndarray
        """
        try:
            return self._frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def shut_down(self) -> None:
        self._shutdown_pending = True
        self._thread.join()
    
    def _update_latest_frame(self, frame : numpy.ndarray ):
        try:
            self._frame_queue.get_nowait()
        except queue.Empty as _:
            pass

        self._frame_queue.put(frame)

class FittingImage(PySide6.QtWidgets.QLabel):
    
    _error_handler : _ErrorHandler
    marginsChanged = PySide6.QtCore.Signal( PySide6.QtCore.QMargins )

    def __init__(self, min_size_x : int, min_size_y : int, error_handler : _ErrorHandler ):
        super().__init__()
        self._error_handler = error_handler
        self.setScaledContents(True)
        self.setMinimumSize(min_size_x,min_size_y)

    def graceful_handler( handler ):
        @functools.wraps( handler )
        def wrapped_handler( self : 'FittingImage', *args, **kwargs ):
            self._error_handler.handle_gracefully_internal( handler, self, *args, **kwargs )
        return wrapped_handler

    def setPixmap( self, pixmap : PySide6.QtGui.QPixmap ) -> None:
        super().setPixmap(pixmap)
        self._updateMargins()
    
    @graceful_handler
    def resizeEvent( self, event : PySide6.QtGui.QResizeEvent ) -> None:
        super().resizeEvent( event )
        self._updateMargins()

    def heightMatchingAspect(self) -> int:
        if not self._are_size_data_available():
             return self.minimumSize().height()

        pixmap_aspect_ratio = self.pixmap().size().width() / self.pixmap().size().height()
        width = self.size().width()

        return int(width / pixmap_aspect_ratio)

    def _are_size_data_available(self) -> bool:
        return (
            self.pixmap() is not None
            and
            all( [size.height() > 0 or size.width() > 0 for size in [self.pixmap().size(), self.size()]])
        )

    def _updateMargins( self ):
        if not self._are_size_data_available():
            self._setSymetricMargins( 0, 0 )
            return
        
        pixmap_aspect_ratio = self.pixmap().size().width() / self.pixmap().size().height()
        widget_aspect_ratio = self.width() / self.height() 

        def calculate_margin_ratio( widget_dimension : float, pixmap_dimension : float ) -> float:
            return max( widget_dimension - pixmap_dimension, 0 ) / (widget_dimension)
        
        horizontal_margin_ratio = calculate_margin_ratio( widget_aspect_ratio, pixmap_aspect_ratio )
        vertical_margin_ratio = calculate_margin_ratio( 1/widget_aspect_ratio, 1/pixmap_aspect_ratio )      
        
        horizontal_half_margin = horizontal_margin_ratio * self.width() * 0.5
        vertical_half_margin = vertical_margin_ratio * self.height() * 0.5

        self._setSymetricMargins( horizontal_half_margin, vertical_half_margin )
    
    def _setSymetricMargins( self, horizontal_half_margins : int, vertical_half_margins : int ) -> None:
        new_margins = PySide6.QtCore.QMargins( horizontal_half_margins, vertical_half_margins, horizontal_half_margins, vertical_half_margins )
        if self.contentsMargins() != new_margins:
            self.setContentsMargins( new_margins )
            self.marginsChanged.emit( new_margins )

_T = typing.TypeVar('T')

class EventDispatcher(typing.Generic[_T]):

    _listeners : list[typing.Callable[[_T],None]]

    def register( self, listener : typing.Callable[[_T],None] ) -> None:
        self._listeners = list()
        self._listeners.append( listener )

    def forget( self, listener : typing.Callable[[_T],None] ) -> None:
        self._listeners.remove( listener )

    def fire( self, event : _T ) -> None:
        for listener in self._listeners:
            listener(event)

@dataclasses.dataclass
class _SoundChunk:
    data : bytes

@dataclasses.dataclass
class _VolumeUpdate:
    volume : float

class _AudioStreamPlayerWorker(PySide6.QtCore.QRunnable):

    _format : PySide6.QtMultimedia.QAudioFormat
    _sound_data_queue : queue.Queue[_SoundChunk | _VolumeUpdate]
    _shutdown_pending : bool = False
    _error_handler : _ErrorHandler
    _target_delay_us : int
    _delay_tolerance_us : int

    def __init__( self,
                  format : PySide6.QtMultimedia.QAudioFormat,
                  error_handler : _ErrorHandler,
                  target_delay : datetime.timedelta,
                  delay_tolerance : datetime.timedelta ):
        self._format = format
        self._target_delay_us = int(target_delay.total_seconds() * pow(10,6))
        self._delay_tolerance_us = int(delay_tolerance.total_seconds() * pow(10,6))
        self._sound_data_queue = queue.Queue()
        self._error_handler = error_handler
        super().__init__()
    
    def graceful_handler( handler ):
        @functools.wraps( handler )
        def wrapped_handler( self : '_AudioStreamPlayerWorker', *args, **kwargs ):
            self._error_handler.handle_gracefully_internal( handler, self, *args, **kwargs )
        return wrapped_handler

    @graceful_handler
    def run(self):
        output_sink = PySide6.QtMultimedia.QAudioSink( PySide6.QtMultimedia.QAudioDevice(), self._format )
        output_sink.setVolume(0.0)
        output_sink.setBufferSize( self._format.bytesForDuration( self._target_delay_us + 3*self._delay_tolerance_us ) )
        output_device = output_sink.start()
        
        while True:

            if self._shutdown_pending:
                return
            
            try:
                audio_data = self._sound_data_queue.get( block=True, timeout=0.1 )
            except queue.Empty:
                continue
            
            if isinstance( audio_data, _VolumeUpdate ):
                output_sink.setVolume(audio_data.volume)
            else:
                assert isinstance( audio_data, _SoundChunk )
                bytes_buffered = output_sink.bufferSize() - output_sink.bytesFree()
                usecs_buffered = self._format.durationForBytes( bytes_buffered )
                # technically, we have a whole new packet to add, and that would give us different usecs_buffered but
                # it would complicate the math a lot to think about it

                if usecs_buffered > self._target_delay_us + self._delay_tolerance_us:
                    # skip for now, TODO speed up playback
                    pass
                elif usecs_buffered < self._target_delay_us - self._delay_tolerance_us:
                    # add for now, TODO slow down playback
                    self._write_data( output_device, audio_data.data )
                else:
                    self._write_data( output_device, audio_data.data )

    def _write_data( self, device : PySide6.QtCore.QIODevice, data : bytes ) -> None:
        while data:
            written = device.write(data)
            if written:
                data = data[written:]
            else:
                print( "did not accept data" )
                time.sleep(0.01)

    def push(self, chunk : bytes):
        self._sound_data_queue.put( _SoundChunk(chunk) )

    def set_volume(self, volume : float):
        self._sound_data_queue.put( _VolumeUpdate(volume) )

    def shutdown(self):
        self._shutdown_pending = True
    
class AudioStreamPlayer:

    _worker : _AudioStreamPlayerWorker
    _pool : PySide6.QtCore.QThreadPool
    _volume : int

    def __init__( self,
                  format : PySide6.QtMultimedia.QAudioFormat,
                  error_handler : _ErrorHandler,
                  target_delay : datetime.timedelta = datetime.timedelta( seconds=0.2 ),
                  delay_tolerance : datetime.timedelta = datetime.timedelta( seconds=0.1 ) ):
        self._volume = 0
        self._worker = _AudioStreamPlayerWorker( format, error_handler, target_delay=target_delay, delay_tolerance=delay_tolerance )
        self._pool = PySide6.QtCore.QThreadPool()
        self._pool.start( self._worker )
        
    def push(self, chunk : bytes) -> None:
        self._worker.push(chunk)
    
    def set_volume(self, volume : float) -> None:
        self._volume = volume
        self._worker.set_volume(volume)

    def get_volume(self) -> int:
        return self._volume

    def shut_down(self):
        self._worker.shutdown()
        self._pool.waitForDone()

def make_percentage_slider( error_handdler : _ErrorHandler, initial_value : int, disable_mouse_wheel : bool = False ) -> tuple[PySide6.QtWidgets.QSlider, PySide6.QtWidgets.QLabel]:
    class Slider( PySide6.QtWidgets.QSlider ):
        _disable_mouse_wheel : bool
        def __init__( self, disable_mouse_wheel : bool ):
            super().__init__(PySide6.QtCore.Qt.Orientation.Horizontal)
            self._disable_mouse_wheel = disable_mouse_wheel
        
        def wheelEvent( self, event : PySide6.QtGui.QWheelEvent ) -> None:
            if self._disable_mouse_wheel:
                event.ignore()
            else:
                super().wheelEvent( event )
    
    slider = Slider( disable_mouse_wheel )
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
        error_handdler.handle_gracefully_internal(
            lambda: percentage.setText(f"{slider.value()} %")
        )

    slider.valueChanged.connect( update_percentage )
    
    return slider, percentage
