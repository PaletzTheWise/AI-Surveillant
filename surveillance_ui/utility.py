import numpy
import threading
import queue
import typing
import sys
import dataclasses
import datetime
import traceback
import functools
import av.container.input
import PySide6.QtWidgets
import PySide6.QtGui
import PySide6.QtCore

class LastFrameVideoCapture:
    _input_container_constructor : typing.Callable[[],av.container.input.InputContainer]
    _thread : threading.Thread
    _frame_queue : queue.Queue[numpy.ndarray]
    _on_frame : typing.Callable[[numpy.ndarray],None]
    _shut_down_pending : bool = False

    def __init__(
            self,
            input_container_constructor : typing.Callable[[],av.container.input.InputContainer],
            on_uncaught_exception : typing.Callable[[BaseException],None],
            on_frame : typing.Callable[[numpy.ndarray],None] | None = None ):
        """
        Parameters:
            input_container_constructor - function that creates the video source
            on_frame - callback used by a worker thread to notify about a new frame being available
        """
        self._input_container_constructor = input_container_constructor
        self._on_frame = on_frame
        self._on_uncaught_exception = on_uncaught_exception
        self._frame_queue = queue.Queue(1)
        
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
                if self._shut_down_pending:
                    return                
                input_container = self._input_container_constructor()
                for frame in input_container.decode(video=0):
                    if self._shut_down_pending:
                        return

                    image = frame.to_ndarray(format="rgb24")
                    
                    if self._on_frame is not None:
                        self._on_frame(image)

                    self._update_latest_frame(image)
            except av.FFmpegError as e:
                print( f"Video capture exception: {e}", file=sys.stderr )
    
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
        self._shut_down_pending = True
        self._thread.join()
    
    def _update_latest_frame(self, frame : numpy.ndarray ):
        try:
            self._frame_queue.get_nowait()
        except queue.Empty as _:
            pass

        self._frame_queue.put(frame)

@dataclasses.dataclass
class _UncaughtExceptionInfo:
    exception : BaseException
    context : str

class _ErrorHandlerSignals(PySide6.QtCore.QObject):
    uncaught_exception = PySide6.QtCore.Signal( _UncaughtExceptionInfo )

class ErrorHandler:

    _application : PySide6.QtWidgets.QApplication
    _last_exception_datetime : datetime
    _signals : _ErrorHandlerSignals

    def __init__(self, application : PySide6.QtWidgets.QApplication ):
        self._last_exception_datetime = datetime.datetime.min
        self._application = application
        self._signals = _ErrorHandlerSignals()
        self._signals.uncaught_exception.connect( self._on_uncaught_exception )

    def handle_gracefully( self, handler : typing.Callable, context : str, *args, **kwargs ):
        '''Handle an event "gracefully".

        That is, catch and report any exception.

        This can be used to create a decorator for handlers, assuming handlers can access ErrorHandler through self:
        ```
        def graceful_handler( handler ):
            @functools.wraps( handler )
                def wrapped_handler( self, *args, **kwargs ):
                self._error_handler.handle_gracefully( handler, "Internal error.", self, *args, **kwargs )
            return wrapped_handler
            
        @graceful_handler
        def on_whatever():
            pass
        ```

        Parameters:
            handler - event handler
            context - error context if one occurs, e.g. "File update detection has crashed."
            args, kwargs - event arguments to be passed to handler
        '''
        try:
            handler( *args, **kwargs )
        except BaseException as e: # NOSONAR
            self.error( e, context )
    
    def error( self, exception : BaseException, context : str ):
        self._signals.uncaught_exception.emit( _UncaughtExceptionInfo(exception,  context) )
    
    @staticmethod
    def log_error( exception : BaseException, context : str ):
        with open("errors.txt", "a") as file:
            file.write( f"{datetime.datetime.now()}\n\n{ErrorHandler._format_error_info(exception, context)}\n---\n\n" )

    @staticmethod
    def _format_error_info( exception : BaseException, context : str, limit : int | None = None ) -> str:
        return f"{context}\n\n{'\n'.join(traceback.format_exception(exception, limit = limit ))}"
    
    def _on_uncaught_exception( self, uncaught_exception_info : _UncaughtExceptionInfo ) -> None:
        ErrorHandler.log_error( uncaught_exception_info.exception, uncaught_exception_info.context )

        if (datetime.datetime.now() - self._last_exception_datetime) > datetime.timedelta(seconds=30):
            self._last_exception_datetime = datetime.datetime.now()
            dialog = PySide6.QtWidgets.QMessageBox(self._application)
            dialog.setWindowTitle("Error")
            dialog.setIcon( PySide6.QtWidgets.QMessageBox.Icon.Critical )
            dialog.setStandardButtons( PySide6.QtWidgets.QMessageBox.StandardButton.Ok )
            dialog.setText( f"The application has encountered an unexpected error and may behave erratically going forward.\n\n {ErrorHandler._format_error_info( uncaught_exception_info.exception, uncaught_exception_info.context, 10 )}" )
            dialog.exec()

class FittingImage(PySide6.QtWidgets.QLabel):
    
    _error_handler : ErrorHandler

    def __init__(self, min_size_x : int, min_size_y : int, error_handler : ErrorHandler ):
        super().__init__()
        self._error_handler = error_handler
        self.setScaledContents(True)
        self.setMinimumSize(min_size_x,min_size_y)

    def graceful_handler( handler ):
        @functools.wraps( handler )
        def wrapped_handler( self : 'FittingImage', *args, **kwargs ):
            self._error_handler.handle_gracefully( handler, "Internal error.", self, *args, **kwargs )
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
             return 1

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
            self.setContentsMargins( 0, 0, 0, 0 )
            return
        
        pixmap_aspect_ratio = self.pixmap().size().width() / self.pixmap().size().height()
        widget_aspect_ratio = self.width() / self.height() 

        def calculate_margin_ratio( widget_dimension : float, pixmap_dimension : float ) -> float:
            return max( widget_dimension - pixmap_dimension, 0 ) / (widget_dimension)
        
        horizontal_margin_ratio = calculate_margin_ratio( widget_aspect_ratio, pixmap_aspect_ratio )
        vertical_margin_ratio = calculate_margin_ratio( 1/widget_aspect_ratio, 1/pixmap_aspect_ratio )      
        
        horizontal_half_margin = horizontal_margin_ratio * self.width() * 0.5
        vertical_half_margin = vertical_margin_ratio * self.height() * 0.5

        self.setContentsMargins( horizontal_half_margin, vertical_half_margin, horizontal_half_margin, vertical_half_margin )

_T = typing.TypeVar('T')

class Synchronized(typing.Generic[_T]):

    _value : _T
    _lock : threading.RLock

    def __init__(self, value : _T):
        self._value = value
        self._lock = threading.RLock()
    
    def lock(self) -> "LockContext[_T]":
        return LockContext(self)

class LockContext(typing.Generic[_T]):

    _synchronized : Synchronized[_T]

    def __init__(self, _synchronized : Synchronized[_T] ):
        self._synchronized = _synchronized

    def __enter__(self):
        self._synchronized._lock.acquire()
        return self._synchronized._value

    def __exit__(self, type, value, traceback):
        return self._synchronized._lock.release()
