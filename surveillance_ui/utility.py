import numpy
import threading
import queue
import av.container.input
import typing
import PySide6.QtWidgets
import PySide6.QtGui

class LastFrameVideoCapture:
    _input_container_constructor : typing.Callable[[],av.container.input.InputContainer]
    _thread : threading.Thread
    _frame_queue : queue.Queue[numpy.ndarray]
    _on_frame : typing.Callable[[numpy.ndarray],None]
    _shut_down_pending : bool = False

    def __init__(self, input_container_constructor : typing.Callable[[],av.container.input.InputContainer], on_frame : typing.Callable[[numpy.ndarray],None] | None = None):
        """
        Parameters:
            input_container_constructor - function that creates the video source
            on_frame - callback used by a worker thread to notify about a new frame being available
        """
        self._input_container_constructor = input_container_constructor
        self._on_frame = on_frame
        self._frame_queue = queue.Queue(1)
        self._thread = threading.Thread(target=self._frame_pulling_process)
        self._thread.daemon = True
        self._thread.start()
    
    def _frame_pulling_process(self):
        input_container = self._input_container_constructor()

        for frame in input_container.decode(video=0):
            if self._shut_down_pending:
                return

            image = frame.to_ndarray(format="rgb24")
            
            if self._on_frame is not None:
                self._on_frame(image)

            try:
                self._frame_queue.get_nowait()
            except queue.Empty as _:
                pass

            self._frame_queue.put(image)
    
    def get_latest_frame(self, timeout=float) -> numpy.ndarray:
        """
        Read the latest frame
        
        The on_frame callback will return the same frame instance, so modification of the frame data may be perilous.

        Returns: frame as 24bit RGB ndarray
        """
        try:
            return self._frame_queue.get(timeout=timeout) # TODO error handling NYI
        except queue.Empty:
            return None
    
    def shut_down(self) -> None:
        self._shut_down_pending = True
        self._thread.join()

class FittingImage(PySide6.QtWidgets.QLabel):
    
    def __init__(self, min_size_x : int, min_size_y : int):
        super().__init__()
        self.setScaledContents(True)
        self.setMinimumSize(min_size_x,min_size_y)

    def setPixmap( self, pixmap : PySide6.QtGui.QPixmap ) -> None:
        super().setPixmap(pixmap)
        self._updateMargins()
    
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