import typing
import threading
import importlib
import sys
import functools

import PySide6.QtCore
import PySide6.QtWidgets
import PySide6.QtGui

from .interface import (
    Interest,
    CamDefinition,
    DetectionLogic,
    Configuration,
)
from .synchronized import (
    Synchronized as _Synchronized
)
from .error_handler import (
    ErrorHandler as _ErrorHandler
)

class _SurveilanceWidgetInterface(typing.Protocol):
    def shutdown(self) -> None:
        """
        Gracefully shut down all threads.
        """
class SurveillanceWindow(PySide6.QtWidgets.QMainWindow):
    
    _configuration : Configuration
    _widget : _Synchronized[_SurveilanceWidgetInterface]
    _shutdown_pending : bool = False
    _preload_thread : threading.Thread
    _preloaded_signal = PySide6.QtCore.Signal()
 
    def __init__(self, configuration : Configuration ):
        """
        Params:
            cam_definitions - cams to be surveilled
            interests - objects of interest to be detected on cams
            detection_logic - detection algorithm to be used, this object will be used by a worker thread
        """
        super().__init__()
        
        self._configuration = configuration
        
        self._error_handler = _ErrorHandler(self)
        
        self._widget = _Synchronized(None)
        self._preloaded_signal.connect( self._on_preloaded )

        self._preload_thread = threading.Thread( target=self._pre_load )
        self._preload_thread.daemon = True
        self._preload_thread.start()
        
        self.setWindowTitle("AI Surveillant")
        self.showMaximized()

    def graceful_handler( handler ):
        @functools.wraps( handler )
        def wrapped_handler( self : 'SurveillanceWindow', *args, **kwargs ):
            self._error_handler.handle_gracefully_internal( handler, self, *args, **kwargs )
        return wrapped_handler
    
    @graceful_handler
    def _pre_load(self):
        modules = [
            "PySide6.QtMultimedia",
            "numpy",
            "PIL",
            "av",            
            "supervision",
        ]
        
        for module in modules:
            if self._shutdown_pending:
                return
            importlib.import_module(module)
        
        self._preloaded_signal.emit()

    @graceful_handler
    def _on_preloaded(self) -> None:
        if not self._shutdown_pending:
            from ._application import SurveillanceWidget
            widget = SurveillanceWidget(configuration=self._configuration)
            self._widget.set( widget )
            self.setCentralWidget( widget )

    def closeEvent( self, event: PySide6.QtGui.QCloseEvent ) -> None:
        self._shutdown_pending = True
        self._preload_thread.join()

        with self._widget.lock() as widget:
            if widget is not None:
                widget.shutdown()

        super().closeEvent(event)


def run_surveillance_application( configuration : Configuration ) -> int:
    """
    Run surveillance application.

    Params:
        cam_definitions - cams to be surveilled
        interests - objects of interest to be detected on cams
        detection_logic - detection algorithm to be used, this object will be used by a worker thread
    
    Returns: Application exit code
    """
    try:
        app = PySide6.QtWidgets.QApplication(sys.argv)
        window = SurveillanceWindow( configuration )
        window.show()
        return app.exec()
    except BaseException as e:
        _ErrorHandler.log_error(e, "The application has failed to start.")
        raise
