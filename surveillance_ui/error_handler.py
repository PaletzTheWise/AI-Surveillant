import typing
import sys
import dataclasses
import datetime
import traceback
import threading
import PySide6.QtWidgets
import PySide6.QtGui
import PySide6.QtCore

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
    _local : threading.local

    def __init__(self, application : PySide6.QtWidgets.QApplication ):
        self._last_exception_datetime = datetime.datetime.min
        self._application = application
        self._local = threading.local()

        self._signals = _ErrorHandlerSignals()
        self._signals.uncaught_exception.connect( self._on_uncaught_exception )
        sys.excepthook = self.excepthook

    def handle_gracefully_internal( self, handler : typing.Callable, *args, **kwargs ):
        self.handle_gracefully( handler, "Internal error.", *args, **kwargs)
    
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
            self._local.exception_handler_count = 1 + getattr( self._local, "exception_handler_count", 0 )
            handler( *args, **kwargs )
        except RecursionError as e:
            if self._local.exception_handler_count > 1:
                raise # kick this up so that we don't run out of stack again while reporting it
            else:
                self._signals.uncaught_exception.emit( _UncaughtExceptionInfo(e,  context) )
        except BaseException as e: # NOSONAR
            self._signals.uncaught_exception.emit( _UncaughtExceptionInfo(e,  context) )
        finally:
            self._local.exception_handler_count -= 1
    
    def report_and_log_error( self, exception : BaseException, context : str ):
        self._signals.uncaught_exception.emit( _UncaughtExceptionInfo(exception,  context) )
    
    @staticmethod
    def excepthook(type, value, traceback):
        ErrorHandler.log_error( value, "Uncaught exception." )
        PySide6.QtCore.QCoreApplication.instance().exit(-1)
        sys.__excepthook__(type,value,traceback)

    @staticmethod
    def log_error( exception : BaseException, context : str ):
        with open("errors.txt", "a", encoding="utf-8") as file:
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
