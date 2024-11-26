import time
import typing
import functools
import PySide6
import PySide6.QtWidgets
import PySide6.QtGui
import PySide6.QtCore

from .common import (
    Configuration,
)
from .utility import (
    FittingImage,
    ErrorHandler,
    make_percentage_slider,
)

class LiveView(PySide6.QtWidgets.QWidget):

    # image_offset = QPointF where coordinates of the full image range from -0.5 to 0.5. ( -0.5; -0.5) corresponds to pixel ( 0; 0),  ( 0.5; 0.5) corresponds to pixel ( width; height).
    # view_offset = QPointF where coordinates of the currently displayed view of the image range from -0.5 to 0.5.

    _configuration : Configuration
    _error_handler : ErrorHandler

    _full_image : PySide6.QtGui.QPixmap
    _zoom_level : int
    _focus_image_offset : PySide6.QtCore.QPointF
    _fitting_image : FittingImage
    _controls_widget : PySide6.QtWidgets.QWidget
    _volume_slider : PySide6.QtWidgets.QSlider
    _disconnection_indicator : PySide6.QtWidgets.QLabel
    _disconnection_image : PySide6.QtGui.QPixmap

    _last_frame_time_monotonic : float

    _drag_pivot_image_offset : PySide6.QtCore.QPointF
    _drag_timer : PySide6.QtCore.QTimer
    
    def __init__( self, configuration : Configuration, error_handler : ErrorHandler, initial_pixmap : PySide6.QtGui.QPixmap, on_volume_change : typing.Callable[[float],None] = None ):
        super().__init__()

        self._configuration = configuration
        self._error_handler = error_handler

        self._last_frame_time_monotonic = None
        self._drag_pivot_image_offset = None
        self._drag_timer = None

        self._fitting_image = FittingImage( 5*16, 5*9 , self._error_handler )

        layout = PySide6.QtWidgets.QStackedLayout()
        layout.setStackingMode( PySide6.QtWidgets.QStackedLayout.StackingMode.StackAll )
        self.setLayout(layout)
        layout.addWidget( self._fitting_image )
        
        # align controls bottom
        self._controls_widget = PySide6.QtWidgets.QWidget()
        layout.addWidget( self._controls_widget )
        self._controls_widget.hide()
        self._controls_widget.setSizePolicy( PySide6.QtWidgets.QSizePolicy.Policy.Expanding, PySide6.QtWidgets.QSizePolicy.Policy.Expanding )
        self._controls_widget.raise_()
        controls_vertical_layout = PySide6.QtWidgets.QVBoxLayout()
        self._controls_widget.setLayout( controls_vertical_layout )
        controls_vertical_layout.addStretch()
        
        # frame so that the image does not camouflage controls
        controls_frame = PySide6.QtWidgets.QFrame()
        controls_frame.setAutoFillBackground(True)
        controls_frame.setContentsMargins( 0, 0, 0, 0 )
        controls_frame.setSizePolicy( PySide6.QtWidgets.QSizePolicy.Policy.Expanding, PySide6.QtWidgets.QSizePolicy.Policy.Preferred )
        controls_vertical_layout.addWidget( controls_frame )
        controls_layout = PySide6.QtWidgets.QHBoxLayout()
        controls_frame.setLayout( controls_layout )
        controls_layout.setAlignment( PySide6.QtCore.Qt.AlignmentFlag.AlignBottom )
        controls_layout.addStretch(1)

        def set_font_size(widget : PySide6.QtWidgets.QWidget):
            font = widget.font()
            font.setPointSize(16)
            widget.setFont(font)
        
        def add_arrow( vector, text ):
            button = PySide6.QtWidgets.QPushButton( text=text )
            set_font_size( button )
            button.pressed.connect( lambda: self._nudge( vector ) )
            controls_layout.addWidget( button )

        add_arrow( PySide6.QtCore.QPointF(-1/5, 0 ), " ← " )
        add_arrow( PySide6.QtCore.QPointF( 1/5, 0 ), " → " )
        add_arrow( PySide6.QtCore.QPointF( 0,-1/5 ), " ↑ " )
        add_arrow( PySide6.QtCore.QPointF( 0, 1/5 ), " ↓ " )
        
        def add_magnifier( delta, text ):
            button = PySide6.QtWidgets.QPushButton( text=text )
            set_font_size( button )
            button.pressed.connect( lambda: self._zoom( delta ) )
            controls_layout.addWidget( button )
        
        add_magnifier( +1, " + " )
        add_magnifier( -1, " - " )
        
        if on_volume_change is not None:
            self._volume_slider, volume_label = make_percentage_slider(self._error_handler, 0)
            set_font_size( volume_label )
            self._volume_slider.valueChanged.connect( on_volume_change )
            icon_label = PySide6.QtWidgets.QLabel(text="🔊 ")
            set_font_size( icon_label )
            controls_layout.addWidget( icon_label )        
            controls_layout.addWidget( self._volume_slider )
            controls_layout.addWidget( volume_label )
        controls_layout.addStretch(1)

        self._disconnection_indicator = PySide6.QtWidgets.QLabel()
        self._disconnection_image = PySide6.QtGui.QPixmap( "surveillance_ui/disconnected_icon.png" )
        self._disconnection_indicator.setSizePolicy( PySide6.QtWidgets.QSizePolicy.Policy.Fixed, PySide6.QtWidgets.QSizePolicy.Policy.Fixed )
        self._disconnection_indicator.setAlignment( PySide6.QtCore.Qt.AlignmentFlag.AlignCenter )
        layout.addWidget( self._disconnection_indicator )
        self._disconnection_indicator.raise_()
        self.update_connection_status()

        self._fitting_image.marginsChanged.connect( self._controls_widget.setContentsMargins )

        self._zoom_level = 0
        self._focus_image_offset = PySide6.QtCore.QPointF()
        self.setPixmap( initial_pixmap )
    
    def shut_down( self ) -> None:
        if self._drag_timer is not None:
            self._drag_timer.stop()
            self._drag_timer = None
    
    def graceful_handler( handler ):
        @functools.wraps( handler )
        def wrapped_handler( self : 'LiveView', *args, **kwargs ):
            self._error_handler.handle_gracefully_internal( handler, self, *args, **kwargs )
        return wrapped_handler

    @graceful_handler
    def enterEvent( self, event ):
        self._controls_widget.show()
    
    @graceful_handler
    def leaveEvent( self, event ):
        self._controls_widget.hide()
    
    @graceful_handler
    def mousePressEvent( self, event : PySide6.QtGui.QMouseEvent ) -> None:
        if event.button() != PySide6.QtCore.Qt.MouseButton.LeftButton:
            return

        cursor_view_offset = self._widget_pos_to_view_offset( event.position() )
        self._drag_pivot_image_offset = self._view_offset_to_image_offset( cursor_view_offset )
        if self._drag_timer is None:
            self._drag_timer = PySide6.QtCore.QTimer()
            self._drag_timer.timeout.connect( self._drag_update )
            self._drag_timer.setInterval(20)
            self._drag_timer.start()

    @graceful_handler
    def _drag_update( self ) -> None:
        cursor_global_position = PySide6.QtGui.QCursor.pos()
        cursor_widget_position = self.mapFromGlobal( cursor_global_position )
        cursor_view_offset = self._widget_pos_to_view_offset( cursor_widget_position )
        cursor_image_offset = self._view_offset_to_image_offset( cursor_view_offset )

        self._focus_image_offset -= (cursor_image_offset - self._drag_pivot_image_offset)
        self._focus_image_offset = self._clamp_focus_image_offest( self._focus_image_offset, self._zoom_level )
        self._apply_full_image()

    @graceful_handler
    def mouseReleaseEvent( self, event : PySide6.QtGui.QMouseEvent ) -> None:
        if event.button() != PySide6.QtCore.Qt.MouseButton.LeftButton:
            return


        self._drag_pivot_image_offset = None
        if self._drag_timer is not None:
            self._drag_timer.stop()
            self._drag_timer = None

    def pixmap(self) -> PySide6.QtGui.QPixmap:
        return self._fitting_image.pixmap()
    
    def setPixmap( self, pixmap : PySide6.QtGui.QPixmap ):
        self._last_frame_time_monotonic = time.monotonic()
        self._full_image = pixmap
        self._apply_full_image()
        self.update_connection_status()

    def _get_magnification(self, zoom_level_difference) -> float:
        ''' Get magnification based on zoom level difference which may be negative.'''
        return pow( 1.2, zoom_level_difference )
    
    def _magnify(self, value : float | PySide6.QtCore.QPointF, zoom_level_difference ) -> float | PySide6.QtCore.QPointF:
        magnification = self._get_magnification( zoom_level_difference )
        return value*magnification
    
    def _get_tranformation_matrix(self) -> PySide6.QtGui.QTransform:
        t = PySide6.QtGui.QTransform
        size = PySide6.QtCore.QPoint( self._full_image.width(), self._full_image.height() )
        magnification = self._get_magnification( self._zoom_level )
        
        # zoom
        matrix = (
            t.fromTranslate( size.x() * -0.5, size.y() * -0.5 ) # center to (0,0) for scaling
            *
            t.fromScale( magnification, magnification )
        )
        matrix *= t.fromTranslate( size.x() * 0.5, size.y() * 0.5 ) # image center to center of window
        size = size * magnification

        # focus
        matrix *= t.fromTranslate( size.x() * self._focus_image_offset.x() * -1, size.y() * self._focus_image_offset.y() * -1 )

        return matrix

    def _apply_full_image(self) -> None:
        zoomed_image = PySide6.QtGui.QPixmap( self._full_image.size() )
        zoomed_image.fill( PySide6.QtGui.QColorConstants.Gray )
        with PySide6.QtGui.QPainter( zoomed_image ) as painter:
            painter.setTransform( self._get_tranformation_matrix() )
            painter.drawPixmap( PySide6.QtCore.QPoint(), self._full_image )
        self._fitting_image.setPixmap( zoomed_image )

    def set_volume( self, volume : int ) -> None:
        """
        Set volume

        Triggers the volume callback passed to the constructor.

        parameters:
            volume - 0-100
        """
        self._volume_slider.setValue( volume )
    
    def heightMatchingAspect( self ) -> int:
        return self._fitting_image.heightMatchingAspect()
    
    def update_connection_status( self ) -> None:
        delay_seconds = self._configuration.get_disconnect_indicator_delay().total_seconds()
        period_seconds = 2.0
        hidden_ratio_seconds = 0.4 # ratio of period during which the indicator overlay should be hidden so the user can actually see the last frame unobstructed

        if self._last_frame_time_monotonic is None :
            self._set_overlay_opacity( 0 )
            return
        
        time_since_frame_seconds = time.monotonic() - self._last_frame_time_monotonic
        if time_since_frame_seconds < delay_seconds:
            self._set_overlay_opacity( 0 )
            return
        
        cycle = abs( (time_since_frame_seconds - delay_seconds) % period_seconds ) / period_seconds # this one goes only up from 0.0 to 1.0
        cycle = (cycle + hidden_ratio_seconds/2) % 1.0 # skip the hidden ratio when going up (/2 because of the 2* below)
        cycle = 2*cycle if cycle < 0.5 else 2*(1-cycle) # this one goes up and down
        if cycle < hidden_ratio_seconds:
            self._set_overlay_opacity( 0 )
            return
        
        self._set_overlay_opacity( (cycle - hidden_ratio_seconds) * 1/(1-hidden_ratio_seconds) )

    def _set_overlay_opacity( self, opacity : float ) -> None:
        if opacity == 0:
            self._disconnection_indicator.hide()
        else:
            self._disconnection_indicator.show()
            transparenced_image = PySide6.QtGui.QPixmap( self._disconnection_image.size() )
            transparenced_image.fill( PySide6.QtCore.Qt.GlobalColor.transparent )
            painter = PySide6.QtGui.QPainter()
            painter.begin(transparenced_image)
            painter.setOpacity( opacity )
            painter.drawPixmap(0, 0, self._disconnection_image)
            painter.end()
            self._disconnection_indicator.setPixmap( transparenced_image )

    @graceful_handler
    def wheelEvent( self, event : PySide6.QtGui.QWheelEvent ) -> None:
        event.accept()

        cursor_view_offset = self._widget_pos_to_view_offset( event.position() )
        cursor_image_offset = self._view_offset_to_image_offset( cursor_view_offset )

        if event.angleDelta().y() > 0:
            
            self._zoom_level += 1
            self._focus_image_offset = cursor_image_offset - self._magnify( cursor_view_offset, -self._zoom_level )
            
        elif event.angleDelta().y() < 0:
            if self._zoom_level == 0:
                return # ignore

            self._zoom_level -= 1
            self._focus_image_offset = cursor_image_offset - self._magnify( cursor_view_offset, -self._zoom_level )
            self._focus_image_offset = self._clamp_focus_image_offest( self._focus_image_offset, self._zoom_level )
        
        self._apply_full_image()

    @graceful_handler
    def _nudge( self, vector : PySide6.QtCore.QPointF ) -> None:
        vector = self._magnify( vector, -self._zoom_level )
        self._focus_image_offset += vector
        self._focus_image_offset = self._clamp_focus_image_offest( self._focus_image_offset, self._zoom_level )
        self._apply_full_image()
    
    @graceful_handler
    def _zoom( self, delta ) -> None:
        self._zoom_level = max( 0, self._zoom_level + delta  )
        self._apply_full_image()

    def _widget_pos_to_view_offset( self, widget_position : PySide6.QtCore.QPointF ) -> PySide6.QtCore.QPointF:
        return PySide6.QtCore.QPointF( (widget_position.x() / self.size().width() ) - 0.5,
                                       (widget_position.y() / self.size().height()) - 0.5 )

    def _view_offset_to_image_offset( self, view_offset : PySide6.QtCore.QPointF ) -> PySide6.QtCore.QPointF:
        return self._focus_image_offset + self._magnify( view_offset, -self._zoom_level )
    
    def _clamp_focus_image_offest( self, offset : PySide6.QtCore.QPointF, zoom_level ) -> PySide6.QtCore.QPointF:
        ''' Clamp focus image offset so that the view fits in the image.'''
        def max_focus_offset(zoom_level) -> float:
            return (
                0.5 # focus with infinite zoom could be at the edge
                - self._magnify( 0.5, -zoom_level ) # subtract half of the zoomed window
            )

        max_value = max_focus_offset(zoom_level)
        def clamp(value):
            value = min( max_value, value )
            value = max( -max_value, value )
            return value
        return PySide6.QtCore.QPointF( clamp(offset.x()), clamp(offset.y()) )