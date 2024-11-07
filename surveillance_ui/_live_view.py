import time
import typing
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

    _configuration : Configuration

    _fitting_image : FittingImage
    _controls_widget : PySide6.QtWidgets.QWidget
    _volume_slider : PySide6.QtWidgets.QSlider
    _disconnection_indicator : PySide6.QtWidgets.QLabel
    _disconnection_image : PySide6.QtGui.QPixmap

    _last_frame_time_monotonic : float
    
    def __init__( self, configuration : Configuration, error_handler : ErrorHandler, on_volume_change : typing.Callable[[float],None] ):
        super().__init__()

        self._configuration = configuration
        self._last_frame_time_monotonic = None
        self._fitting_image = FittingImage( 5*16, 5*9 , error_handler )
        self._fitting_image.setPixmap( PySide6.QtGui.QPixmap( "surveillance_ui/disconnected.png" ) )

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
        #controls_frame.setMaximumSize( 1000, 50 )
        controls_vertical_layout.addWidget( controls_frame )
        controls_layout = PySide6.QtWidgets.QHBoxLayout()
        controls_frame.setLayout( controls_layout )
        self._volume_slider, volume_label = make_percentage_slider(error_handler, 0)
        if on_volume_change is not None:
            self._volume_slider.valueChanged.connect( on_volume_change )
        controls_layout.setAlignment( PySide6.QtCore.Qt.AlignmentFlag.AlignBottom )
        controls_layout.addStretch(1)
        controls_layout.addWidget( PySide6.QtWidgets.QLabel(text="🔊 ") )
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
    
    def enterEvent( self, event ):
        self._controls_widget.show()
    
    def leaveEvent( self, event ):
        self._controls_widget.hide()
    
    def pixmap(self) -> PySide6.QtGui.QPixmap:
        return self._fitting_image.pixmap()
    
    def setPixmap( self, pixmap : PySide6.QtGui.QPixmap ):
        self._last_frame_time_monotonic = time.monotonic()
        self._fitting_image.setPixmap( pixmap )
        self.update_connection_status()

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
