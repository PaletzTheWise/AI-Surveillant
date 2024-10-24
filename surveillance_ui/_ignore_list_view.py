import functools
import typing
from .common import (
    Configuration,
    Point2D,
    _IgnorePoint,
) 
from .utility import (
    FittingImage,
    ErrorHandler,
)
from ._ignore_list import (
    IgnoreList
)
import PySide6.QtWidgets
import PySide6.QtGui
import PySide6.QtCore

class IgnoreListView(PySide6.QtWidgets.QFrame):
    _PREVIEW_SIZE = Point2D( 640, 360 )

    _ignore_list : IgnoreList
    _configuration : Configuration
    _error_handler : ErrorHandler
    _get_cam_image : typing.Callable[[int], PySide6.QtGui.QPixmap]

    _ignore_list_widget : PySide6.QtWidgets.QTreeWidget
    _ignore_item_display : FittingImage

    def __init__( self,
                  ignore_list : IgnoreList,
                  configuration : Configuration,
                  error_handler : ErrorHandler,
                  get_cam_image : typing.Callable[[int], PySide6.QtGui.QPixmap] ):
        super().__init__()
        self._ignore_list = ignore_list
        self._configuration = configuration
        self._error_handler = error_handler
        self._get_cam_image = get_cam_image

        ignore_list_layout = PySide6.QtWidgets.QHBoxLayout()
        self.setLayout(ignore_list_layout)
        self._ignore_list_widget = PySide6.QtWidgets.QTreeWidget()
        self._ignore_list_widget.setSizePolicy( PySide6.QtWidgets.QSizePolicy.Policy.Expanding, PySide6.QtWidgets.QSizePolicy.Policy.Expanding )
        self._ignore_list_widget.setColumnCount( 4 )
        self._ignore_list_widget.setHeaderHidden(  True )
        self._ignore_list_widget.setSelectionMode( PySide6.QtWidgets.QListWidget.SelectionMode.SingleSelection )
        self._ignore_list_widget.setSelectionBehavior( PySide6.QtWidgets.QListWidget.SelectionBehavior.SelectRows )
        ignore_list_layout.addWidget( self._ignore_list_widget )
        ignore_list_layout.setStretch( 0, 100)
        self._ignore_item_display = FittingImage( 50, 50, self._error_handler )
        ignore_list_layout.addWidget( self._ignore_item_display )
        ignore_list_layout.setStretch( 1, 1)
        self._ignore_list_widget.currentItemChanged.connect( lambda: self._on_current_item_change() )
        self._ignore_list_widget.model().rowsInserted.connect( lambda: self._adjust_list_view_width() )       
        self._ignore_list_widget.model().rowsRemoved.connect( lambda: self._adjust_list_view_width() )

        for ignore_point in self._ignore_list.get_ignore_points():
            self._append( ignore_point )
        
        self._ignore_list._added_dispatcher.register( self._append )
        self._ignore_list._removed_dispatcher.register( self._remove )

    def graceful_handler( handler ):
        @functools.wraps( handler )
        def wrapped_handler( self : 'IgnoreListView', *args, **kwargs ):
            self._error_handler.handle_gracefully( handler, "Internal error.", self, *args, **kwargs )
        return wrapped_handler

    def _set_item_ignore_point( self, item_widget : PySide6.QtWidgets.QTreeWidgetItem, ignore_point : _IgnorePoint ) -> None:
        item_widget.user_ignore_point = ignore_point
    def _get_item_ignore_point( self, item_widget : PySide6.QtWidgets.QTreeWidgetItem ) -> _IgnorePoint:
        return item_widget.user_ignore_point
    
    def _append(self, ignore_point : _IgnorePoint ) -> None:
        strings = [
            self._configuration.get_cam_definition( ignore_point.cam_id ).label,
            self._configuration.get_interest( ignore_point.coco_class_id ).label,
            str( [ignore_point.at.x, ignore_point.at.y] )
        ]
        item = PySide6.QtWidgets.QTreeWidgetItem( None, strings )
        self._set_item_ignore_point( item, ignore_point )
        self._ignore_list_widget.addTopLevelItem(item)

        button = PySide6.QtWidgets.QPushButton(" ✖ ")      
        button.pressed.connect( lambda: self._remove_from_model(ignore_point) )
        self._ignore_list_widget.setItemWidget(item, 3, button)


    def _remove(self, removed_ignore_point : _IgnorePoint ):
        it = PySide6.QtWidgets.QTreeWidgetItemIterator(self._ignore_list_widget)
        while it.value():
            item = it.value()
            if self._get_item_ignore_point(item) is removed_ignore_point:
                self._ignore_list_widget.invisibleRootItem().removeChild( item )
                return
            it += 1
        raise ValueError("Ignore point not found.")

    @graceful_handler
    def _remove_from_model(self, ignore_point : _IgnorePoint ) -> None:
        self._ignore_list.remove( ignore_point )
    
    @graceful_handler
    def _on_current_item_change(self) -> None:
        if self._ignore_list_widget.currentIndex() is None:
            self._ignore_item_display.clear()
            return
        
        item = self._ignore_list_widget.currentItem()
        ignore_point = self._get_item_ignore_point(item)
        image = self._get_cam_image( ignore_point.cam_id )
        if image.isNull():
            image = PySide6.QtGui.QPixmap( self._PREVIEW_SIZE.x , self._PREVIEW_SIZE.y )
            image.fill( PySide6.QtGui.QColorConstants.Gray )
        with PySide6.QtGui.QPainter( image ) as painter:
            scale = min( image.size().width(), image.size().height() ) / min( self._PREVIEW_SIZE.x, self._PREVIEW_SIZE.y )
            scale = max( 1, scale )
            def set_color(color):
                painter.setPen( PySide6.QtGui.QPen( color, scale ) )    
            at_screen_point = PySide6.QtCore.QPoint( int(ignore_point.at.x * image.size().width()), int(ignore_point.at.y * image.size().height()) )
            set_color( PySide6.QtGui.QColorConstants.White )
            painter.drawEllipse( at_screen_point, 3*scale, 3*scale )
            set_color( PySide6.QtGui.QColorConstants.Red )
            painter.drawEllipse( at_screen_point, 4*scale, 4*scale )
            set_color( PySide6.QtGui.QColorConstants.White )
            painter.drawEllipse( at_screen_point, 5*scale, 5*scale )
        self._ignore_item_display.setPixmap( image )
    
    @graceful_handler
    def _adjust_list_view_width(self):
        for i in range( 0, self._ignore_list_widget.columnCount()):
            self._ignore_list_widget.resizeColumnToContents(i)
        
        total_width = sum([self._ignore_list_widget.sizeHintForColumn(i) for i in range(0, self._ignore_list_widget.columnCount())])
        self._ignore_list_widget.setMaximumWidth( total_width + 50)

