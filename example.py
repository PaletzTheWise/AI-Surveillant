import sys
import surveillance_ui
from PySide6.QtCore import (
    QRectF,
    QSizeF,
    QPointF,
)
from yolov9_detection_logic import (
    YoloV9Weights,
    YoloV9DetectionLogic
)

if __name__ == '__main__':
    configuration = surveillance_ui.Configuration(
        cam_definitions=[
            surveillance_ui.CamDefinition(
                url="rtsp://127.0.0.1:8554/", # URL to your camera stream.
                id=1,
                label="Cam 1",
                sound_alert_path=None, # Path to mp3 you want played for detections on the cam
            ),
            surveillance_ui.CamDefinition(
                url="", # URL to your camera.
                id=2,
                label="Cam 2",
                sound_alert_path=None, # Path to mp3 you want played for detections on the cam
            ),
            surveillance_ui.CamDefinition(
                url="", # URL to your camera.
                id=3,
                label="Cam 3",
                sound_alert_path=None, # Path to mp3 you want played for detections on the cam
            ),           
            # For testing purposes, you can use VLC media player to stream a video file:
            #  * Open VLC media player.
            #  * Go to Media -> Stream...
            #  * Add a video file to stream.
            #  * Click "Stream"
            #  * Click "Next"
            #  * Select the "RTSP" destination and click "Add"
            #  * Enter port 8554 and path "/"
            #  * Click "Next"
            #  * Unless you know the file is compatible as-is, activate transcoding and select a compatible format like "Video - H.264 + MP3 (MP4)".
            #  * Click "Next" and then click "Stream"
            # Such a stream is limited by the length of the video and VLC may have trouble streaming again on the same port. Killing all VLC processes should fix the issue.
            # Also, VLC does not support TCP/IP transport and may artifact a lot especially on HD resolution.
        ],
        interests=[
            surveillance_ui.Interest(
                coco_class_id=0, # id used by the model, usually a COCO class ID
                label="Person",
                enabled_by_default=True,
                sound_alert_path=None # Path to mp3 you want played for detections of the interest
            ),
        ],
        max_history_entries=100,
        detection_logic=YoloV9DetectionLogic(YoloV9Weights.Medium), # bigger model yields better results but requires more memory and CPU
        grid_widget_locs= [ # Manually arranged camera widgets for the overview tab. Everything gets scaled to the actual window width.
            QRectF( QPointF( 0, 0), QSizeF(16, 9) ), # Left cam   - 16:9 aspect ratio (assumes the camera streams in landscape mode)
            QRectF( QPointF(16, 0), QSizeF(16, 9) ), # Middle cam - 16:9 aspect ratio (assumes the camera streams in landscape mode)
            QRectF( QPointF(32, 0), QSizeF(9, 16)*(9/16) ), # Right cam  - 9:16 (but scaled so it matches the height of other cams) aspect ratio (assumes the camera streams in portrait mode)
        ],
        use_tcp_transport=False, # better for compatibility but more glitchy
    )

    sys.exit( surveillance_ui.run_surveillance_application( configuration ) )