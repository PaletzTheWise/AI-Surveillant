import sys
import surveillance_ui
from yolov9_detection_logic import (
    YoloV9Weights,
    YoloV9DetectionLogic
)

if __name__ == '__main__':
    configuration = surveillance_ui.Configuration(
        cam_definitions=[
            surveillance_ui.CamDefinition( 
                url="rtsp:/camera.stream.url",
                id=1,
                label="Cam 1",
                sound_alert_path=None # Path to mp3 you want played for detections on the cam
            )
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
        grid_column_count=2
    )

    sys.exit( surveillance_ui.run_surveillance_application( configuration ) )