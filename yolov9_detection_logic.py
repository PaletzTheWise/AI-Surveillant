import sys
import numpy
import typing
import surveillance_ui
import os.path
import enum
import supervision
import supervision.config

# This code assumes YOLOv9 (https://github.com/WongKinYiu/yolov9) is cloned into the yolov9 subfolder.

class YoloV9Weights(enum.Enum):
    Tiny = 't'
    Small = 's'
    Medium = 'm'
    Compact = 'c'
    Extended = 'e'

class YoloV9DetectionLogic(surveillance_ui.DetectionLogic):
    _model : object
    _weight : YoloV9Weights

    def __init__( self, weight : YoloV9Weights ):
        self._model = None
        self._weight = weight

    def _ensure_model_initialized(self) -> None:
        import torch
        self._ensure_yolov9_is_on_path()
        from yolov9.models.common import DetectMultiBackend, AutoShape

        if self._model is not None:
            return

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self._model = AutoShape( DetectMultiBackend( weights=f"yolov9/weights/yolov9-{self._weight.value}-converted.pt", device=device, data='data/coco.yaml', fuse=True) )
        self._model.iou = 0.6 # intersection over union (when to merge overlapping detections into one)
        self._model.agnostic = False
        self._model.max_det = 1000
        
    def configure( self, coco_class_ids : list[int], confidence : float ) -> None:
        self._ensure_model_initialized()
        self._ensure_yolov9_is_on_path()
        from yolov9.models.common import AutoShape
        model = typing.cast( AutoShape, self._model)
        model.classes = coco_class_ids
        model.conf = confidence

    def detect( self, image : numpy.ndarray ) -> surveillance_ui.SupervisionDetections:
        self._ensure_model_initialized()
        results = self._model(image, (image.shape[1], image.shape[0]), augment=False)
        return self._yolov9_detections_to_sv(results)
    
    def _yolov9_detections_to_sv(self, yolov9_results) -> surveillance_ui.SupervisionDetections:
        import torch

        xyxy, confidences, class_ids = [], [], []

        for det in yolov9_results.pred:
            for *xyxy_coords, conf, cls_id in reversed(det):
                xyxy.append(torch.stack(xyxy_coords).cpu().numpy())
                confidences.append(float(conf))
                class_ids.append(int(cls_id))

        class_names = numpy.array([yolov9_results.names[i] for i in class_ids])
        
        if not xyxy:
            return supervision.Detections.empty()  
        
        return supervision.Detections(
            xyxy=numpy.vstack(xyxy),
            confidence=numpy.array(confidences),
            class_id=numpy.array(class_ids),
            data={supervision.config.CLASS_NAME_DATA_FIELD: class_names},
        )
    
    def _ensure_yolov9_is_on_path(self):
        yolov9_path = os.path.join( os.path.dirname( os.path.realpath(__file__) ), "yolov9" )
        if yolov9_path not in sys.path:
            sys.path.append(yolov9_path)