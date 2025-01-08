import sys
import typing
import surveillance_ui
import os.path
import enum

if typing.TYPE_CHECKING:
    import numpy
    import supervision

# This code assumes YOLOv9 (https://github.com/WongKinYiu/yolov9) is cloned into the yolov9 subfolder.

class YoloV9Weights(enum.Enum):
    Tiny = 't'
    Small = 's'
    Medium = 'm'
    Compact = 'c'
    Extended = 'e'

# object ids in the default weights (extracted from the contents of results.names in detect()):
# {
#    0: 'person', 
#    1: 'bicycle',
#    2: 'car',
#    3: 'motorcycle',
#    4: 'airplane',
#    5: 'bus',
#    6: 'train',
#    7: 'truck',
#    8: 'boat',
#    9: 'traffic light',
#   10: 'fire hydrant',
#   11: 'stop sign',
#   12: 'parking meter',
#   13: 'bench',
#   14: 'bird',
#   15: 'cat',
#   16: 'dog',
#   17: 'horse',
#   18: 'sheep',
#   19: 'cow',
#   20: 'elephant',
#   21: 'bear',
#   22: 'zebra',
#   23: 'giraffe',
#   24: 'backpack',
#   25: 'umbrella',
#   26: 'handbag',
#   27: 'tie',
#   28: 'suitcase',
#   29: 'frisbee',
#   30: 'skis',
#   31: 'snowboard',
#   32: 'sports ball',
#   33: 'kite',
#   34: 'baseball bat',
#   35: 'baseball glove',
#   36: 'skateboard',
#   37: 'surfboard',
#   38: 'tennis racket',
#   39: 'bottle',
#   40: 'wine glass',
#   41: 'cup',
#   42: 'fork',
#   43: 'knife',
#   44: 'spoon',
#   45: 'bowl',
#   46: 'banana',
#   47: 'apple',
#   48: 'sandwich',
#   49: 'orange',
#   50: 'broccoli',
#   51: 'carrot',
#   52: 'hot dog',
#   53: 'pizza',
#   54: 'donut',
#   55: 'cake',
#   56: 'chair',
#   57: 'couch',
#   58: 'potted plant', 
#   59: 'bed',
#   60: 'dining table',
#   61: 'toilet',
#   62: 'tv',
#   63: 'laptop',
#   64: 'mouse',
#   65: 'remote',
#   66: 'keyboard',
#   67: 'cell phone',
#   68: 'microwave',
#   69: 'oven',
#   70: 'toaster',
#   71: 'sink',
#   72: 'refrigerator',
#   73: 'book',
#   74: 'clock',
#   75: 'vase',
#   76: 'scissors',
#   77: 'teddy bear',
#   78: 'hair drier',
#   79: 'toothbrush'
# }

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
        
    def configure( self, interest_ids : list[int], confidence : float ) -> None:
        self._ensure_model_initialized()
        self._ensure_yolov9_is_on_path()
        from yolov9.models.common import AutoShape
        model = typing.cast( AutoShape, self._model)
        model.classes = interest_ids
        model.conf = confidence

    def detect( self, image : 'numpy.ndarray' ) -> 'supervision.Detections':
        self._ensure_model_initialized()
        results = self._model(image, (image.shape[1], image.shape[0]), augment=False)
        return self._yolov9_detections_to_sv(results)
    
    def _yolov9_detections_to_sv(self, yolov9_results) -> 'supervision.Detections':
        import torch
        import numpy
        import supervision
        import supervision.config

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