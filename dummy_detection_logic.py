import surveillance_ui
import supervision
import numpy

class DummyDetectionLogic(surveillance_ui.DetectionLogic):
    def detect( self, _ : numpy.ndarray ) -> supervision.Detections:
        return supervision.Detections.empty()