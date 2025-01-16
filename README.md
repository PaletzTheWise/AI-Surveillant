This repository is a self-assembly kit for an application that can view RTSP streams (usually IP cameras) and detect objects via a 3rd-party YOLO (you only look once) detection model. There is no installer, you'll need a basic understanding of git, Python, and pip to take advantage of this.

# YOLO models #

YOLO models analyze only one frame at a time so they can't detect motion but they can still recognize an object such as a person, car, cat, etc.
This means the application does not have to worry about framerate and if the detection side lags behind it will just skip frames (and any detections in those frames).
Models that take previous frames into account, e.g. to detect motion, are a bit more finicky in that regard.

YOLO models seem to be the subject of intense research, so the idea is to make it easy to integrate with cutting-edge models.

The repository includes an example of integration with [YOLOv9](https://github.com/WongKinYiu/yolov9).

# Features #

 * RTSP stream live footage display
    * Sound is supported too but it is muted by default. Each stream has its own sound volume slider.
 * Detections
     * Filtering
        * Select which interests to enable/disable (e.g. Person, Car)
        * Adjustable sensitivity - AI models report confidence, low confidence is more likely to be a false positive.
        * For example Displays the last annotated detection for a stream - the detected objects are highlighted by a colored rectangle
     * Redetection cooldown logic prevents detection spam for the same object on the same stream. However, if the new detection has higher confidence (meaning it's probably a better image), it will replace the old detection.
     * Play sound files on detection identifying the detected object and stream. (Alert volume can be adjusted.)
     * History
        * Detections are saved as a jpg file with a user-hardcoded maximum count - the oldest detection will be deleted to honor the limit.
        * The history tab allows the user to view past detections.
     * Ignore list
        * Each previous detection in the history tab may be added to the ignore list so future detection of the same object on the same stream in the same position will be ignored.
          This is useful when the model consistently misidentifies an object or detects objects outside of the area of interest.
        * The ignore list tab allows the user to review and remove entries. Note that this displays the ignored location on an image from the _current_ live stream so the original detected object might not be visible anymore.
 * The overview tab displays all live footage from all streams and their respective last detections.
    * The layout can be automatic (good results if all streams have the same aspect ratio) or manually hard-coded.
 * Zoom controls on all displays - GUI buttons and/or mouse (hold ctrl and zoom with the wheel or left-click drag to pan)
 * Internationalization support
    * English and Czech localizations are available.

<img src="https://github.com/user-attachments/assets/19ddd019-bbb0-4a10-b059-7746d31b674a" width="400"/>
<img src="https://github.com/user-attachments/assets/011926ee-147d-4d48-a798-21081d564081" width="400"/>
<img src="https://github.com/user-attachments/assets/0a11ffce-051a-447e-851a-7c140621036e" width="400"/>
<img src="https://github.com/user-attachments/assets/73bab144-c26e-4983-a042-b503d819389a" width="400"/>

# Example Setup #

 1. Download pytorch from https://pytorch.org/. CUDA 11.8 and probably higher is supported but it requires a non-ancient Nvidia GPU. AI-Surveillant will fall back to CPU if CUDA is not available but it's very slow.
    (Note, pytorch is also listed in YOLOv9 requirements but pip -r might install it without CUDA support.)
 3. Clone AI-Surveillant repository.  
    git clone https://github.com/PaletzTheWise/AI-Surveillant.git
 4. Change directory to the cloned repository.
 5. Install AI-Surveillant prerequisites.  
    pip install -r requirements.txt
 6. Edit example.py. Change "rtsp:/camera.stream.url" to an actual RTSP stream URL.
 7. Clone YOLOv9  
    git clone https://github.com/WongKinYiu/yolov9.git
 8. Change directory to yolov9.
 9. Optionally, checkout to the latest version tested with AI Surveillant.  
    git checkout 5b1ea9a8b3f0ffe4fe0e203ec6232d788bb3fcff
 10. Install YOLOv9 prerequisites.
    pip install -r requirements.txt
 11. Download yolov9-m-converted.pt from https://github.com/WongKinYiu/yolov9/releases and put it into the weights/ folder under the yolov9 repo clone.
     (Other weights offer a different size/speed/accuracy balance, but the example, as provided, uses medium.)
 12. Change directory back to the cloned AI surveillant repository.
 13. Run the example:  
     python.exe example.py  
     If it fails, there should be a clue in the terminal.
 14. To run the example without console output, use  
     pythonw.exe example.py


