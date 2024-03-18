# Driver Drowsiness Detection

Driver Drowsiness Detection is a project aimed at preventing accidents caused by drowsy driving. It utilizes computer vision techniques to monitor the driver's face and detect signs of drowsiness, such as eye closure or yawning. When drowsiness is detected, the system alerts the driver to take a break or pull over safely.

## Features

- Real-time face detection and tracking using Haar cascade classifier.
- Eye aspect ratio (EAR) calculation to detect eye closure.
- Lip distance calculation to detect yawning.
- Auditory and visual alerts when drowsiness is detected.
- Adjustable sensitivity settings for customizing detection thresholds.

## Requirements

- Python 3.x
- OpenCV
- dlib
- imutils
- playsound
