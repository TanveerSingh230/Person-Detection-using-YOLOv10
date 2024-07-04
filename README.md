# Person Detection using YOLOv10

This project is a web application that detects and counts the number of people in an image using the YOLOv10 object detection model. Users can input an image URL through an HTML interface, and the application processes the image to count the number of people present. The application is built using FastAPI and is deployed on a web server.

## Features

- **Person Detection**: Utilizes YOLOv10 to detect and count the number of people in an image.
- **Web Interface**: An HTML interface allows users to enter an image URL and get the count of people in the image.
- **FastAPI Backend**: A FastAPI backend processes the image and performs the detection.
- **Automatic Orientation Correction**: The application corrects the orientation of the image to maximize the detection accuracy.

## Usage

1. Open the web interface in your browser.
2. Enter the URL of the image you want to analyze.
3. Click the "Get Count" button.
4. The application will process the image and display the number of people detected in the image.

## Code Explanation

### `main.py`

This is the main FastAPI application file.

- **Dependencies**:
  - `cv2`: OpenCV library for image processing.
  - `requests`: To fetch images from the provided URL.
  - `fastapi`: The FastAPI framework.
  - `ultralytics`: YOLOv10 model for object detection.
  - `numpy`: For numerical operations.

- **Load the YOLOv10 model**:
  ```python
  !git clone https://github.com/THU-MIG/yolov10.git
  %cd yolov10
  pip install .

  # Download the weights
  import os
  import urllib.request
  #Create a directory for the weights in the current working directory
  weights_dir = os.path.join(os.getcwd(), "weights")
  os.makedirs(weights_dir, exist_ok=True)

  # URLs of the weight files
  urls = [
    "https://github.com/jameslahm/yolov10/releases/download/v1.0/yolov10n.pt",
    "https://github.com/jameslahm/yolov10/releases/download/v1.0/yolov10s.pt",
    "https://github.com/jameslahm/yolov10/releases/download/v1.0/yolov10m.pt",
    "https://github.com/jameslahm/yolov10/releases/download/v1.0/yolov10b.pt",
    "https://github.com/jameslahm/yolov10/releases/download/v1.0/yolov10x.pt",
    "https://github.com/jameslahm/yolov10/releases/download/v1.0/yolov10l.pt"
  ]

  # Download each file
  for url in urls:
      file_name = os.path.join(weights_dir, os.path.basename(url))
      urllib.request.urlretrieve(url, file_name)
      print(f"Downloaded {file_name}")
