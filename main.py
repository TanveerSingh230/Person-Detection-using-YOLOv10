import cv2
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from ultralytics import YOLO
import numpy as np

app = FastAPI()

# Load the pre-trained YOLOv8 model
model = YOLO('yolov10l.pt')  # Adjust the model path if necessary

def correct_orientation(image):
    h, w = image.shape[:2]

    # Perform object detection on the original and rotated images
    results_original = model(image, imgsz=1280)
    results_90_cw = model(cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE), imgsz=1280)
    results_90_ccw = model(cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE), imgsz=1280)

    # Function to count relevant objects in detection results
    def count_objects(results):
        count = 0
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = box.cls[0]
                # You can adjust the class IDs to focus on specific objects
                if class_id == 0:  # Assuming class_id 0 is for 'person'
                    count += 1
        return count

    # Count objects in each orientation
    counts = [
        count_objects(results_original),
        count_objects(results_90_cw),
        count_objects(results_90_ccw),
    ]

    # Find the orientation with the most detected objects
    max_count_idx = np.argmax(counts)

    if max_count_idx == 1:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif max_count_idx == 2:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        return image

def detect_objects_from_url(image_url):
    try:
        # Fetch the image from the URL
        response = requests.get(image_url)
        response.raise_for_status()  # Ensure we got a valid response
        nparr = np.frombuffer(response.content, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode image from URL")
        
        # Correct the orientation
        image_corrected = correct_orientation(image)
        
        # Perform object detection
        results = model(image_corrected, imgsz=1280)
        
        # Initialize a counter for the number of people
        person_count = 0
        
        # Iterate through the detections
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = box.cls[0]
                
                # Check if the detected object is a person (class id for 'person' is 0 in COCO dataset)
                if class_id == 0:
                    person_count += 1
        
        # Return the detection result
        return {"image_url": image_url, "count": person_count}
    
    except Exception as e:
        print(f"Error processing image from URL {image_url}: {e}")
        return {"image_url": image_url, "count": "Error"}

@app.get("/")
async def main():
    html_content = """
    <html>
    <head>
        <title>Person Detection</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
                padding: 20px;
                background-color: #f0f0f0;
            }
            .container {
                max-width: 600px;
                margin: 0 auto;
                background-color: #ffffff;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            }
            h1 {
                text-align: center;
                color: #333333;
            }
            form {
                text-align: center;
            }
            label {
                display: block;
                margin-bottom: 10px;
                font-weight: bold;
            }
            input[type="text"] {
                width: 100%;
                padding: 8px;
                margin-bottom: 10px;
                border: 1px solid #cccccc;
                border-radius: 4px;
                box-sizing: border-box;
            }
            button[type="submit"] {
                background-color: #4CAF50;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
            }
            button[type="submit"]:hover {
                background-color: #45a049;
            }
            #results {
                margin-top: 20px;
                padding: 10px;
                background-color: #f9f9f9;
                border: 1px solid #cccccc;
                border-radius: 4px;
                overflow-wrap: break-word;
            }
            .error {
                color: red;
            }
        </style>
        <script>
            async function getCount() {
                event.preventDefault();
                const form = document.getElementById('imageForm');
                const formData = new FormData(form);
                const imageUrl = formData.get('image_url');

                document.getElementById('results').innerHTML = 'Processing...';

                try {
                    const response = await fetch(`/detect/?image_url=${encodeURIComponent(imageUrl)}`);
                    if (!response.ok) {
                        throw new Error('Failed to fetch data');
                    }
                    const data = await response.json();
                    document.getElementById('results').innerHTML = `<strong>Image URL:</strong> ${data.image_url}<br><strong>Person Count:</strong> ${data.count}`;
                } catch (error) {
                    console.error('Error:', error);
                    document.getElementById('results').innerHTML = '<span class="error">Error occurred while processing image.</span>';
                }
            }
        </script>
    </head>
    <body>
        <div class="container">
            <h1>Person Detection</h1>
            <form id="imageForm" onsubmit="getCount()">
                <label for="image_url">Enter Image URL:</label>
                <input type="text" id="image_url" name="image_url" required>
                <button type="submit">Get Count</button>
            </form>
            <div id="results"></div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/detect/")
async def detect_from_url(image_url: str):
    # Perform object detection using the provided image URL
    detection_result = detect_objects_from_url(image_url)
    return JSONResponse(content=detection_result)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=80)
