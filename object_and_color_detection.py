import torch
import cv2
import numpy as np

# Load YOLOv5 model via PyTorch Hub
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # YOLOv5s is a smaller model

def detect_objects(image_path):
    # Load image
    image = cv2.imread(image_path)
    
    # Run object identification with YOLO
    results = model(image)

    print("Printar ut resultatet från vad modellen hittar:", results)
    
    # Retrieve bounding boxes and classifications from YOLO results
    boxes = results.xyxy[0]  # Bounding boxes in xyxy-format
    objects_info = []
    
    for box in boxes:
        x1, y1, x2, y2, confidence, class_id = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Crop the object from the image
        object_image = image[y1:y2, x1:x2]
        
        # Call function to get dominant color
        dominant_color = get_dominant_color(object_image)
        
        # Call function to get shape
        shape = get_shape(object_image)
        
        # Save information on object, its color, and shape
        objects_info.append({
            "class_id": class_id.item(),
            "confidence": confidence.item(),
            "dominant_color": dominant_color,
            "shape": shape
        })
    
    return objects_info

def hsv_to_color_name(h, s, v):
    # Ensure hue is between 0 and 360
    h = h % 360
    s = s / 100
    v = v / 100

    # Check for grayscale colors
    if v <= 0.2:
        return "Black"
    elif v >= 0.9 and s <= 0.2:
        return "White"
    elif s <= 0.2:
        return "Gray"

    # Define color ranges based on hue
    if (h >= 0 and h <= 15) or (h >= 345 and h <= 360):
        color = "Red"
    elif h > 15 and h <= 45:
        color = "Orange"
    elif h > 45 and h <= 75:
        color = "Yellow"
    elif h > 75 and h <= 90:
        color = "Lime"
    elif h > 90 and h <= 150:
        color = "Green"
    elif h > 150 and h <= 180:
        color = "Cyan"
    elif h > 180 and h <= 210:
        color = "Sky Blue"
    elif h > 210 and h <= 270:
        color = "Blue"
    elif h > 270 and h <= 300:
        color = "Purple"
    elif h > 300 and h < 345:
        color = "Magenta/Pink"
    else:
        color = "Unknown"

    # Further adjust based on saturation and value for brightness and darkness
    if v >= 0.7 and s <= 0.5:
        color = "Light " + color
    elif v <= 0.5:
        color = "Dark " + color

    return color

def get_dominant_color(image):
    # Convert image to HSV color wheel
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Calculate mean of color code
    avg_color_per_row = cv2.mean(hsv_image)
    
    # Return dominant color in HSV
    return avg_color_per_row

def get_shape(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply a threshold to get a binary image
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Assuming there is at least one contour found
    if contours:
        # Get the largest contour
        contour = max(contours, key=cv2.contourArea)

        # Approximate the shape of the contour
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        # Determine shape based on the number of vertices
        num_vertices = len(approx)

        if num_vertices == 3:
            return "Triangle"
        elif num_vertices == 4:
            aspect_ratio = cv2.contourArea(contour) / (cv2.boundingRect(contour)[2] * cv2.boundingRect(contour)[3])
            return "Square" if 0.95 <= aspect_ratio <= 1.05 else "Rectangle"
        elif num_vertices == 5:
            return "Pentagon"
        elif num_vertices > 5:
            return "Circle"
    return "Unknown"

if __name__ == "__main__":
    # Path to image in images-folder
    image_path = "images/muggar.jpg"
    
    # Run object- and color detection
    detected_objects = detect_objects(image_path)

    # Print results
    for obj in detected_objects:
        print(f"Object Class ID: {obj['class_id']}, Confidence: {obj['confidence']}")
        print(f"Dominant Color: {obj['dominant_color']}")
        
        h = obj['dominant_color'][0]
        s = obj['dominant_color'][1]
        v = obj['dominant_color'][2]

        name = hsv_to_color_name(h, s, v)
        print(f"Color Name: {name}")
        print(f"Shape: {obj['shape']}")