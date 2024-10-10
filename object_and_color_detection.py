import torch
import cv2
import numpy as np
from scipy.spatial import distance

# Load YOLOv5 model via PyTorch Hub
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # YOLOv5s is a smaller model

# Definiera en lista med vanliga färger (BGR-format) och deras namn
colors = {
    "red": [255, 0, 0],
    "green": [0, 255, 0],
    "blue": [0, 0, 255],
    "yellow": [255, 255, 0],
    "orange": [255, 165, 0],
    "purple": [128, 0, 128],
    "pink": [255, 65, 170],
    "light blue": [85, 165, 220],
    "light green": [115,160, 70]
}
def detect_objects(image_path):
    # Load image
    image = cv2.imread(image_path)
    
    # Run object identification with YOLO
    results = model(image)

    print("Printar ut resultatet från vad modellen hittar:", results)
    
    # Retrieve bounding boxes and classifications from YOLO results
    boxes = results.xyxy[0].numpy()  # Bounding boxes in xyxy-format
    objects_info = []

    if len(boxes) == 0:
        # Ingen objekt funnen, beräkna medelvärdet av färgerna för hela bilden
        avg_color = np.mean(image, axis=(0, 1))
        avg_color_rgb = avg_color[::-1]

        avg_color_name = closest_color(avg_color_rgb)

        # Lägg till information om "unknown object"
        objects_info.append({
            "class_id": "unknown object",
            "confidence": None,
            "color": avg_color_name,
            "rgb": avg_color_rgb,
            "shape": "unknown"
        })
    else:
        for box in boxes:
            x1, y1, x2, y2, confidence, class_id = box
            print("Cordinates", x1, x2, y1, y2)
            
            # Crop the object from the image
            object_image = image[int(y1):int(y2), int(x1):int(x2)]
            
            width = x2 - x1
            height = y2 - y1

            # Bestäm mitten av bounding box
            center_x = x1 + width / 2
            center_y = y1 + height / 2

            # Beräkna 20% kvadraten i mitten
            crop_size = 0.5 * min(width, height)
            crop_x1 = int(center_x - crop_size / 2)
            crop_y1 = int(center_y - crop_size / 2)
            crop_x2 = int(center_x + crop_size / 2)
            crop_y2 = int(center_y + crop_size / 2)

            # Beskär mittenområdet av bounding boxen
            cropped_img = image[crop_y1:crop_y2, crop_x1:crop_x2]
        
            # Beräkna medelvärdet av färgerna (RGB) 
            # Använd cropped_img (för mittersta 20%) eller object_image (för hela kvadraten)
            avg_color = np.mean(cropped_img, axis=(0, 1))
            avg_color_rgb = avg_color[::-1]

            avg_color_name = closest_color(avg_color_rgb)
            
            # Detect shape using contours
            shape = detect_shape(object_image)
            
            # Save information on object, its color, and shape
            objects_info.append({
                "class_id": class_id.item(),
                "confidence": confidence.item(),
                "color": avg_color_name,
                "rgb": avg_color_rgb,
                "shape": shape
            })
    
    return objects_info

def detect_shape(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply GaussianBlur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply thresholding to get a binary image (objects in white, background in black)
    _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)
    
    # Perform edge detection
    edges = cv2.Canny(thresh, 50, 150)

    # Find contours from edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return "unknown"  # No contours found
    
    # Loop over the contours and find the largest one
    for contour in contours:
        # Approximate the contour to a polygon
        approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
        
        # Classify shape based on the number of vertices in the polygon approximation
        if len(approx) == 3:
            return "triangle"
        elif len(approx) == 4:
            # Check if the shape is a square or a rectangle
            (x, y, w, h) = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            if 0.95 <= aspect_ratio <= 1.05:
                return "square"
            else:
                return "rectangle"
        elif len(approx) > 4:
            # Check if it's a circle (using a threshold for circularity)
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * (area / (perimeter ** 2))
            if circularity > 0.7:
                return "circle"
    
    return "unknown"


# Funktion för att hitta den närmaste färgen
def closest_color(avg_color):
    dark_threshold = 10  # Justera denna siffra för att definiera "supermörk"
    light_threshold = 250
    # Kontrollera om färgen är "supermörk"
    if np.mean(avg_color) < dark_threshold:
        return "black"
    
    if np.mean(avg_color) > light_threshold:
        return "white"
    
    min_distance = float("inf")
    closest_color_name = None
    
    for color_name, color_value in colors.items():
        # Beräkna Euklidiskt avstånd mellan färgen och de fördefinierade färgerna
        dist = distance.euclidean(avg_color, color_value)
        if dist < min_distance:
            min_distance = dist
            closest_color_name = color_name

    return closest_color_name

if __name__ == "__main__":
    # Path to image in images-folder
    image_path = "images/grönKloss.jpg"
    
    # Run object- and color detection
    detected_objects = detect_objects(image_path)

    # Print results
    for obj in detected_objects:
        print(f"Dominant Color: {obj['color']}")
        print(f"RGB Color: {obj['rgb']}")
        print(f"Shape: {obj['shape']}")
