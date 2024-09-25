import torch
import cv2

# Load YOLOv5-modellen via PyTorch Hub
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # YOLOv5s is a smaller model

def detect_objects(image_path):
    # Load image
    image = cv2.imread(image_path)
    
    # Run object identification with YOLO
    results = model(image)
    
    # Retrieve bounding boxes and classifications from YOLO-results
    boxes = results.xyxy[0]  # Bounding boxes in xyxy-format
    objects_info = []
    
    for box in boxes:
        x1, y1, x2, y2, confidence, class_id = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Crop the object from the image
        object_image = image[y1:y2, x1:x2]
        
        # Call function to get dominant color
        dominant_color = get_dominant_color(object_image)
        
        # Save information on object and its color
        objects_info.append({
            "class_id": class_id.item(),
            "confidence": confidence.item(),
            "dominant_color": dominant_color
        })
    
    return objects_info

def get_dominant_color(image):
    # Omvandla bilden till HSV färgrymd
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Calculate mean of color code
    avg_color_per_row = cv2.mean(hsv_image)
    
    # Return dominant color in HSV
    return avg_color_per_row

if __name__ == "__main__":
    # Path to image in images-folder
    image_path = "test_image.jpg"
    
    # Run object- and color detection
    detected_objects = detect_objects(image_path)

    # Print results
    for obj in detected_objects:
        print(f"Objektklass: {obj['class_id']}, Dominant färg (HSV): {obj['dominant_color']}")
