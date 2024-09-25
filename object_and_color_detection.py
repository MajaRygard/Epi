import torch
import cv2

# Ladda YOLOv5-modellen via PyTorch Hub
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # YOLOv5s är en mindre modell

def detect_objects(image_path):
    # Läs in bilden
    image = cv2.imread(image_path)
    
    # Kör objektigenkänning med YOLO
    results = model(image)
    
    # Hämta bounding boxes och klassificeringar från YOLO-resultaten
    boxes = results.xyxy[0]  # Bounding boxes i xyxy-format
    objects_info = []
    
    for box in boxes:
        x1, y1, x2, y2, confidence, class_id = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Beskär ut objektet från bilden
        object_image = image[y1:y2, x1:x2]
        
        # Kalla på funktionen för att få den dominanta färgen i objektet
        dominant_color = get_dominant_color(object_image)
        
        # Spara information om objektet och dess färg
        objects_info.append({
            "class_id": class_id.item(),
            "confidence": confidence.item(),
            "dominant_color": dominant_color
        })
    
    return objects_info

def get_dominant_color(image):
    # Omvandla bilden till HSV färgrymd
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Beräkna medelvärdet av färgerna (dominant färg)
    avg_color_per_row = cv2.mean(hsv_image)
    
    # Returnera dominant färg som HSV-värden
    return avg_color_per_row

if __name__ == "__main__":
    # Sökväg till bilden i images-mappen
    image_path = "test_image.jpg"
    
    # Kör objekt- och färgdetektering
    detected_objects = detect_objects(image_path)

    # Skriv ut resultaten
    for obj in detected_objects:
        print(f"Objektklass: {obj['class_id']}, Dominant färg (HSV): {obj['dominant_color']}")
