# Epi
Interaktion 1 Projekt, Epi White

# Objekt- och färgdetektering med YOLOv5

Det här projektet identifierar objekt och deras dominanta färger i en bild med hjälp av YOLOv5-modellen och OpenCV för färgigenkänning.

## Installation

1. Klona detta repository:

    ```bash
    git clone https://github.com/your-username/my-object-detection-project.git
    cd my-object-detection-project
    ```

2. Installera beroenden:

    ```bash
    pip install -r requirements.txt
    ```

3. Placera din bild i mappen `images/` och döp den till `test_image.jpg` (eller uppdatera sökvägen i koden).

4. Kör skriptet:

    ```bash
    python object_and_color_detection.py
    ```

## Output

Koden kommer att skriva ut:
- Objektklass (med ett numeriskt klass-ID från YOLO)
- Dominant färg (i HSV-format) för varje identifierat objekt.
