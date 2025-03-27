import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import torch

# костыль
torch.classes.__path__ = []

# Initialize YOLO model
yolo_model = YOLO('yolov8x.pt')

# Define body parts and facial features classes
BODY_PARTS = {
    0: 'person',
    1: 'face',
    15: 'person',
    16: 'eye',
    17: 'nose',
    27: 'ear',
    28: 'hair',
    29: 'eyebrow',
    30: 'mouth',
    31: 'neck'
}


def preprocess_image(image):
    """Preprocess image for detection"""
    if isinstance(image, np.ndarray):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
    return image


def calculate_skin_percentage(image):
    """Calculate the percentage of skin-colored pixels in the image"""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Define skin color range in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
    skin_pixels = np.sum(skin_mask > 0)
    total_pixels = skin_mask.size

    return (skin_pixels / total_pixels) * 100


def check_facial_features(image, results):
    """Check for facial features in YOLO results"""
    facial_features = {16, 17, 27, 28, 29, 30}  # eye, nose, ear, hair, eyebrow, mouth
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if cls in facial_features and conf > 0.35:
                return True
    return False


def check_skin_tone(image):
    """Check if image contains significant skin-colored areas"""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Multiple skin tone ranges for different ethnicities
    skin_ranges = [
        ((0, 20, 70), (20, 255, 255)),  # Light to medium skin
        ((0, 10, 60), (25, 255, 255)),  # Dark skin
        ((0, 15, 50), (30, 255, 255))  # Very dark skin
    ]

    total_skin = np.zeros(image.shape[:2], dtype=np.uint8)
    for lower, upper in skin_ranges:
        skin_mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        total_skin = cv2.bitwise_or(total_skin, skin_mask)

    skin_percentage = (np.sum(total_skin > 0) / total_skin.size) * 100
    return skin_percentage > 25


def is_body_part(image):
    """Enhanced check if the image contains a body part using multiple methods"""
    try:
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            np_image = np.array(image)
        else:
            np_image = image.copy()

        # Convert to RGB
        rgb_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)

        # Process with YOLO
        results = yolo_model(rgb_image, conf=0.5)  # Increased confidence threshold

        skin_related_boxes = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                # Consider both person and facial features
                if cls in BODY_PARTS and conf > 0.4:  # Lowered threshold for facial features
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    crop = rgb_image[y1:y2, x1:x2]
                    if crop.size > 0:
                        skin_percentage = calculate_skin_percentage(crop)
                        if skin_percentage > 30:  # At least 30% should be skin-colored
                            # Calculate if the box is a reasonable size
                            box_area = (x2 - x1) * (y2 - y1)
                            image_area = rgb_image.shape[0] * rgb_image.shape[1]
                            area_ratio = box_area / image_area

                            # Box should not be too small or too large
                            if 0.05 < area_ratio < 0.95:
                                skin_related_boxes.append({
                                    'conf': conf,
                                    'skin_percentage': skin_percentage,
                                    'area_ratio': area_ratio
                                })

        # Multiple verification methods
        verifications = []

        # Method 1: Check general body part detection
        if skin_related_boxes:
            best_detection = max(skin_related_boxes,
                                 key=lambda x: x['conf'] * (x['skin_percentage'] / 100))
            if (best_detection['conf'] > 0.4 and
                    best_detection['skin_percentage'] > 20 and
                    0.01 < best_detection['area_ratio'] < 0.95):
                verifications.append(True)

        # Method 2: Check for facial features
        if check_facial_features(rgb_image, results):
            verifications.append(True)

        # Method 3: Check skin tone presence
        if check_skin_tone(rgb_image):
            verifications.append(True)

        # Return True if any verification method confirms it's a body part
        return any(verifications)

    except Exception as e:
        print(f"Error in body part detection: {str(e)}")
        return False  # Return False on errors to prevent false positives
