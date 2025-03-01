import cv2
import numpy as np

def process_image(image):
    """Process the uploaded image for skin analysis."""
    # Convert to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize image maintaining aspect ratio
    height = 300
    aspect_ratio = image_rgb.shape[1] / image_rgb.shape[0]
    width = int(height * aspect_ratio)
    image_resized = cv2.resize(image_rgb, (width, height))

    # Apply color correction
    lab = cv2.cvtColor(image_resized, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    image_corrected = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # Denoise image
    image_denoised = cv2.fastNlMeansDenoisingColored(image_corrected)

    return image_denoised

def analyze_skin(image):
    """Extract skin features from the processed image."""
    # Convert to different color spaces for analysis
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)

    # Create skin mask
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Apply mask to get only skin regions
    skin_region = cv2.bitwise_and(image, image, mask=skin_mask)

    # Calculate features
    features = {
        "Tone Uniformity": calculate_uniformity(lab, skin_mask),
        "Brightness": calculate_brightness(ycrcb, skin_mask),
        "Texture": calculate_texture(image, skin_mask),
        "Spots Detected": detect_spots(lab, skin_mask),
        "Redness": calculate_redness(image, skin_mask),
        "Pigmentation": calculate_pigmentation(lab, skin_mask)
    }

    return features

def calculate_uniformity(lab_image, mask):
    """Calculate color uniformity using LAB color space."""
    l, a, b = cv2.split(lab_image)
    masked_l = cv2.bitwise_and(l, l, mask=mask)
    std_dev = np.std(masked_l[mask > 0])
    # Normalize to 0-100 scale, lower std_dev means more uniform
    return max(0, min(100, 100 - (std_dev * 2)))

def calculate_brightness(ycrcb_image, mask):
    """Calculate overall brightness using YCrCb color space."""
    y, _, _ = cv2.split(ycrcb_image)
    masked_y = cv2.bitwise_and(y, y, mask=mask)
    return np.mean(masked_y[mask > 0])

def calculate_texture(image, mask):
    """Calculate skin texture score using gradient analysis."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    masked_gray = cv2.bitwise_and(gray, gray, mask=mask)

    # Calculate gradients
    sobelx = cv2.Sobel(masked_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(masked_gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)

    # Normalize to 0-100 scale
    return min(100, np.mean(gradient_magnitude[mask > 0]))

def detect_spots(lab_image, mask):
    """Detect number of spots/blemishes using LAB color space."""
    l, _, _ = cv2.split(lab_image)
    masked_l = cv2.bitwise_and(l, l, mask=mask)

    # Threshold to detect darker spots
    _, thresh = cv2.threshold(masked_l, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours by size to avoid noise
    min_spot_size = 5
    spots = [cnt for cnt in contours if cv2.contourArea(cnt) > min_spot_size]

    return len(spots)

def calculate_redness(image, mask):
    """Calculate skin redness level."""
    r, g, b = cv2.split(image)
    masked_r = cv2.bitwise_and(r, r, mask=mask)
    masked_g = cv2.bitwise_and(g, g, mask=mask)

    # Calculate relative redness
    redness = np.mean(masked_r[mask > 0]) - np.mean(masked_g[mask > 0])
    return max(0, min(100, redness))

def calculate_pigmentation(lab_image, mask):
    """Calculate pigmentation variation using a channel from LAB color space."""
    _, a, _ = cv2.split(lab_image)
    masked_a = cv2.bitwise_and(a, a, mask=mask)

    # Calculate pigmentation score based on variation in a channel
    pigmentation = np.std(masked_a[mask > 0])
    return max(0, min(100, pigmentation))