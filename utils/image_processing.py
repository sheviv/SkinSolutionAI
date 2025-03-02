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
    _, thresh = cv2.threshold(masked_l, 127, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

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


def detect_problem_areas(image, features):
    """Detect and mark problem areas in the skin using precise segmentation."""
    # Create a copy of the image for marking
    marked_image = image.copy()

    # Initialize a transparent overlay for segmentations
    overlay = np.zeros_like(image, dtype=np.uint8)

    # Convert to different color spaces for analysis
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    # Create skin mask
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Apply morphological operations to improve skin mask
    kernel = np.ones((5, 5), np.uint8)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)

    problem_areas = []

    # Detect dark spots using precise segmentation
    if features["Spots Detected"] > 2:
        l, _, _ = cv2.split(lab)
        masked_l = cv2.bitwise_and(l, l, mask=skin_mask)

        # Apply adaptive thresholding for better spot detection
        thresh = cv2.adaptiveThreshold(masked_l, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 21, 5)

        # Clean up with morphological operations
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours by size to avoid noise
        min_spot_size = 5
        max_spot_size = 500
        spots = [
            cnt for cnt in contours
            if min_spot_size < cv2.contourArea(cnt) < max_spot_size
        ]

        # Create spot masks on the overlay
        for i, cnt in enumerate(spots[:8]):  # Increased to 8 spots
            spot_mask = np.zeros_like(skin_mask)
            cv2.drawContours(spot_mask, [cnt], 0, 255, -1)

            # Get bounding box for ROI extraction
            x, y, w, h = cv2.boundingRect(cnt)

            # Apply blue color to spot areas in overlay with transparency
            blue_tint = np.zeros_like(image)
            blue_tint[spot_mask > 0] = [255, 0, 0]  # Blue color

            # Apply the tint to the overlay with alpha blending
            overlay = cv2.addWeighted(overlay, 1, blue_tint, 0.5, 0)

            spot_size = cv2.contourArea(cnt)
            severity = "High" if spot_size > 100 else "Moderate" if spot_size > 50 else "Mild"

            area = {
                "id":
                f"spot_{i + 1}",
                "type":
                "Dark Spot",
                "bbox": (x, y, w, h),
                "size":
                int(spot_size),
                "severity":
                severity,
                "description":
                f"Dark spot indicating possible hyperpigmentation or sun damage. The affected area is approximately {int(spot_size)} pixels."
            }
            problem_areas.append(area)

    # Detect redness using precise segmentation
    if features["Redness"] > 35:
        r, g, b = cv2.split(image)

        # Create a more sophisticated redness mask
        redness_mask = np.zeros_like(r)
        redness_mask[((r > g * 1.1) & (r > b * 1.1) & (skin_mask > 0))] = 255

        # Clean up with morphological operations
        kernel = np.ones((5, 5), np.uint8)
        redness_mask = cv2.morphologyEx(redness_mask, cv2.MORPH_OPEN, kernel)
        redness_mask = cv2.morphologyEx(redness_mask, cv2.MORPH_CLOSE, kernel)

        # Find contours of red areas
        contours, _ = cv2.findContours(redness_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        # Filter by size
        min_area_size = 30
        max_area_size = 2000
        redness_areas = [
            cnt for cnt in contours
            if min_area_size < cv2.contourArea(cnt) < max_area_size
        ]

        # Create redness masks on the overlay
        for i, cnt in enumerate(redness_areas[:5]):  # Increased to 5 areas
            red_mask = np.zeros_like(skin_mask)
            cv2.drawContours(red_mask, [cnt], 0, 255, -1)

            # Get bounding box for ROI extraction
            x, y, w, h = cv2.boundingRect(cnt)

            # Apply red color to redness areas in overlay with transparency
            red_tint = np.zeros_like(image)
            red_tint[red_mask > 0] = [0, 0, 255]  # Red color

            # Apply the tint to the overlay with alpha blending
            overlay = cv2.addWeighted(overlay, 1, red_tint, 0.4, 0)

            area_size = cv2.contourArea(cnt)
            severity = "High" if area_size > 300 else "Moderate" if area_size > 150 else "Mild"

            area = {
                "id":
                f"redness_{i + 1}",
                "type":
                "Redness",
                "bbox": (x, y, w, h),
                "size":
                int(area_size),
                "severity":
                severity,
                "description":
                f"Area of increased redness indicating possible inflammation or irritation. The affected area is approximately {int(area_size)} pixels."
            }
            problem_areas.append(area)

    # Detect texture issues with precise segmentation
    if features["Texture"] > 45:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        masked_gray = cv2.bitwise_and(gray, gray, mask=skin_mask)

        # Use a combination of Gabor filters to detect texture patterns
        ksize = 9
        sigma = 3
        theta = 0
        lambd = 10.0
        gamma = 0.5

        # Apply Gabor filter at different orientations
        texture_mask = np.zeros_like(gray)

        for angle in [0, 45, 90, 135]:
            theta = np.deg2rad(angle)
            gabor_kernel = cv2.getGaborKernel((ksize, ksize),
                                              sigma,
                                              theta,
                                              lambd,
                                              gamma,
                                              0,
                                              ktype=cv2.CV_32F)
            filtered = cv2.filter2D(masked_gray, cv2.CV_8UC3, gabor_kernel)

            # Threshold the filtered image
            _, thresh = cv2.threshold(filtered, 50, 255, cv2.THRESH_BINARY)
            texture_mask = cv2.bitwise_or(texture_mask, thresh)

        # Clean up with morphological operations
        kernel = np.ones((5, 5), np.uint8)
        texture_mask = cv2.morphologyEx(texture_mask, cv2.MORPH_OPEN, kernel)
        texture_mask = cv2.morphologyEx(texture_mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(texture_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        # Filter by size
        min_texture_size = 50
        texture_areas = [
            cnt for cnt in contours if cv2.contourArea(cnt) > min_texture_size
        ]

        if texture_areas:
            # Create separate texture regions
            for i, cnt in enumerate(
                    texture_areas[:3]):  # Top 3 texture regions
                texture_region_mask = np.zeros_like(gray)
                cv2.drawContours(texture_region_mask, [cnt], 0, 255, -1)

                # Get bounding box
                x, y, w, h = cv2.boundingRect(cnt)

                # Apply green color to texture areas with transparency
                green_tint = np.zeros_like(image)
                green_tint[texture_region_mask > 0] = [0, 255,
                                                       0]  # Green color

                # Apply the tint to the overlay with alpha blending
                overlay = cv2.addWeighted(overlay, 1, green_tint, 0.35, 0)

                area_size = cv2.contourArea(cnt)
                severity = "High" if area_size > 500 else "Moderate" if area_size > 200 else "Mild"

                area = {
                    "id":
                    f"texture_{i + 1}",
                    "type":
                    "Texture Irregularity",
                    "bbox": (x, y, w, h),
                    "size":
                    int(area_size),
                    "severity":
                    severity,
                    "description":
                    f"Area with uneven texture indicating possible roughness, fine lines, or enlarged pores. The affected area is approximately {int(area_size)} pixels."
                }
                problem_areas.append(area)

    # Combine overlay with original image
    marked_image = cv2.addWeighted(marked_image, 1.0, overlay, 0.6, 0)

    return marked_image, problem_areas
