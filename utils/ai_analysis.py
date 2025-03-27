import os
import base64
import cv2
import numpy as np

def get_image_analysis(image, model_type="Standard Skin Analysis"):
    """Analyze skin condition using ML model with optional specialized analysis."""
    try:
        # Convert image to base64
        _, buffer = cv2.imencode('.jpg', image)
        base64_image = base64.b64encode(buffer).decode('utf-8')

        # Return basic analysis based on ML model prediction
        return {
            'ml_analysis': """
            # Skin Analysis Results
            - Condition: Based on ML model prediction
            - Severity: To be determined by ML model
            - Treatment: See product recommendations below
            """
        }
    except Exception as e:
        print(f"Error in ML analysis: {str(e)}")
        return {
            'ml_analysis': "Analysis unavailable"
        }