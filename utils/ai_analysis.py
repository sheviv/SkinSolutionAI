import os
import base64
from io import BytesIO
import anthropic
from anthropic import Anthropic
from openai import OpenAI
import cv2
import numpy as np

# the newest Anthropic model is "claude-3-5-sonnet-20241022" which was released October 22, 2024
# the newest OpenAI model is "gpt-4o" which was released May 13, 2024

def get_image_analysis(image):
    """Analyze skin condition using multiple AI models."""
    try:
        # Convert image to base64
        _, buffer = cv2.imencode('.jpg', image)
        base64_image = base64.b64encode(buffer).decode('utf-8')

        # Get analysis from both models
        anthropic_analysis = analyze_with_anthropic(base64_image)
        openai_analysis = analyze_with_openai(base64_image)

        # Combine and return results
        return {
            'anthropic_analysis': anthropic_analysis,
            'openai_analysis': openai_analysis
        }
    except Exception as e:
        print(f"Error in AI analysis: {str(e)}")
        return None

def analyze_with_anthropic(base64_image):
    """Get skin analysis from Anthropic's Claude."""
    try:
        client = Anthropic(api_key=os.environ.get('ANTHROPIC_API_KEY'))

        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """Analyze this skin image and provide a detailed assessment including:
                        1. Skin condition identification
                        2. Notable concerns or issues
                        3. Specific recommendations
                        4. Severity level (mild/moderate/severe)
                        Structure the response as a concise, bulleted list."""
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": base64_image
                        }
                    }
                ]
            }]
        )
        return response.content

    except Exception as e:
        print(f"Anthropic analysis error: {str(e)}")
        return "Analysis unavailable"

def analyze_with_openai(base64_image):
    """Get skin analysis from OpenAI's GPT-4 Vision."""
    try:
        client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """Analyze this skin image and provide:
                            1. Primary skin concerns
                            2. Recommended treatments
                            3. Skincare routine suggestions
                            Keep the analysis concise and actionable."""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=400
        )
        return response.choices[0].message.content

    except Exception as e:
        print(f"OpenAI analysis error: {str(e)}")
        return "Analysis unavailable"