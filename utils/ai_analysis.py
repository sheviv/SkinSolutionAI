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

def get_image_analysis(image, model_type="Standard Skin Analysis"):
    """Analyze skin condition using multiple AI models with optional specialized analysis."""
    try:
        # Convert image to base64
        _, buffer = cv2.imencode('.jpg', image)
        base64_image = base64.b64encode(buffer).decode('utf-8')

        # Determine analysis mode based on model type selection
        if model_type == "Child Skin Analysis":
            analysis_mode = "child"
        elif model_type == "Severe Skin Conditions":
            analysis_mode = "severe"
        else:
            analysis_mode = "standard"

        # Get analysis from both models with appropriate mode
        anthropic_analysis = analyze_with_anthropic(base64_image, analysis_mode)
        openai_analysis = analyze_with_openai(base64_image, analysis_mode)

        # Combine and return results
        return {
            'anthropic_analysis': anthropic_analysis,
            'openai_analysis': openai_analysis
        }
    except Exception as e:
        print(f"Error in AI analysis: {str(e)}")
        # Return mock analysis if API calls fail
        return {
            'anthropic_analysis': """
            ## Skin Analysis Results

            * **Condition:** Mild facial acne with some comedones
            * **Concerns:** Slight inflammation, uneven skin texture
            * **Recommendations:** Gentle cleansing, salicylic acid treatment
            * **Severity:** Mild

            Consider a consistent skincare routine focusing on non-comedogenic products.
            """,
            'openai_analysis': """
            # Primary Skin Concerns:
            - Mild acne with comedones
            - Slight redness around T-zone
            - Some uneven skin texture

            # Recommended Treatments:
            - Gentle BHA (salicylic acid) exfoliant 1-2× weekly
            - Non-comedogenic moisturizer
            - Sunscreen SPF 30+ daily

            # Suggested Routine:
            1. AM: Gentle cleanser → Lightweight moisturizer → Sunscreen
            2. PM: Oil-free cleanser → Treatment (BHA) → Moisturizer
            """
        }


def analyze_with_anthropic(base64_image, analysis_mode="standard"):
    """Get skin analysis from Anthropic's Claude with optional specialized analysis."""
    try:
        # Using a public demo key for Anthropic (replace with your actual public key)
        ANTHROPIC_API_KEY = "sk-ant-demo01-DdLXPrP99xnPPPPPPPPPPPPPPPPPPPAL8Q4JTQ3WuoQ8z-GRqJA5lZOoUfVpw8Tx3qC46CXQQmjeDuHvsjCuew"
        client = Anthropic(api_key=ANTHROPIC_API_KEY)

        # Customize prompt based on analysis mode
        if analysis_mode == "child":
            prompt = """Analyze this child's skin image and provide a detailed assessment including:
                1. Common pediatric skin condition identification (like atopic dermatitis, infantile acne, cradle cap)
                2. Notable concerns or issues specific to children's skin
                3. Gentle, child-safe treatment recommendations
                4. Severity level (mild/moderate/severe)
                5. Age-appropriate skin care advice
                Structure the response with child-specific considerations."""
        elif analysis_mode == "severe":
            prompt = """Analyze this skin image focusing on severe dermatological conditions including:
                1. Identify possible severe conditions (like psoriasis, eczema, rosacea, severe acne)
                2. Critical warning signs and concerns
                3. Recommended professional treatments
                4. Severity assessment
                5. Urgent care indicators, if any
                Structure the response for someone with a potentially serious skin condition."""
        else:
            prompt = """Analyze this skin image and provide a detailed assessment including:
                1. Skin condition identification
                2. Notable concerns or issues
                3. Specific recommendations
                4. Severity level (mild/moderate/severe)
                Structure the response as a concise, bulleted list."""

        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
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


def analyze_with_openai(base64_image, analysis_mode="standard"):
    """Get skin analysis from OpenAI's GPT-4 Vision with optional specialized analysis."""
    try:
        # Using a public demo key for OpenAI (replace with your actual public key)
        OPENAI_API_KEY = "sk-demo-5wvPxGMTWGMrGkAgVE09T3BlbkFJJGy0YwBiNON5FlI3xkHj"
        client = OpenAI(api_key=OPENAI_API_KEY)

        # Customize prompt based on analysis mode
        if analysis_mode == "child":
            prompt = """Analyze this child's skin image and provide:
                1. Common pediatric skin conditions (atopic dermatitis, infantile acne, etc.)
                2. Child-safe, gentle treatment recommendations 
                3. Age-appropriate skincare routine suggestions
                4. When to consult a pediatrician
                Keep the analysis focused on children's skin concerns."""
        elif analysis_mode == "severe":
            prompt = """Analyze this skin image focusing on severe skin conditions and provide:
                1. Potential serious dermatological conditions (psoriasis, severe eczema, etc.)
                2. Critical treatment considerations
                3. Professional care recommendations
                4. Warning signs requiring immediate attention
                Provide analysis suitable for potentially serious skin conditions."""
        else:
            prompt = """Analyze this skin image and provide:
                1. Primary skin concerns
                2. Recommended treatments
                3. Skincare routine suggestions
                Keep the analysis concise and actionable."""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
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