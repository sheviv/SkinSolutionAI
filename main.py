import streamlit as st
from utils.image_processing import process_image, analyze_skin
from utils.ml_model import predict_skin_condition
from utils.ai_analysis import get_image_analysis
from data.products import get_product_recommendations
from data.doctors import get_nearby_doctors
import cv2
import numpy as np
import os

# Page configuration
st.set_page_config(
    page_title="SkinHealth AI - Professional Skin Analysis",
    page_icon="🏥",
    layout="wide"
)

# Load custom CSS
with open('assets/style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Medical Disclaimer
def show_disclaimer():
    with st.sidebar:
        st.warning("""
        ⚕️ **Medical Disclaimer**

        This application is for informational purposes only and is not a substitute 
        for professional medical advice, diagnosis, or treatment. Always seek the 
        advice of your physician or other qualified health provider with any 
        questions you may have regarding a medical condition.
        """)

def main():
    st.title("SkinHealth AI")
    st.subheader("Professional Skin Analysis & Recommendations")

    # Display clinic images in columns
    col1, col2 = st.columns(2)
    with col1:
        st.image("https://images.unsplash.com/photo-1598300188904-6287d52746ad",
                caption="Professional Dermatology Care", use_container_width=True)
    with col2:
        st.image("https://images.unsplash.com/photo-1690306815613-f839b74af330",
                caption="Expert Consultation", use_container_width=True)

    # Check for API keys
    if not st.session_state.get('api_keys_checked'):
        st.session_state.api_keys_checked = True
        with st.spinner("Checking API configuration..."):
            anthropic_key = os.environ.get('ANTHROPIC_API_KEY')
            openai_key = os.environ.get('OPENAI_API_KEY')
            if not anthropic_key or not openai_key:
                st.warning("⚠️ AI-powered detailed analysis requires API keys. Some features may be limited.")

    # Image Upload Section
    st.header("Upload Your Skin Photo")
    st.info("For best results, please ensure good lighting and a clear view of the skin area.")
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        # Convert uploaded file to image array
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        with st.spinner("Processing image and analyzing skin condition..."):
            # Process and analyze image
            processed_image = process_image(image)
            skin_features = analyze_skin(processed_image)

            # Get ML model prediction and AI analysis
            ml_prediction = predict_skin_condition(skin_features)
            ai_analysis = get_image_analysis(processed_image)

        # Display Results
        st.header("Analysis Results")

        # Display the processed image
        st.image(processed_image, caption="Analyzed Image", use_container_width=True)

        # Create three columns for different analyses
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("ML Model Analysis")
            st.write(f"**Condition:** {ml_prediction['condition']}")
            st.write(f"**Confidence:** {ml_prediction['confidence']}")

            st.write("**Key Factors:**")
            for factor, importance in ml_prediction['key_factors']:
                st.write(f"- {factor}: {importance:.2%}")

            with st.expander("View Detailed Probabilities"):
                for condition, prob in ml_prediction['probabilities'].items():
                    st.write(f"- {condition}: {prob}")

        with col2:
            st.subheader("Anthropic Analysis")
            if ai_analysis and ai_analysis['anthropic_analysis']:
                st.markdown(ai_analysis['anthropic_analysis'])
            else:
                st.info("Anthropic analysis unavailable")

        with col3:
            st.subheader("OpenAI Analysis")
            if ai_analysis and ai_analysis['openai_analysis']:
                st.markdown(ai_analysis['openai_analysis'])
            else:
                st.info("OpenAI analysis unavailable")

        # Display detected characteristics
        st.subheader("Detailed Skin Metrics:")
        metrics_cols = st.columns(3)

        for i, (feature, value) in enumerate(skin_features.items()):
            with metrics_cols[i % 3]:
                st.metric(
                    label=feature,
                    value=f"{float(value):.1f}" if isinstance(value, (float, int)) else value
                )

        # Product Recommendations section
        st.header("Recommended Products")
        products = get_product_recommendations(ml_prediction['condition'])

        for product in products:
            with st.expander(f"🏥 {product['name']} - ${product['price']}", expanded=False):
                # Basic Information
                st.write(f"**Description:** {product['description']}")

                # Ingredients Section
                st.subheader("🧪 Ingredients")
                st.write(", ".join(product['ingredients']))

                # Key Benefits
                st.subheader("✨ Key Benefits")
                for benefit in product['key_benefits']:
                    st.write(f"- {benefit}")

                # Usage Instructions
                st.subheader("📝 How to Use")
                st.write(f"**Frequency:** {product['usage_instructions']['frequency']}")
                st.write("**Steps:**")
                for step in product['usage_instructions']['steps']:
                    st.write(f"- {step}")
                if product['usage_instructions']['warnings']:
                    st.warning(f"⚠️ **Warning:** {product['usage_instructions']['warnings']}")

                # Skin Compatibility
                st.subheader("👥 Suitable For")
                st.write(", ".join(product['skin_compatibility']))

                # Ingredient Analysis
                st.subheader("🔬 Ingredient Analysis")

                # Active Ingredients
                st.write("**Active Ingredients:**")
                for ingredient, description in product['ingredient_analysis']['active_ingredients'].items():
                    st.markdown(f"**💊 {ingredient}**")
                    st.write(description)
                    st.divider()

                # Potential Irritants
                if product['ingredient_analysis']['potential_irritants']:
                    st.write("**⚠️ Potential Irritants:**")
                    for ingredient, warning in product['ingredient_analysis']['potential_irritants'].items():
                        st.markdown(f"**⚠️ {ingredient}**")
                        st.warning(warning)

                # Additional Information
                st.write("**Additional Information:**")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**pH Level:** {product['ingredient_analysis'].get('ph_level', 'Not specified')}")
                    st.write(f"**Fragrance Free:** {product['ingredient_analysis'].get('fragrance_free', 'Not specified')}")
                with col2:
                    st.write(f"**Comedogenic Rating:** {product['ingredient_analysis']['comedogenic_rating']}/5")
                    st.caption("(0 = non-comedogenic, 5 = highly comedogenic)")

                # Certifications
                if product['ingredient_analysis'].get('certifications'):
                    st.write("**🏆 Certifications:**")
                    st.write(", ".join(product['ingredient_analysis']['certifications']))


        # Doctor Directory section
        st.header("Consult a Professional")
        doctors = get_nearby_doctors()
        for doctor in doctors:
            with st.expander(f"👨‍⚕️ Dr. {doctor['name']} - {doctor['speciality']}", expanded=False):
                st.write(f"**Experience:** {doctor['experience']} years")
                st.write(f"**Location:** {doctor['location']}")
                st.write(f"**Contact:** {doctor['contact']}")
    show_disclaimer()

if __name__ == "__main__":
    main()