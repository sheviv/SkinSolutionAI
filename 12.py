import streamlit as st
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from flask_login import login_required, current_user, logout_user
import cv2
import numpy as np
import os
from utils.image_processing import process_image, analyze_skin, detect_problem_areas
from utils.ml_model import predict_skin_condition
from utils.ai_analysis import get_image_analysis
from utils.language import get_translation, init_language, change_language, LANGUAGES
from utils.location import get_user_location_from_ip, get_doctors_in_radius
from utils.chat import init_chat, start_chat_with_doctor, render_chat_interface
from utils.database import init_db, User, db
from utils.auth import init_auth, register_user, login_user_with_credentials
from data.products import get_product_recommendations
from data.doctors import get_nearby_doctors

# Initialize Flask app
app = Flask(__name__)

# Initialize database
init_db(app)

# Initialize authentication
init_auth(app)

# Initialize language settings
init_language()

# Initialize chat system
init_chat()

# Create a translation function with current language
def t(key):
    return get_translation(key)

# Streamlit registration form
def show_registration_form():
    st.title("Регистрация")
    st.write("Пожалуйста, заполните форму ниже для регистрации.")

    # Input fields for registration
    email = st.text_input("Электронная почта")
    username = st.text_input("Имя пользователя")
    password = st.text_input("Пароль", type="password")
    confirm_password = st.text_input("Подтвердите пароль", type="password")

    # Register button
    if st.button("Зарегистрироваться"):
        if password != confirm_password:
            st.error("Пароли не совпадают!")
        else:
            # Call the register_user function from utils.auth
            try:
                success, message = register_user(email, username, password)
                if success:
                    st.success(message)
                    st.session_state.registered = True  # Mark user as registered
                    st.rerun()  # Перезагрузить страницу для отображения основной программы
                else:
                    st.error(message)
            except Exception as e:
                st.error(f"Ошибка при регистрации: {str(e)}")

# Page configuration
st.set_page_config(page_title="SkinHealth AI - Professional Skin Analysis",
                   page_icon="🏥",
                   layout="wide")

# Load custom CSS
with open('assets/style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def main():
    # Initialize session state variables if they don't exist
    if 'user_location' not in st.session_state:
        st.session_state.user_location = get_user_location_from_ip()

    if 'search_radius' not in st.session_state:
        st.session_state.search_radius = 50  # Default radius in kilometers

    # Check if chat is active
    if st.session_state.get('chat_active', False):
        # Render chat interface
        render_chat_interface()

        # Provide a way to return to the main app
        if st.button("Вернуться к анализу кожи"):
            st.session_state.chat_active = False
            st.rerun()
        return  # Exit the function early to show only chat interface

    # Show registration form if user is not registered
    if not st.session_state.get('registered', False):
        show_registration_form()
        return  # Exit the function early to show only registration form

    # Add a settings expander in the sidebar
    with st.sidebar:
        with st.expander(t("settings")):
            # Language settings
            st.write(f"**{t('language')}**")
            language_options = list(LANGUAGES.keys())
            current_lang_name = [name for name, code in LANGUAGES.items()
                                 if code == st.session_state.language][0]
            selected_language = st.selectbox(
                t("select_language"),
                options=language_options,
                index=language_options.index(current_lang_name)
            )

            # Change language if user selects a different one
            selected_lang_code = LANGUAGES[selected_language]
            if selected_lang_code != st.session_state.language:
                change_language(selected_lang_code)

            # Location settings
            st.write(f"**{t('location_settings')}**")

            # Display current location
            user_location = st.session_state.user_location
            st.write(f"{t('your_location')}: {user_location.get('city', '')}, {user_location.get('state', '')}")

            # Option to enter location manually (simplified for demo)
            if st.button(t("detect_location")):
                st.session_state.user_location = get_user_location_from_ip()
                st.rerun()

            # Search radius setting
            st.write(f"**{t('search_radius')}**")
            radius_options = range(0, 100 + 1)
            selected_radius = st.select_slider(
                "",
                options=radius_options,
                value=st.session_state.search_radius
            )

            # Update search radius if changed
            if selected_radius != st.session_state.search_radius:
                st.session_state.search_radius = selected_radius

    st.title(t("app_title"))
    st.subheader(t("app_subtitle"))

    # Display clinic images in columns
    col1, col2 = st.columns(2)
    with col1:
        st.image(
            "https://images.unsplash.com/photo-1598300188904-6287d52746ad",
            caption=t("clinic_image_1"),
            use_container_width=True)
    with col2:
        st.image(
            "https://images.unsplash.com/photo-1690306815613-f839b74af330",
            caption=t("clinic_image_2"),
            use_container_width=True)

    # Set API keys checked to true so we don't need to check again
    if not st.session_state.get('api_keys_checked'):
        st.session_state.api_keys_checked = True

    # Image Upload Section
    st.header(t("upload_header"))
    st.info(t("upload_info"))
    uploaded_file = st.file_uploader(t("upload_button"),
                                     type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        # Convert uploaded file to image array
        file_bytes = np.asarray(bytearray(uploaded_file.read()),
                                dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        with st.spinner("Processing image and analyzing skin condition..."):
            # Process and analyze image
            processed_image = process_image(image)
            skin_features = analyze_skin(processed_image)

            # Detect problem areas in the image
            marked_image, problem_areas = detect_problem_areas(
                processed_image, skin_features)

            # Get ML model predictions and AI analysis
            from utils.ml_model import predict_skin_condition
            from utils.ml_ensemble import predict_with_ensemble

            # Get predictions from all models
            ensemble_predictions = predict_with_ensemble(skin_features)

            # Store all models' predictions for display
            st.session_state.ensemble_predictions = ensemble_predictions

            # Use the most reliable model's prediction for recommendations
            most_reliable_model = ensemble_predictions.get("most_reliable_model", "Random Forest")
            ml_prediction = predict_skin_condition(skin_features) if most_reliable_model == "Random Forest" else \
                ensemble_predictions[most_reliable_model]

            # Get AI analysis without trying external API calls
            ai_analysis = get_image_analysis(processed_image)

            # Store problem areas in session state for display
            st.session_state.problem_areas = problem_areas
            st.session_state.marked_image = marked_image

        # Display Results
        st.header(t("analysis_results"))
        # Display the original and marked images side by side
        col_img1, col_img2 = st.columns(2)

if __name__ == "__main__":
    main()