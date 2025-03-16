import streamlit as st
import os
import sys
from utils.image_processing import process_image, analyze_skin, detect_problem_areas
from utils.ml_model import predict_skin_condition
from utils.ai_analysis import get_image_analysis
from utils.language import get_translation, init_language, change_language, LANGUAGES
from utils.location import get_user_location_from_ip, get_doctors_in_radius
from utils.chat import init_chat, start_chat_with_doctor, render_chat_interface
from data.products import get_product_recommendations
from data.doctors import get_nearby_doctors
from utils.product_recommendation import get_database_products
# from pages.product_recommendations import display_product
import cv2
import numpy as np
from utils.auth import register_user, login_user  # Import necessary functions

# Check if user is authenticated
if 'authenticated' not in st.session_state or not st.session_state.authenticated:
    st.warning("Please login or register to access the application")
    st.info("Redirecting to login page...")
    st.markdown("""
    <meta http-equiv="refresh" content="3;url=http://0.0.0.0:5000/login">
    """, unsafe_allow_html=True)
    st.stop()  # Stop execution of the rest of the app

# Initialize language settings
init_language()

# Initialize chat system
init_chat()

# Initialize session state variables if they don't exist
if 'user_location' not in st.session_state:
    st.session_state.user_location = get_user_location_from_ip()

if 'search_radius' not in st.session_state:
    st.session_state.search_radius = 50  # Default radius in kilometers


# Create a translation function with current language
def t(key):
    return get_translation(key)


# Page configuration
st.set_page_config(page_title="SkinHealth AI - Professional Skin Analysis",
                   page_icon="🏥",
                   layout="wide")

# Load custom CSS
with open('assets/style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


# If app is running, execute main app
def show_auth_forms():
    tab1, tab2 = st.tabs(["Вход", "Регистрация"])

    with tab1:
        st.header("Вход в систему")

        # Input fields for login
        login_email = st.text_input("Электронная почта", key="login_email")
        login_password = st.text_input("Пароль", type="password", key="login_password")
        if st.button("Войти"):
            try:
                from utils.database import User
                user = User.query.filter_by(email=login_email).first()

                if user and user.check_password(login_password):
                    st.session_state.authenticated = True
                    st.session_state.user_id = user.id
                    st.session_state.username = user.username
                    st.session_state.registered = True

                    st.success("Вход выполнен успешно")
                    st.rerun()
                else:
                    st.error("Неверный email или пароль")
            except Exception as e:
                st.error(f"Ошибка при входе: {str(e)}")

    with tab2:
        st.header("Регистрация")
        st.write("Пожалуйста, заполните форму ниже для регистрации.")

        # Input fields for registration
        email = st.text_input("Электронная почта", key="reg_email")
        username = st.text_input("Имя пользователя", key="reg_username")
        password = st.text_input("Пароль", type="password", key="reg_password")
        confirm_password = st.text_input("Подтвердите пароль", type="password", key="reg_confirm_password")
        user_type = st.selectbox("Тип пользователя", ["Пациент", "Врач", "Косметическая фирма"], key="reg_user_type")

        # Register button
        if st.button("Зарегистрироваться"):
            if password != confirm_password:
                st.error("Пароли не совпадают!")
            else:
                # Call the register_user function from utils.auth
                try:
                    from utils.auth import register_user
                    success, message = register_user(email, username, password, user_type)
                    print(f"streamlit: {success, message}")
                    if success:
                        st.success(message)
                        st.session_state.authenticated = True
                        st.session_state.registered = True
                        st.rerun()
                    else:
                        st.error(message)
                except Exception as e:
                    st.error(f"Ошибка при регистрации: {str(e)}")


if __name__ == "__main__":
    show_auth_forms()