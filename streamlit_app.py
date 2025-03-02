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
import cv2
import numpy as np

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
from main import main
main()