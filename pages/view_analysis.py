import streamlit as st
import cv2
import numpy as np
import os
from utils.published_analysis import PublishedAnalysis
from utils.language import get_translation, init_language
from utils.database import User, db
from flask import Flask
from utils.database import init_db

# Create a Flask app for database context
app = Flask(__name__)
init_db(app)

# Initialize language
init_language()


def t(key):
    return get_translation(key)


st.title("View Published Skin Analysis")


# Add login form at the top of the page
def show_login_form():
    with st.form("login_form"):
        st.subheader("Doctor Login")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")

        if submit:
            # Verify user credentials within app context
            with app.app_context():
                user = User.query.filter_by(email=email).first()

                if user and user.check_password(password):
                    # Check if user is a doctor
                    if user.user_type == "Врач":
                        st.session_state.doctor_authenticated = True
                        st.session_state.doctor_id = user.id
                        st.session_state.doctor_name = user.username
                        st.rerun()
                    else:
                        st.error("Only doctors can access this page.")
                else:
                    st.error("Invalid email or password")


# Check if doctor is already authenticated
if not st.session_state.get('doctor_authenticated', False):
    show_login_form()
    st.info("Please login as a doctor to view published analyses.")
    st.stop()  # Stop execution if not authenticated

# The rest of the page only shows if authenticated as a doctor
st.header(f"Welcome, Dr. {st.session_state.get('doctor_name', '')}")

# Add logout button
if st.button("Logout"):
    st.session_state.doctor_authenticated = False
    st.session_state.doctor_id = None
    st.session_state.doctor_name = None
    st.rerun()

# Look up Analysis by ID section at the top
st.header("Look up Analysis by ID")
with st.form("analysis_lookup_form"):
    analysis_id = st.text_input("Enter Analysis ID", placeholder="e.g. ab12cd34")
    submitted = st.form_submit_button("View Analysis")

    if submitted and analysis_id:
        # Look up the analysis
        analysis = PublishedAnalysis.get_analysis(analysis_id)

        if analysis:
            st.success(f"Analysis found!")
            st.session_state.selected_analysis_id = analysis_id
            st.rerun()
        else:
            st.error("Analysis not found. Please check the ID and try again.")

# Display all public analyses by default
st.header("All Published Analyses")
with app.app_context():
    all_public_analyses = PublishedAnalysis.get_all_public_analyses()

if all_public_analyses:
    # Create a grid layout
    col1, col2, col3 = st.columns(3)
    cols = [col1, col2, col3]

    # Distribute analyses across columns
    for i, (analysis_id, analysis) in enumerate(all_public_analyses.items()):
        col = cols[i % len(cols)]

        with col:
            with st.container():
                st.subheader(f"Analysis #{analysis_id[:4]}")

                # Display condition
                st.info(f"{analysis.get('condition', 'Unknown')}")

                # Show image if available
                image_path = analysis.get('image_path')
                if image_path:
                    # Handle both relative and absolute paths
                    if not os.path.isabs(image_path):
                        # Make path absolute relative to project root
                        abs_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), image_path)
                    else:
                        abs_path = image_path

                    if os.path.exists(abs_path):
                        image = cv2.imread(abs_path)
                        if image is not None:
                            # Convert from BGR to RGB
                            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            st.image(image_rgb, use_container_width=True)
                    else:
                        st.warning(f"Image file not found: {image_path}")

                # Show date in a smaller font
                st.caption(f"Published: {analysis.get('timestamp', 'Unknown')[:10]}")

                # Create a unique key for storing detail view state
                detail_view_key = f"detail_view_{analysis_id}"

                # Initialize the state if it doesn't exist
                if detail_view_key not in st.session_state:
                    st.session_state[detail_view_key] = False

                # Add a button to toggle details view
                button_label = "Hide Details" if st.session_state[detail_view_key] else "View Details"
                if st.button(button_label, key=f"view_{analysis_id}"):
                    # Toggle the detail view state
                    st.session_state[detail_view_key] = not st.session_state[detail_view_key]
                    st.rerun()

                # Show details if the state is True
                if st.session_state[detail_view_key]:
                    analysis_detail = PublishedAnalysis.get_analysis(analysis_id)
                    if analysis_detail:
                        st.subheader("Detail View")

                        # Create columns for image and details
                        detail_col1, detail_col2 = st.columns([1, 1])

                        with detail_col1:
                            # Display the image if available
                            img_path = analysis_detail.get('image_path')
                            if img_path:
                                # Handle both relative and absolute paths
                                if not os.path.isabs(img_path):
                                    # Make path absolute relative to project root
                                    abs_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                            img_path)
                                else:
                                    abs_path = img_path

                                if os.path.exists(abs_path):
                                    img = cv2.imread(abs_path)
                                    if img is not None:
                                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                        st.image(img_rgb, caption="Analyzed Skin Image", use_container_width=True)
                                else:
                                    st.warning(f"Image file not found: {img_path}")

                        with detail_col2:
                            # Display analysis information
                            st.markdown("#### Skin Condition")
                            st.info(f"{analysis_detail.get('condition', 'Unknown')}")
                            st.text(f"Published on: {analysis_detail.get('timestamp', 'Unknown date')}")

                            # Display skin features/metrics if available
                            features = analysis_detail.get('features', {})
                            if features:
                                st.markdown("#### Skin Metrics")
                                for feature, value in features.items():
                                    if isinstance(value, (int, float)):
                                        st.metric(label=feature, value=f"{value:.1f}")

                            # Try to get username of the publisher
                            try:
                                with app.app_context():
                                    user = User.query.filter_by(id=analysis_detail.get('user_id')).first()
                                    if user:
                                        st.text(f"Published by: {user.username}")
                            except Exception as e:
                                # If database query fails, just skip username display
                                pass
else:
    st.info("No published analyses available yet.")

# If an analysis is selected for detailed view from ID lookup
if 'selected_analysis_id' in st.session_state:
    analysis_id = st.session_state.selected_analysis_id
    analysis = PublishedAnalysis.get_analysis(analysis_id)

    if analysis:
        st.header(f"Detailed Analysis #{analysis_id}")

        # Create two columns for layout
        col1, col2 = st.columns([1, 1])

        with col1:
            # Display the image if available
            image_path = analysis.get('image_path')
            if image_path:
                # Handle both relative and absolute paths
                if not os.path.isabs(image_path):
                    # Make path absolute relative to project root
                    abs_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), image_path)
                else:
                    abs_path = image_path

                if os.path.exists(abs_path):
                    image = cv2.imread(abs_path)
                    if image is not None:
                        # Convert from BGR to RGB
                        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        st.image(image_rgb, caption="Analyzed Skin Image", use_container_width=True)
                else:
                    st.warning(f"Image file not found: {image_path}")

        with col2:
            # Display analysis information
            st.subheader("Skin Condition")
            st.info(f"{analysis.get('condition', 'Unknown')}")
            st.text(f"Published on: {analysis.get('timestamp', 'Unknown date')}")

            # Display skin features/metrics if available
            features = analysis.get('features', {})
            if features:
                st.subheader("Skin Metrics")
                for feature, value in features.items():
                    if isinstance(value, (int, float)):
                        st.metric(label=feature, value=f"{value:.1f}")

            # Try to get username of the publisher
            try:
                with app.app_context():
                    user = User.query.filter_by(id=analysis.get('user_id')).first()
                    if user:
                        st.text(f"Published by: {user.username}")
            except Exception as e:
                # If database query fails, just skip username display
                pass

        # Button to close detailed view
        if st.button("Close Details"):
            del st.session_state.selected_analysis_id
            st.rerun()
