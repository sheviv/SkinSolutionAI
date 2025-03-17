from flask import redirect, url_for, flash, session
from flask_login import LoginManager, login_user, logout_user, current_user
from utils.database import User, db
import streamlit as st

login_manager = LoginManager()


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


def init_auth(app):
    """Initialize authentication for the app"""
    login_manager.init_app(app)
    login_manager.login_view = 'login'


def register_user(email, username, password, user_type="Пациент", doctor_data=None):
    """Register a new user"""
    # Check if user already exists
    existing_email = User.query.filter_by(email=email).first()
    if existing_email:
        return False, "Email already registered"

    existing_username = User.query.filter_by(username=username).first()
    if existing_username:
        return False, "Username already taken"

    try:
        if user_type in ["Врач", "Doctor"]:
            if not doctor_data:
                return False, "Doctor registration requires additional information"

            if not doctor_data.get('speciality'):
                return False, "Speciality field is required"
            if not doctor_data.get('experience'):
                return False, "Experience field is required"
            if not doctor_data.get('license_number'):
                return False, "License number is required"

            # Create user first
            new_user = User(email=email, username=username, user_type=user_type)
            new_user.set_password(password)
            db.session.add(new_user)
            db.session.flush()  # Get the new user's ID

            # Then create doctor with user reference
            from utils.database import Doctor
            new_doctor = Doctor(
                user_id=new_user.id,
                email=email,
                username=username,
                speciality=doctor_data.get('speciality', ''),
                experience=int(doctor_data.get('experience', 0)) if doctor_data.get('experience') else 0,
                license_number=doctor_data.get('license_number', ''),
                address=doctor_data.get('address', ''),
                phone=doctor_data.get('phone', ''),
                affordable_hours=doctor_data.get('affordable_hours', '')
            )
            db.session.add(new_doctor)
        else:
            # Create regular user
            new_user = User(email=email, username=username, user_type=user_type)
            new_user.set_password(password)
            db.session.add(new_user)

        db.session.commit()
        return True, "Registration successful"
    except Exception as e:
        db.session.rollback()
        return False, f"Registration failed: {str(e)}"


def login_user_with_credentials(email, password):
    """Login a user with email and password"""
    # Try to find the user by email
    user = User.query.filter_by(email=email).first()

    if not user or not user.check_password(password):
        return False, "Invalid email or password"

    # Check if we're in a Flask request context
    from flask import has_request_context

    # Login the user with Flask-Login only if we're in a request context
    if has_request_context():
        login_user(user, remember=True)

    # Set Streamlit session state
    try:
        st.session_state.authenticated = True
        st.session_state.user_id = user.id
        st.session_state.username = user.username
        st.session_state.registered = True
    except Exception as e:
        print(f"Error setting Streamlit session: {e}")
        # Ignore if Streamlit context is not available
        pass

    return True, "Login successful"
