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


def register_user(email, username, password, user_type="Пациент"):
    """Register a new user"""
    # Check if user already exists
    existing_email = User.query.filter_by(email=email).first()
    if existing_email:
        return False, "Email already registered"

    existing_username = User.query.filter_by(username=username).first()
    if existing_username:
        return False, "Username already taken"

    # Create new user
    new_user = User(email=email, username=username)
    new_user.set_password(password)
    new_user.user_type = user_type  # Установка типа пользователя после создания объекта

    try:
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
