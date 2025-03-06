from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_login import LoginManager, login_required, login_user, logout_user, current_user
from utils.database import User, db, init_db
from utils.auth import init_auth, register_user, login_user_with_credentials

# Initialize Flask app
app = Flask(__name__)

# Initialize database
init_db(app)

# Initialize authentication
init_auth(app)


# User authentication routes
@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        email = request.form.get('email')
        username = request.form.get('username')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        role = request.form.get('role', 'Patient')

        if password != confirm_password:
            flash('Passwords do not match!', 'error')
            return render_template('register.html')

        success, message = register_user(email, username, password, role)
        if success:
            flash(message, 'success')
            return redirect(url_for('login'))
        else:
            flash(message, 'error')

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        try:
            # Find the user first
            user = User.query.filter_by(email=email).first()

            if user and user.check_password(password):
                # Manually log in the user
                login_user(user, remember=True)

                # Update Streamlit session if needed
                try:
                    import streamlit as st
                    st.session_state.authenticated = True
                    st.session_state.user_id = user.id
                    st.session_state.username = user.username
                    st.session_state.registered = True
                except Exception as e:
                    print(f"Streamlit session error: {e}")
                    # Ignore if Streamlit context is not available
                    pass

                flash("Login successful", 'success')
                return redirect(url_for('dashboard'))
            else:
                flash("Invalid email or password", 'error')
        except Exception as e:
            flash(f"Login error: {str(e)}", 'error')

    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


@app.route('/dashboard')
@login_required
def dashboard():
    # Redirect to Streamlit app
    return redirect("http://0.0.0.0:8501")


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
