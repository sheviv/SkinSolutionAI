from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import LoginManager, login_required, login_user, logout_user, current_user
from utils.database import User, db, init_db, Doctor
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

        if role in ["Doctor", "Врач"]:
            try:
                # Get doctor-specific fields
                speciality = request.form.get('speciality', '')
                experience = request.form.get('experience', '')
                license_number = request.form.get('license_number', '')
                address = request.form.get('address', '')
                phone = request.form.get('phone', '')
                affordable_hours = request.form.get('affordable_hours', '')

                # Validate required fields
                if not speciality:
                    flash('Speciality field is required', 'error')
                    return render_template('register.html')
                if not experience:
                    flash('Experience field is required', 'error')
                    return render_template('register.html')
                if not license_number:
                    flash('License number is required', 'error')
                    return render_template('register.html')

                # Create user with doctor data
                doctor_data = {
                    'speciality': speciality.strip(),
                    'experience': int(experience) if experience.isdigit() else 0,
                    'license_number': license_number.strip(),
                    'address': address.strip() if address else '',
                    'phone': phone.strip() if phone else '',
                    'affordable_hours': affordable_hours.strip() if phone else ''
                }
                success, message = register_user(email, username, password, "Врач", doctor_data)
                if success:
                    flash(message, 'success')
                    return redirect(url_for('login'))
                else:
                    flash(message, 'error')
                    return render_template('register.html')
            except Exception as e:
                db.session.rollback()
                flash(f"Error creating doctor profile: {str(e)}", 'error')
                return render_template('register.html')
        else:
            success, message = register_user(email, username, password, role)
            if success:
                flash(message, 'success')
                return redirect(url_for('login'))
            else:
                flash(message, 'error')
                return render_template('register.html')

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        try:
            success, message = login_user_with_credentials(email, password)
            if success:
                flash("Login successful", 'success')
                return redirect(url_for('dashboard'))
            else:
                flash(message, 'error')
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
