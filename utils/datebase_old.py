import os
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

# Initialize SQLAlchemy
db = SQLAlchemy()


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    # user_type = db.Column(db.String(20), nullable=False, default="Пациент")  # Тип пользователя
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    # The following attributes are required by Flask-Login
    @property
    def is_authenticated(self):
        return True

    def get_id(self):
        return str(self.id)

    def is_anonymous(self):
        return False

    def __repr__(self):
        return f'<User {self.username}>'


def init_db(app):
    """Initialize the database with the Flask app"""
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///skinhealth.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'default-secret-key')
    app.app_context().push()
    db.init_app(app)

    # Create tables if they don't exist
    with app.app_context():
        db.create_all()

        # Ensure is_active column exists by executing raw SQL
        # This is a safer approach than dropping and recreating tables
        try:
            import sqlite3
            conn = sqlite3.connect('instance/skinhealth.db')
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(user)")
            columns = [column[1] for column in cursor.fetchall()]
            if 'is_active' not in columns:
                cursor.execute("ALTER TABLE user ADD COLUMN is_active BOOLEAN DEFAULT 1")
                conn.commit()
            conn.close()
        except Exception as e:
            print(f"Migration error: {e}")
