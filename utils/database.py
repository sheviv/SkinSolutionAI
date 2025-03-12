import os
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import json

# Initialize SQLAlchemy
db = SQLAlchemy()


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    user_type = db.Column(db.String(20), nullable=False,
                          default="Пациент")  # Тип пользователя
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)

    # Relationship with published analyses
    analyses = db.relationship('PublishedAnalysis', backref='user', lazy=True, cascade="all, delete-orphan")

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class PublishedAnalysis(db.Model):
    id = db.Column(db.String(8), primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    condition = db.Column(db.String(100), nullable=False, default="Unknown")
    features = db.Column(db.Text, nullable=True)  # Stored as JSON
    image_path = db.Column(db.String(200), nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    is_public = db.Column(db.Boolean, default=True)

    def get_features_dict(self):
        """Convert the JSON string of features back to dictionary"""
        if self.features:
            return json.loads(self.features)
        return {}

    # Связи с профессиональными таблицами
    # doctor = db.relationship('Doctor',
    #                          backref='user',
    #                          uselist=False,
    #                          cascade="all, delete-orphan")
    # cosmetic_firm = db.relationship('CosmeticFirm',
    #                                 backref='user',
    #                                 uselist=False,
    #                                 cascade="all, delete-orphan")

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


# class Doctor(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
#     speciality = db.Column(db.String(100))
#     experience = db.Column(db.Integer)
#     license_number = db.Column(db.String(50))
#     address = db.Column(db.String(200))
#     phone = db.Column(db.String(20))
#
#     def __repr__(self):
#         return f'<Doctor {self.id}>'
#
#
# class CosmeticFirm(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
#     company_name = db.Column(db.String(100))
#     registration_number = db.Column(db.String(50))
#     address = db.Column(db.String(200))
#     phone = db.Column(db.String(20))
#     website = db.Column(db.String(100))
#
#     def __repr__(self):
#         return f'<CosmeticFirm {self.id}>'


def init_db(app):
    """Initialize the database with the Flask app"""
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///skinhealth.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY',
                                              'default-secret-key')
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
                cursor.execute(
                    "ALTER TABLE user ADD COLUMN is_active BOOLEAN DEFAULT 1")
                conn.commit()
            conn.close()
        except Exception as e:
            print(f"Migration error: {e}")


class CosmeticProduct(db.Model):
    """Model for storing cosmetic products"""
    id = db.Column(db.String(8), primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text, nullable=False)
    product_type = db.Column(db.String(50), nullable=False)
    image_path = db.Column(db.String(200))
    price = db.Column(db.Float, default=0.0)
    ingredients = db.Column(db.Text)
    key_benefits = db.Column(db.Text)
    usage_instructions = db.Column(db.Text)
    warnings = db.Column(db.Text)
    suitable_for = db.Column(db.Text)
    ph_level = db.Column(db.String(20))
    fragrance = db.Column(db.String(50))
    comedogenic_rating = db.Column(db.Integer)
    additional_info = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    def get_ingredients_list(self):
        """Return ingredients as a list"""
        if self.ingredients:
            return json.loads(self.ingredients)
        return []

    def get_key_benefits_list(self):
        """Return key benefits as a list"""
        if self.key_benefits:
            return json.loads(self.key_benefits)
        return []

    def get_suitable_for_list(self):
        """Return suitable for skin types as a list"""
        if self.suitable_for:
            return json.loads(self.suitable_for)
        return []
