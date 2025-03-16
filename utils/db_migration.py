
import os
import sys
from flask import Flask
from database import db, User, Doctor
import sqlite3

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Create temporary Flask app for database work
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///../instance/skinhealth.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

with app.app_context():
    # Drop and recreate all tables
    print("Recreating database tables...")
    db.drop_all()
    db.create_all()
    print("Database tables recreated successfully!")

if __name__ == "__main__":
    print("Database migration completed!")
