import os
import sys

# Add the parent directory to sys.path to make the utils module importable
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from flask import Flask
from utils.database import db, CosmeticProduct


def run_migration():
    """Run migration to create cosmetic products table"""
    app = Flask(__name__)

    # Use absolute path to database
    db_path = os.path.join(parent_dir, 'instance', 'skinhealth.db')
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    db.init_app(app)

    with app.app_context():
        # Create tables
        db.create_all()
        print("Cosmetic products table migration successful!")


if __name__ == "__main__":
    run_migration()
