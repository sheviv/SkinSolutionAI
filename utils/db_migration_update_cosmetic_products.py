import os
import sys
import sqlite3

# Add the parent directory to sys.path to make the utils module importable
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from flask import Flask
from utils.database import db, CosmeticProduct


def run_migration():
    """Run migration to update cosmetic products table with new fields"""
    app = Flask(__name__)

    # Use absolute path to database
    db_path = os.path.join(parent_dir, 'instance', 'skinhealth.db')
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    db.init_app(app)

    with app.app_context():
        # First, check if the table exists and has the required columns
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Check if the cosmetic_product table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='cosmetic_product'")
        table_exists = cursor.fetchone() is not None

        if table_exists:
            # Check if the table has all required columns
            cursor.execute("PRAGMA table_info(cosmetic_product)")
            columns = [column[1] for column in cursor.fetchall()]

            required_columns = [
                'key_benefits', 'usage_instructions', 'warnings',
                'suitable_for', 'ph_level', 'fragrance',
                'comedogenic_rating', 'additional_info'
            ]

            missing_columns = [col for col in required_columns if col not in columns]

            if missing_columns:
                print(f"Missing columns: {missing_columns}")
                # Drop the table to recreate it with all columns
                cursor.execute("DROP TABLE cosmetic_product")
                conn.commit()
                print("Dropped existing table to recreate with all columns")

        conn.close()

        # Create all tables including the updated cosmetic_product table
        db.create_all()
        print("Cosmetic products table updated successfully!")


if __name__ == "__main__":
    run_migration()
