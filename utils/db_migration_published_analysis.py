import os
import sys
from flask import Flask

# Ensure all directories are in the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils.database import db, init_db, PublishedAnalysis

# Initialize Flask app and database
app = Flask(__name__)
init_db(app)


def migrate_published_analysis_table():
    """Create the published_analysis table if it doesn't exist"""
    try:
        # Create the table if it doesn't exist
        with app.app_context():
            db.create_all()

            # Check if we need to migrate data from session state to database
            # This would be done in the main app when users access it

            print("Published analysis table migration successful!")
    except Exception as e:
        print(f"Migration error: {str(e)}")


if __name__ == "__main__":
    migrate_published_analysis_table()
