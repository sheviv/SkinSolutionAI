import sqlite3
import os


def migrate_user_table():
    """Add user_type column to user table if it doesn't exist"""
    db_path = 'instance/skinhealth.db'

    # Make sure the database exists
    if not os.path.exists(db_path):
        print(f"Database {db_path} does not exist. No migration needed.")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check if user_type column exists
    cursor.execute("PRAGMA table_info(user)")
    columns = [column[1] for column in cursor.fetchall()]

    if 'user_type' not in columns:
        print("Adding user_type column to user table...")
        # Add the column
        cursor.execute("ALTER TABLE user ADD COLUMN user_type VARCHAR(20) DEFAULT 'Пациент' NOT NULL")
        conn.commit()
        print("Migration successful!")
    else:
        print("user_type column already exists. No migration needed.")

    conn.close()


if __name__ == "__main__":
    migrate_user_table()
