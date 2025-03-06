import os
import sqlite3

def add_is_active_column():
    """Add is_active column to the user table if it doesn't exist"""
    # Connect to the database
    conn = sqlite3.connect('instance/skinhealth.db')
    cursor = conn.cursor()

    # Check if the column exists
    cursor.execute("PRAGMA table_info(user)")
    columns = [column[1] for column in cursor.fetchall()]

    # Add the column if it doesn't exist
    if 'is_active' not in columns:
        cursor.execute("ALTER TABLE user ADD COLUMN is_active BOOLEAN DEFAULT 1")
        conn.commit()

    # Проверка и добавление колонки user_type
    if 'user_type' not in columns:
        cursor.execute("ALTER TABLE user ADD COLUMN user_type VARCHAR(20) DEFAULT 'Пациент'")
        conn.commit()


    # Close the connection
    conn.close()

if __name__ == "__main__":
    add_is_active_column()