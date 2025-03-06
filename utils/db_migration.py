import os
import sys
from flask import Flask
from utils.database import db, User, Doctor, CosmeticFirm
import sqlite3

# Добавить текущую директорию в путь Python
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Создаем временное приложение Flask для работы с базой данных
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///../instance/skinhealth.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

with app.app_context():
    # Удаляем существующие таблицы и создаем их заново
    print("Recreating database tables...")
    db.drop_all()
    db.create_all()
    print("Database tables recreated successfully!")

if __name__ == "__main__":
    print("Database migration completed!")
