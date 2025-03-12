import os
from flask import Flask
from utils.database import db, CosmeticProduct
import streamlit as st


def get_database_products():
    """Get products from the database to recommend to users"""
    try:
        # Create app context with proper path to database
        app = Flask(__name__)
        base_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
        db_path = os.path.join(base_dir, 'instance', 'skinhealth.db')
        app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
        app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
        db.init_app(app)

        with app.app_context():
            # Get all products from the database
            products = CosmeticProduct.query.all()

            if products:
                # Convert from database models to dictionary format
                product_list = []
                for product in products:
                    product_dict = {
                        'id': product.id,
                        'name': product.name,
                        'description': product.description,
                        'product_type': product.product_type,
                        'image_path': product.image_path,
                        'price': product.price,
                        'ingredients': product.get_ingredients_list(),
                        'key_benefits': product.get_key_benefits_list(),
                        'usage_instructions': product.usage_instructions,
                        'warnings': product.warnings,
                        'suitable_for': product.get_suitable_for_list(),
                        'ph_level': product.ph_level,
                        'fragrance': product.fragrance,
                        'comedogenic_rating': product.comedogenic_rating,
                        'additional_info': product.additional_info,
                        'timestamp': product.timestamp.isoformat()
                    }
                    product_list.append(product_dict)
                return product_list
            return []
    except Exception as e:
        st.error(f"Error retrieving products from database: {str(e)}")
        return []


import sqlite3


def get_database_products():
    """Get products from the database that can be recommended to users"""
    try:
        # Connect to the SQLite database
        conn = sqlite3.connect('instance/skinhealth.db')
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Query to get all cosmetic products
        cursor.execute('''
            SELECT * FROM cosmetic_product
        ''')

        # Fetch all records
        db_products = []
        for row in cursor.fetchall():
            product = {
                'id': row['id'],
                'name': row['name'],
                'description': row['description'],
                'price': row['price'],
                'product_type': row['product_type'],
                'image_path': row['image_path'],
                'ingredients': row['ingredients'].split(',') if row['ingredients'] else [],
                'key_benefits': row['key_benefits'].split(',') if row['key_benefits'] else [],
                'usage_instructions': row['usage_instructions'],
                'suitable_for': row['suitable_for'].split(',') if row['suitable_for'] else [],
                'ph_level': row['ph_level'],
                'comedogenic_rating': row['comedogenic_rating'],
                'fragrance': row['fragrance'],
                'warnings': row['warnings']
            }
            db_products.append(product)

        conn.close()
        return db_products

    except Exception as e:
        print(f"Error fetching products from database: {e}")
        return []
