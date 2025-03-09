import streamlit as st
import datetime
import uuid
import os
import cv2
import numpy as np
import json
from utils.database import db, CosmeticProduct
from flask import Flask


class ProductManager:
    """Handles saving and retrieving cosmetic products"""

    @staticmethod
    def save_product(user_id, product_data, image=None):
        """Save product data to database and image to file system"""
        # Generate unique ID for the product
        product_id = str(uuid.uuid4())[:8]

        # Save image to file system if provided, but don't require it
        image_path = None
        if image is not None and isinstance(image, np.ndarray) and image.size > 0:
            image_path = f"data/product_images/{product_id}.jpg"
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            # Make sure image is valid before saving
            try:
                cv2.imwrite(image_path, image)
            except Exception as e:
                st.warning(f"Could not save image: {str(e)}")
                # Continue anyway - image is optional

        # Convert ingredients to JSON string if present
        ingredients_json = None
        if 'ingredients' in product_data and product_data['ingredients']:
            ingredients_json = json.dumps(product_data['ingredients'])

        # Create database record
        try:
            # Create app context
            app = Flask(__name__)
            base_dir = os.path.abspath(
                os.path.dirname(os.path.dirname(__file__)))
            db_path = os.path.join(base_dir, 'instance', 'skinhealth.db')
            app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
            app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
            db.init_app(app)

            with app.app_context():
                new_product = CosmeticProduct(
                    id=product_id,
                    user_id=user_id,
                    name=product_data.get('name', 'Unnamed Product'),
                    description=product_data.get('description', ''),
                    product_type=product_data.get('product_type', 'Other'),
                    image_path=image_path,
                    price=product_data.get('price', 0.0),
                    ingredients=ingredients_json,
                    timestamp=datetime.datetime.now())

                # Add to database and commit
                db.session.add(new_product)
                db.session.commit()

                return product_id
        except Exception as e:
            # If database operation fails, rollback
            db.session.rollback()
            st.error(f"Error saving product: {str(e)}")
            return None

    @staticmethod
    def get_company_products(user_id):
        """Get all products for a specific company"""
        try:
            # Create app context with proper path to database
            app = Flask(__name__)
            base_dir = os.path.abspath(
                os.path.dirname(os.path.dirname(__file__)))
            db_path = os.path.join(base_dir, 'instance', 'skinhealth.db')
            app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
            app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
            db.init_app(app)

            with app.app_context():
                products = CosmeticProduct.query.filter_by(
                    user_id=user_id).order_by(
                        CosmeticProduct.timestamp.desc()).all()

                if products:
                    # Convert from database models to dictionary
                    return {
                        product.id: {
                            'id': product.id,
                            'name': product.name,
                            'description': product.description,
                            'product_type': product.product_type,
                            'image_path': product.image_path,
                            'price': product.price,
                            'ingredients': product.get_ingredients_list(),
                            'timestamp': product.timestamp.isoformat()
                        }
                        for product in products
                    }
                return {}
        except Exception as e:
            st.error(f"Error retrieving products: {str(e)}")
            return {}

    @staticmethod
    def get_product(product_id):
        """Get a specific product by ID"""
        try:
            # Create app context with proper path to database
            app = Flask(__name__)
            base_dir = os.path.abspath(
                os.path.dirname(os.path.dirname(__file__)))
            db_path = os.path.join(base_dir, 'instance', 'skinhealth.db')
            app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
            app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
            db.init_app(app)

            with app.app_context():
                product = CosmeticProduct.query.filter_by(
                    id=product_id).first()

                if product:
                    return {
                        'id': product.id,
                        'name': product.name,
                        'description': product.description,
                        'product_type': product.product_type,
                        'image_path': product.image_path,
                        'price': product.price,
                        'ingredients': product.get_ingredients_list(),
                        'timestamp': product.timestamp.isoformat()
                    }
                return None
        except Exception as e:
            st.error(f"Error retrieving product: {str(e)}")
            return None

    @staticmethod
    def delete_product(product_id):
        """Delete a product by ID"""
        try:
            # Create app context with proper path to database
            app = Flask(__name__)
            base_dir = os.path.abspath(
                os.path.dirname(os.path.dirname(__file__)))
            db_path = os.path.join(base_dir, 'instance', 'skinhealth.db')
            app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
            app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
            db.init_app(app)

            with app.app_context():
                product = CosmeticProduct.query.filter_by(
                    id=product_id).first()

                if product:
                    # Delete image if exists
                    if product.image_path and os.path.exists(
                            product.image_path):
                        os.remove(product.image_path)

                    # Delete database record
                    db.session.delete(product)
                    db.session.commit()
                    return True
                return False
        except Exception as e:
            db.session.rollback()
            st.error(f"Error deleting product: {str(e)}")
            return False
