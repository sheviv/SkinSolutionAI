from email.policy import default

import streamlit as st
import sys
import os
import numpy as np
import cv2
from PIL import Image
import io

# Ensure all directories are in the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils.database import User
from utils.product_management import ProductManager
from flask import Flask

# Initialize a mini Flask app for database context
app = Flask(__name__)

# Fix the database path to point to the correct location
# Use absolute path to ensure the database is found regardless of where the script is run from
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
db_path = os.path.join(base_dir, 'instance', 'skinhealth.db')
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

from utils.database import db

db.init_app(app)

# Create product_images directory if it doesn't exist
os.makedirs("data/product_images", exist_ok=True)

# Page configuration
st.set_page_config(page_title="Cosmetic Firm Portal - SkinHealth AI",
                   page_icon="💄",
                   layout="wide")

st.title("Cosmetic Firm Portal")
st.subheader("Manage your skincare products")

# Initialize session state for cosmetic firm login
if 'cosmetic_firm_logged_in' not in st.session_state:
    st.session_state.cosmetic_firm_logged_in = False
if 'cosmetic_firm_id' not in st.session_state:
    st.session_state.cosmetic_firm_id = None
if 'cosmetic_firm_name' not in st.session_state:
    st.session_state.cosmetic_firm_name = None
if 'product_image' not in st.session_state:
    st.session_state.product_image = None


# Function to convert uploaded file to OpenCV image
def load_image(image_file):
    if image_file is not None:
        # Convert the file to an opencv image
        file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        return image
    return None


# Main content
if not st.session_state.cosmetic_firm_logged_in:
    # Login form
    with st.form("cosmetic_login_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

        if submitted:
            if email and password:
                # Authenticate cosmetic firm
                with app.app_context():
                    user = User.query.filter_by(email=email).first()

                    if user and user.check_password(password):
                        if user.user_type == "Косметическая фирма":
                            # Set session state
                            st.session_state.cosmetic_firm_logged_in = True
                            st.session_state.cosmetic_firm_id = user.id
                            st.session_state.cosmetic_firm_name = user.username
                            st.success(f"Welcome, {user.username}!")
                            st.rerun()
                        else:
                            st.error(
                                "Only cosmetics firm can access this page.")
                    else:
                        st.error(
                            "Invalid email or password, or this account is not registered as a cosmetic firm."
                        )
            else:
                st.error("Please enter both email and password.")

    # Add registration info
    st.info(
        "Not registered? Please register through the main application first.")

else:
    # Display cosmetic firm dashboard
    st.write(f"## Welcome, {st.session_state.cosmetic_firm_name}!")

    # Add tabs for different features
    tab1, tab2 = st.tabs(["Product Management", "Analytics"])

    with tab1:
        # Create columns for product list and product form
        col1, col2 = st.columns([1, 1])

        with col1:
            st.header("Add New Product")

            # Product form
            with st.form("product_form"):
                product_name = st.text_input("Product Name")
                product_description = st.text_area("Product Description")
                product_type = st.selectbox("Product Type", (
                    "Anti-Acne", "Anti-Oily Skin", "Moisturizer", "Cleanser",
                    "Sunscreen", "Serum", "Other"
                ), index=None, placeholder="Select product type.")
                # Price input with two options
                # price_input_method = st.radio(
                #     "Price input method:", ["Number slider", "Manual entry"])

                price_str = st.text_input("Price ($)", value="0.00")
                try:
                    product_price = float(price_str.replace(',', '.'))
                except ValueError:
                    product_price = 0.0

                # Ingredients input (comma-separated)
                ingredients_input = st.text_area(
                    "Ingredients (comma-separated)")

                # Image upload
                uploaded_file = st.file_uploader("Choose product image...",
                                                 type=["jpg", "jpeg", "png"])

                # Submit button
                submitted = st.form_submit_button("Add Product")

                if submitted:
                    # Only check for product name as the essential field
                    if not product_name: # and not product_description and not product_type:
                        st.error("Please provide a Properties")
                    else:
                        # Process image if uploaded
                        product_image = None
                        if uploaded_file is not None:
                            product_image = load_image(uploaded_file)

                        # Process ingredients
                        ingredients_list = []
                        if ingredients_input:
                            ingredients_list = [
                                item.strip()
                                for item in ingredients_input.split(",")
                                if item.strip()
                            ]

                        # Prepare product data
                        product_data = {
                            'name': product_name,
                            'description': product_description,
                            'product_type': product_type,
                            'price': product_price,
                            'ingredients': ingredients_list
                        }

                        # Save product
                        product_id = ProductManager.save_product(
                            st.session_state.cosmetic_firm_id, product_data,
                            product_image)

                        if product_id:
                            st.success(
                                f"Product added successfully! ID: {product_id}"
                            )
                            st.rerun()
                        else:
                            st.error(
                                "Failed to add product. Please try again.")

        with col2:
            st.header("Your Products")

            # Get company products
            products = ProductManager.get_company_products(
                st.session_state.cosmetic_firm_id)

            if not products:
                st.info(
                    "You haven't added any products yet. Use the form on the left to add your first product!"
                )
            else:
                # Display products in a grid
                for i, (product_id, product) in enumerate(products.items()):
                    with st.container():
                        st.subheader(product['name'])

                        # Display product type as a badge
                        st.info(f"Type: {product['product_type']}")

                        # Display price
                        st.write(f"Price: ${product['price']:.2f}")

                        # Display image if available
                        image_path = product.get('image_path')
                        if image_path and os.path.exists(image_path):
                            image = cv2.imread(image_path)
                            if image is not None:
                                # Convert from BGR to RGB
                                image_rgb = cv2.cvtColor(
                                    image, cv2.COLOR_BGR2RGB)
                                st.image(image_rgb, use_container_width=True)

                        # Display product description (truncated)
                        description = product.get('description', '')
                        if len(description) > 100:
                            st.write(f"{description[:100]}...")
                        else:
                            st.write(description)

                        # Display added date
                        st.caption(
                            f"Added: {product.get('timestamp', '')[:10]}")

                        # View details button
                        if st.button("View Details", key=f"view_{product_id}"):
                            st.session_state.selected_product_id = product_id
                            st.rerun()

                        # Delete product button
                        if st.button("Delete", key=f"delete_{product_id}"):
                            if ProductManager.delete_product(product_id):
                                st.success("Product deleted successfully!")
                                st.rerun()
                            else:
                                st.error(
                                    "Failed to delete product. Please try again."
                                )

                        st.divider()

            # Show product details if selected
            if 'selected_product_id' in st.session_state:
                product_id = st.session_state.selected_product_id
                product = ProductManager.get_product(product_id)

                if product:
                    st.header("Product Details")

                    # Create columns for image and details
                    detail_col1, detail_col2 = st.columns([1, 1])

                    with detail_col1:
                        # Display the image if available
                        img_path = product.get('image_path')
                        if img_path and os.path.exists(img_path):
                            img = cv2.imread(img_path)
                            if img is not None:
                                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                st.image(img_rgb,
                                         caption="Product Image",
                                         use_container_width=True)

                    with detail_col2:
                        # Display product information
                        st.markdown(f"### {product['name']}")
                        st.info(f"Type: {product['product_type']}")
                        st.write(f"Price: ${product['price']:.2f}")
                        st.write(f"**Description:** {product['description']}")

                        # Display ingredients
                        if product.get('ingredients'):
                            st.write("**Ingredients:**")
                            for ingredient in product['ingredients']:
                                st.write(f"- {ingredient}")

                        st.write(
                            f"**Added on:** {product.get('timestamp', '')[:10]}"
                        )

                    # Button to close details
                    if st.button("Close Details"):
                        del st.session_state.selected_product_id
                        st.rerun()

    with tab2:
        st.header("Analytics Dashboard")
        st.info("View insights about how your products are performing.")

        # Placeholder for analytics
        st.write("Analytics features coming soon...")

    # Logout button
    if st.button("Logout"):
        st.session_state.cosmetic_firm_logged_in = False
        st.session_state.cosmetic_firm_id = None
        st.session_state.cosmetic_firm_name = None
        st.success("Logged out successfully!")
        st.rerun()
