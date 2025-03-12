import streamlit as st
import pandas as pd
import os
from PIL import Image
from data.products import get_product_recommendations
from utils.product_recommendation import get_database_products

def display_product(product):
    """Display a single product in a formatted card"""
    col1, col2 = st.columns([1, 3])

    # Display image if available
    with col1:
        if product.get('image_path') and os.path.exists(product.get('image_path')):
            try:
                img = Image.open(product.get('image_path'))
                st.image(img, width=150)
            except Exception:
                st.image("https://via.placeholder.com/150x150?text=No+Image", width=150)
        else:
            st.image("https://via.placeholder.com/150x150?text=No+Image", width=150)

    # Display product details
    with col2:
        st.subheader(product.get('name', 'Unnamed Product'))
        st.write(product.get('description', ''))

        # Price
        if product.get('price'):
            st.write(f"**Price:** ${product.get('price', 0.0):.2f}")

        # Product type
        if product.get('product_type'):
            st.write(f"**Type:** {product.get('product_type')}")

        # Show expandable sections for more details
        with st.expander("Key Benefits"):
            benefits = product.get('key_benefits', [])
            if benefits:
                for benefit in benefits:
                    st.write(f"• {benefit}")
            else:
                st.write("No key benefits listed")

        with st.expander("Ingredients"):
            ingredients = product.get('ingredients', [])
            if ingredients:
                for ingredient in ingredients:
                    st.write(f"• {ingredient}")
            else:
                st.write("No ingredients listed")

        if product.get('usage_instructions'):
            with st.expander("How to Use"):
                st.write(product.get('usage_instructions'))

        if product.get('suitable_for'):
            with st.expander("Suitable For"):
                suitable_for = product.get('suitable_for', [])
                if suitable_for:
                    for skin_type in suitable_for:
                        st.write(f"• {skin_type}")
                else:
                    st.write("Suitable for all skin types")

        # Additional technical information
        with st.expander("Technical Details"):
            if product.get('ph_level'):
                st.write(f"**pH Level:** {product.get('ph_level')}")
            if product.get('comedogenic_rating') is not None:
                st.write(f"**Comedogenic Rating:** {product.get('comedogenic_rating')}/5")
            if product.get('fragrance'):
                st.write(f"**Fragrance:** {product.get('fragrance')}")
            if product.get('warnings'):
                st.write(f"**Warnings:** {product.get('warnings')}")

def app():
    st.title("Recommended Skincare Products")

    # Get database products first
    db_products = get_database_products()

    # Get some sample recommendations as fallback
    skin_conditions = [
        "Acne-Prone Skin",
        "Uneven Skin Tone",
        "Dull Skin",
        "Healthy Skin"
    ]

    st.write("""
    Below you'll find skincare products tailored to different skin concerns. 
    These products are from our verified brands and partners.
    """)

    # Create tabs for different product sources
    tab1, tab2 = st.tabs(["Database Products", "Sample Recommendations"])

    # Tab 1: Products from the database
    with tab1:
        if db_products:
            st.success(f"Found {len(db_products)} products in our database")

            # Group products by product type
            product_types = {}
            for product in db_products:
                product_type = product.get('product_type', 'Other')
                if product_type not in product_types:
                    product_types[product_type] = []
                product_types[product_type].append(product)

            # Display products by type
            for product_type, products in product_types.items():
                st.subheader(f"{product_type} ({len(products)})")
                for product in products:
                    with st.container():
                        st.markdown("---")
                        display_product(product)
        else:
            st.info("No products found in the database. Please check the 'Sample Recommendations' tab for examples.")
            st.warning("If you are a cosmetic company, please log in to add your products to our database.")

    # Tab 2: Sample recommendations
    with tab2:
        # Create subtabs for each skin condition
        condition_tabs = st.tabs(skin_conditions)

        for i, condition in enumerate(skin_conditions):
            with condition_tabs[i]:
                st.write(f"Recommended products for {condition}:")

                # Get recommendations for this condition
                recommendations = get_product_recommendations(condition)

                # Display each recommended product
                for product in recommendations:
                    with st.container():
                        st.markdown("---")
                        display_product(product)

# Hide this page from the sidebar
st.set_page_config(page_title="Recommended Skincare Products", page_icon="🧴", initial_sidebar_state="collapsed")
st.sidebar.markdown('<style>div[data-testid="stSidebarNav"] ul li:has(a[href*="product_recommendations"]) {display: none;}</style>', unsafe_allow_html=True)

if __name__ == "__main__":
    app()