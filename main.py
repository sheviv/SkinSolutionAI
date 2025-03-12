import streamlit as st
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from flask_login import login_required, current_user, logout_user
import cv2
import numpy as np
import os
import sys

# Ensure all directories are in the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from utils.image_processing import process_image, analyze_skin, detect_problem_areas
from utils.ml_model import predict_skin_condition
from utils.ai_analysis import get_image_analysis
from utils.language import get_translation, init_language, change_language, LANGUAGES
from utils.location import get_user_location_from_ip, get_doctors_in_radius
from utils.chat import init_chat, start_chat_with_doctor, render_chat_interface
from utils.database import init_db, User, db
from utils.auth import init_auth, register_user, login_user_with_credentials
from data.products import get_product_recommendations
from data.doctors import get_nearby_doctors

# Initialize Flask app
app = Flask(__name__)

# Initialize database
init_db(app)

# Initialize authentication
init_auth(app)

# Initialize language settings
init_language()

# Initialize chat system
init_chat()


# Create a translation function with current language
def t(key):
    return get_translation(key)


# Streamlit authentication forms
def show_auth_forms():
    tab1, tab2 = st.tabs(["Вход", "Регистрация"])

    with tab1:
        st.header("Вход в систему")

        # Input fields for login
        login_email = st.text_input("Электронная почта", key="login_email")
        login_password = st.text_input("Пароль", type="password", key="login_password")

        # Login button
        if st.button("Войти"):
            try:
                if not login_email or not login_password:
                    st.error("Пожалуйста, введите email и пароль")
                else:
                    # Find the user directly without using Flask-Login
                    from utils.database import User
                    user = User.query.filter_by(email=login_email).first()

                    if user and user.check_password(login_password):
                        # Set session state directly without Flask login
                        st.session_state.authenticated = True
                        st.session_state.user_id = user.id
                        st.session_state.username = user.username
                        st.session_state.registered = True

                        st.success("Вход выполнен успешно")
                        st.rerun()  # Перезагрузить страницу для отображения основной программы
                    else:
                        st.error("Неверный email или пароль")
            except Exception as e:
                st.error(f"Ошибка при входе: {str(e)}")
                import traceback
                st.error(traceback.format_exc())

    with tab2:
        st.header("Регистрация")
        st.write("Пожалуйста, заполните форму ниже для регистрации.")

        # Input fields for registration
        email = st.text_input("Электронная почта", key="reg_email")
        username = st.text_input("Имя пользователя", key="reg_username")
        password = st.text_input("Пароль", type="password", key="reg_password")
        confirm_password = st.text_input("Подтвердите пароль", type="password", key="reg_confirm_password")
        user_type = st.selectbox("Тип пользователя",
                                 ["Пациент", "Врач", "Косметическая фирма"],
                                 key="reg_user_type")

        # Register button
        if st.button("Зарегистрироваться"):
            if password != confirm_password:
                st.error("Пароли не совпадают!")
            else:
                # Call the register_user function from utils.auth
                try:
                    success, message = register_user(email, username, password, user_type)
                    if success:
                        # st.success(message)
                        st.session_state.registered = True  # Mark user as registered
                        st.rerun()  # Перезагрузить страницу для отображения основной программы
                    else:
                        st.error(message)
                except Exception as e:
                    st.error(f"Ошибка при регистрации: {str(e)}")


# Page configuration
st.set_page_config(page_title="SkinHealth AI - Professional Skin Analysis",
                   page_icon="🏥",
                   layout="wide")


# Add custom pages to sidebar if not shown automatically
# if st.session_state.get('registered', False):
#     with st.sidebar:
#         st.page_link("main.py", label="Main", icon="🏠")
#         st.page_link("pages/view_analysis.py", label="View Analysis", icon="🔍")
#         st.page_link("pages/cosmetic_login.py", label="Cosmetic Firm Portal", icon="💄")


def main():
    # Initialize session state variables if they don't exist
    if 'user_location' not in st.session_state:
        st.session_state.user_location = get_user_location_from_ip()

    if 'search_radius' not in st.session_state:
        st.session_state.search_radius = 50  # Default radius in kilometers

    # Check if chat is active
    if st.session_state.get('chat_active', False):
        # Render chat interface
        render_chat_interface()

        # Provide a way to return to the main app
        if st.button("Вернуться к анализу кожи"):
            st.session_state.chat_active = False
            st.rerun()  # Используем st.rerun() вместо st.experimental_rerun()
        return  # Exit the function early to show only chat interface

    # Show authentication forms if user is not registered
    if not st.session_state.get('registered', False):
        show_auth_forms()
        return  # Exit the function early to show only authentication forms

    # Add a settings expander in the sidebar
    with st.sidebar:
        with st.expander(t("settings")):
            # Language settings
            st.write(f"**{t('language')}**")
            language_options = list(LANGUAGES.keys())
            current_lang_name = [name for name, code in LANGUAGES.items()
                                 if code == st.session_state.language][0]
            selected_language = st.selectbox(
                t("select_language"),
                options=language_options,
                index=language_options.index(current_lang_name)
            )

            # Change language if user selects a different one
            selected_lang_code = LANGUAGES[selected_language]
            if selected_lang_code != st.session_state.language:
                change_language(selected_lang_code)

            # Location settings
            st.write(f"**{t('location_settings')}**")

            # Display current location
            user_location = st.session_state.user_location
            st.write(f"{t('your_location')}: {user_location.get('city', '')}, {user_location.get('state', '')}")

            # Option to enter location manually (simplified for demo)
            if st.button(t("detect_location")):
                st.session_state.user_location = get_user_location_from_ip()
                st.rerun()  # Используем st.rerun() вместо st.experimental_rerun()

            # Search radius setting
            st.write(f"**{t('search_radius')}**")
            radius_options = range(0, 100 + 1)
            selected_radius = st.select_slider(
                "",
                options=radius_options,
                value=st.session_state.search_radius
            )

            # Update search radius if changed
            if selected_radius != st.session_state.search_radius:
                st.session_state.search_radius = selected_radius

    st.title(t("app_title"))
    st.subheader(t("app_subtitle"))

    # Display clinic images in columns
    col1, col2 = st.columns(2)
    with col1:
        st.image(
            "https://images.unsplash.com/photo-1598300188904-6287d52746ad",
            caption=t("clinic_image_1"),
            use_container_width=True)
    with col2:
        st.image(
            "https://images.unsplash.com/photo-1690306815613-f839b74af330",
            caption=t("clinic_image_2"),
            use_container_width=True)

    # Set API keys checked to true so we don't need to check again
    if not st.session_state.get('api_keys_checked'):
        st.session_state.api_keys_checked = True

    # Image Upload Section
    st.header(t("upload_header"))
    st.info(t("upload_info"))
    uploaded_file = st.file_uploader(t("upload_button"),
                                     type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        # Convert uploaded file to image array
        file_bytes = np.asarray(bytearray(uploaded_file.read()),
                                dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        with st.spinner("Processing image and analyzing skin condition..."):
            # Process and analyze image
            processed_image = process_image(image)
            skin_features = analyze_skin(processed_image)

            # Detect problem areas in the image
            marked_image, problem_areas = detect_problem_areas(
                processed_image, skin_features)

            # Get ML model predictions and AI analysis
            from utils.ml_model import predict_skin_condition
            from utils.ml_ensemble import predict_with_ensemble
            from utils.specialized_models import predict_severe_condition, predict_child_skin_condition

            # Get predictions from all models
            ensemble_predictions = predict_with_ensemble(skin_features)

            # Store all models' predictions for display
            st.session_state.ensemble_predictions = ensemble_predictions

            # Use the most reliable model's prediction for recommendations
            most_reliable_model = ensemble_predictions.get("most_reliable_model", "Random Forest")
            ml_prediction = predict_skin_condition(skin_features) if most_reliable_model == "Random Forest" else \
                ensemble_predictions[most_reliable_model]

            # Get model type from radio button selection (defaults to standard if not set)
            selected_model_type = st.session_state.get('selected_model_type', "Standard Skin Analysis")

            # Get AI analysis for the selected model type
            ai_analysis = get_image_analysis(processed_image, selected_model_type)

            # Store problem areas in session state for display
            st.session_state.problem_areas = problem_areas
            st.session_state.marked_image = marked_image

        # Display Results
        st.header(t("analysis_results"))

        # Add model selection for different use cases
        model_type = st.radio(
            "Select analysis mode:",
            ["Standard Skin Analysis", "Severe Skin Conditions", "Child Skin Analysis"],
            horizontal=True
        )

        # Store selected model type in session state
        st.session_state.selected_model_type = model_type

        # Display appropriate explanation based on selected model
        if model_type == "Standard Skin Analysis":
            st.info("Standard analysis model focuses on common skin conditions like acne, uneven tone, and dullness.")
        elif model_type == "Severe Skin Conditions":
            st.info(
                "This specialized model analyzes severe skin conditions like psoriasis, eczema, rosacea, and severe acne.")
        else:  # Child Skin Analysis
            st.info(
                "Child skin analysis is optimized for detecting conditions common in children like atopic dermatitis, infantile acne, and cradle cap.")

        # Display the original and marked images side by side
        col_img1, col_img2 = st.columns(2)

        with col_img1:
            st.subheader(t("original_image"))
            st.image(processed_image,
                     caption=t("original_upload"),
                     use_container_width=True)

        with col_img2:
            st.subheader(t("analysis_detected"))
            st.image(marked_image,
                     use_container_width=True)

        # Display problem area details
        if problem_areas:
            st.subheader(t("problem_areas"))
            area_selection = st.selectbox(
                t("select_area"),
                options=[
                    f"{area['type']} (Area {i + 1})"
                    for i, area in enumerate(problem_areas)
                ],
                format_func=lambda x: x)

            # Find the selected area
            selected_index = int(
                area_selection.split("Area ")[-1].split(")")[0]) - 1
            selected_area = problem_areas[selected_index]

            # Display details about the selected area
            st.info(f"**{t('area_type')}** {selected_area['type']}")
            st.info(f"**{t('area_severity')}** {selected_area['severity']}")
            st.info(f"**{t('area_size')}** {selected_area['size']} pixels")
            st.info(f"**{t('area_description')}** {selected_area['description']}")

        # Create three columns for different analyses
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader(t("ml_model_analysis"))

            # Show which model is being used based on user selection
            if 'ensemble_predictions' in st.session_state:
                ensemble_preds = st.session_state.ensemble_predictions

                # Determine which model to show based on user selection
                selected_model_type = st.session_state.get('selected_model_type', "Standard Skin Analysis")

                if selected_model_type == "Standard Skin Analysis":
                    # Use standard models (Random Forest, Gradient Boosting, SVM)
                    standard_models = [name for name in ensemble_preds.keys()
                                       if name not in ["most_reliable_model", "Severe Conditions", "Child Skin"]]

                    if standard_models:
                        most_reliable_model = ensemble_preds.get("most_reliable_model", "Random Forest")
                        if most_reliable_model not in standard_models:
                            most_reliable_model = standard_models[0]

                        st.info(f"**{t('primary_model')}** {most_reliable_model} {t('used_for_recommendations')}")

                        # Add a model selection dropdown for standard models
                        selected_model = st.selectbox(t("view_other_models"), standard_models)
                    else:
                        selected_model = "Random Forest"
                        st.warning("Standard skin analysis models not available")

                elif selected_model_type == "Severe Skin Conditions":
                    # Use severe conditions model
                    if "Severe Conditions" in ensemble_preds:
                        selected_model = "Severe Conditions"
                        st.info("Using specialized model for severe skin conditions analysis")
                    else:
                        # Fallback to standard model
                        selected_model = ensemble_preds.get("most_reliable_model", "Random Forest")
                        st.warning("Severe conditions model not available, using standard model")

                else:  # Child Skin Analysis
                    # Use child skin model
                    if "Child Skin" in ensemble_preds:
                        selected_model = "Child Skin"
                        st.info("Using specialized model for children's skin analysis")
                    else:
                        # Fallback to standard model
                        selected_model = ensemble_preds.get("most_reliable_model", "Random Forest")
                        st.warning("Child skin model not available, using standard model")

                # Display the selected model's prediction
                if selected_model in ensemble_preds:
                    model_result = ensemble_preds[selected_model]

                    # Display model type badge
                    model_type = model_result.get('model_type', 'standard')
                    if model_type == 'severe_conditions':
                        st.markdown("🔍 **Severe Conditions Analysis**")
                    elif model_type == 'child_skin':
                        st.markdown("👶 **Child Skin Analysis**")

                    st.write(f"**{t('condition')}** {model_result['condition']}")
                    st.write(f"**{t('confidence')}** {model_result['confidence']}")

                # Display model probabilities as a bar chart
                if 'probabilities' in model_result:
                    import altair as alt
                    import pandas as pd

                    # Convert probabilities to dataframe for visualization
                    prob_data = []
                    for condition, prob_str in model_result['probabilities'].items():
                        prob_value = float(prob_str.rstrip('%'))
                        prob_data.append({"Condition": condition, "Probability": prob_value})

                    if prob_data:
                        df = pd.DataFrame(prob_data)
                        chart = alt.Chart(df).mark_bar().encode(
                            x=alt.X('Probability:Q', title='Probability (%)'),
                            y=alt.Y('Condition:N', title='Skin Condition'),
                            color=alt.Color('Condition:N', legend=None),
                            tooltip=['Condition', 'Probability']
                        ).properties(height=min(len(prob_data) * 40, 300))
                        st.altair_chart(chart, use_container_width=True)
            else:
                st.write(f"**{t('condition')}** {ml_prediction['condition']}")
                st.write(f"**{t('confidence')}** {ml_prediction['confidence']}")

                st.write(f"**{t('key_factors')}**")
                if 'key_factors' in ml_prediction:
                    for factor, importance in ml_prediction['key_factors']:
                        st.write(f"- {factor}: {importance:.2%}")

                with st.expander(t("detailed_probabilities")):
                    for condition, prob in ml_prediction['probabilities'].items():
                        st.write(f"- {condition}: {prob}")

        with col2:
            st.subheader(t("anthropic_analysis"))
            if ai_analysis and ai_analysis['anthropic_analysis']:
                st.markdown(ai_analysis['anthropic_analysis'])
            else:
                st.info(t("anthropic_unavailable"))

        with col3:
            st.subheader(t("openai_analysis"))
            if ai_analysis and ai_analysis['openai_analysis']:
                st.markdown(ai_analysis['openai_analysis'])
            else:
                st.info(t("openai_unavailable"))

        # Handle publish button functionality
        publish_button = st.button("Publish")
        if publish_button:
            if st.session_state.get('authenticated', False):
                from utils.published_analysis import PublishedAnalysis

                # Prepare analysis data to save
                analysis_data = {
                    'condition': ml_prediction['condition'],
                    'features': skin_features,
                    'image': processed_image
                }

                # Save the analysis
                analysis_id = PublishedAnalysis.save_analysis(
                    st.session_state.user_id,
                    analysis_data
                )

                # Show success message with ID
                st.success(f"Analysis published successfully! Share ID: {analysis_id}")
            else:
                st.warning("Please log in to publish your analysis.")

        # Display Detected characteristics with expanded metrics and descriptions
        st.subheader(t("skin_metrics"))

        # Add descriptions for each metric - using translation keys
        metric_descriptions = {
            "Tone Uniformity": t("tone_uniformity_desc"),
            "Brightness": t("brightness_desc"),
            "Texture": t("texture_desc"),
            "Spots Detected": t("spots_desc"),
            "Redness": t("redness_desc"),
            "Pigmentation": t("pigmentation_desc")
        }

        # Add reference ranges for interpretation with translated category names
        reference_ranges = {
            "Tone Uniformity": {t("uneven"): (0, 65), t("moderate"): (65, 85), t("even"): (85, 100)},
            "Brightness": {t("dull"): (0, 50), t("moderate"): (50, 75), t("bright"): (75, 100)},
            "Texture": {t("smooth"): (0, 30), t("moderate"): (30, 60), t("rough"): (60, 100)},
            "Spots Detected": {t("few"): (0, 3), t("moderate"): (3, 7), t("many"): (7, 100)},
            "Redness": {t("low"): (0, 30), t("moderate"): (30, 60), t("high"): (60, 100)},
            "Pigmentation": {t("even"): (0, 25), t("moderate"): (25, 50), t("uneven"): (50, 100)}
        }

        # Function to get interpretation based on value and ranges
        def get_interpretation(feature, value):
            if feature not in reference_ranges:
                return ""

            ranges = reference_ranges[feature]
            for category, (min_val, max_val) in ranges.items():
                if min_val <= value < max_val:
                    return category
            return "Unknown"

        # Create expandable sections for each metric
        for feature, value in skin_features.items():
            numeric_value = float(value) if isinstance(value, (float, int)) else 0
            interpretation = get_interpretation(feature, numeric_value)

            with st.expander(f"{feature}: {numeric_value:.1f} ({interpretation})", expanded=False):
                # Description
                st.markdown(f"**{t('description')}** {metric_descriptions.get(feature, 'No description available')}")

                # Reference range visualization
                if feature in reference_ranges:
                    st.markdown(f"**{t('reference_range')}**")
                    col1, col2, col3 = st.columns(3)

                    ranges = reference_ranges[feature]
                    categories = list(ranges.keys())

                    with col1:
                        st.markdown(f"**{categories[0]}**")
                        st.markdown(f"{ranges[categories[0]][0]} - {ranges[categories[0]][1]}")
                    with col2:
                        st.markdown(f"**{categories[1]}**")
                        st.markdown(f"{ranges[categories[1]][0]} - {ranges[categories[1]][1]}")
                    with col3:
                        st.markdown(f"**{categories[2]}**")
                        st.markdown(f"{ranges[categories[2]][0]} - {ranges[categories[2]][1]}")

                    # Progress bar to visualize where the value falls in the range
                    st.progress(min(numeric_value / 100.0, 1.0))

                # Recommendations based on the metric
                st.markdown(f"**{t('recommendations')}**")
                if feature == "Tone Uniformity" and numeric_value < 65:
                    st.markdown(f"- {t('rec_niacinamide')}")
                    st.markdown(f"- {t('rec_sunscreen')}")
                elif feature == "Brightness" and numeric_value < 50:
                    st.markdown(f"- {t('rec_exfoliants')}")
                    st.markdown(f"- {t('rec_vitamin_c')}")
                elif feature == "Texture" and numeric_value > 60:
                    st.markdown(f"- {t('rec_gentle_exfoliation')}")
                    st.markdown(f"- {t('rec_retinol')}")
                elif feature == "Spots Detected" and numeric_value > 3:
                    st.markdown(f"- {t('rec_targeted_treatments')}")
                    st.markdown(f"- {t('rec_sun_protection')}")
                elif feature == "Redness" and numeric_value > 30:
                    st.markdown(f"- {t('rec_centella')}")
                    st.markdown(f"- {t('rec_avoid_hot')}")
                elif feature == "Pigmentation" and numeric_value > 25:
                    st.markdown(f"- {t('rec_licorice')}")
                    st.markdown(f"- {t('rec_consistent_sunscreen')}")
                else:
                    st.markdown(f"- {t('rec_continue')}")
                    st.markdown(f"- {t('rec_prevention')}")

        # Detailed Problem Area Analysis
        st.header(t("comprehensive_analysis"))
        if problem_areas:
            problem_tabs = st.tabs([
                f"{area['type']} {i + 1}"
                for i, area in enumerate(problem_areas)
            ])

            for i, (tab, area) in enumerate(zip(problem_tabs, problem_areas)):
                with tab:
                    col1, col2 = st.columns([1, 2])

                    with col1:
                        # Extract the region of interest
                        x, y, w, h = area['bbox']
                        x, y, w, h = max(0, x), max(0, y), w, h

                        # Check if coordinates are valid
                        x_end = min(x + w, processed_image.shape[1])
                        y_end = min(y + h, processed_image.shape[0])

                        # Extract and display the region of interest
                        if x < x_end and y < y_end:
                            roi = processed_image[y:y_end, x:x_end]
                            if roi.size > 0:  # Make sure the ROI is not empty
                                # Increase quality by using full image size
                                st.image(roi,
                                         use_container_width=True,
                                         clamp=True)

                    with col2:
                        st.markdown(f"### {area['type']} {t('details')}")
                        st.markdown(f"**{t('area_severity')}** {area['severity']}")
                        st.markdown(f"**{t('area_size')}** {area['size']} pixels")
                        st.markdown(f"**{t('area_description')}**")
                        st.markdown(area['description'])

                        # Add recommendations based on the area type
                        st.markdown(f"### {t('recommended_actions')}")
                        if "Spot" in area['type']:
                            st.markdown(
                                f"- {t('rec_niacinamide')}"
                            )
                            st.markdown(
                                f"- {t('rec_sunscreen')}"
                            )
                        elif "Redness" in area['type']:
                            st.markdown(
                                f"- {t('rec_centella')}"
                            )
                            st.markdown(
                                f"- {t('rec_avoid_hot')}"
                            )
                        elif "Texture" in area['type']:
                            st.markdown(
                                f"- {t('rec_gentle_exfoliation')}"
                            )
                            st.markdown(
                                f"- {t('rec_retinol')}"
                            )
        else:
            st.info(t("no_problem_areas"))

        # Model Comparison Section
        st.header(t("model_comparison"))
        if 'ensemble_predictions' in st.session_state:
            ensemble_preds = st.session_state.ensemble_predictions
            model_names = [name for name in ensemble_preds.keys() if name != "most_reliable_model"]

            # Create a comparison table
            import pandas as pd

            comparison_data = []
            for model_name in model_names:
                if model_name in ensemble_preds:
                    model_data = ensemble_preds[model_name]
                    comparison_data.append({
                        "Model": model_name,
                        t("condition"): model_data['condition'],
                        t("confidence"): model_data['confidence']
                    })

            if comparison_data:
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True)

                # Highlight the most reliable model
                most_reliable = ensemble_preds.get("most_reliable_model", "Random Forest")
                st.success(t("most_reliable").format(model=most_reliable))

        # Product Recommendations section
        st.header(t("recommended_products"))
        from utils.product_recommendation import get_database_products
        products = get_database_products()

        for product in products:
            with st.expander(f"🏥 {product['name']} - ${product['price']}",
                             expanded=False):
                # Basic Information
                st.write(f"**{t('description')}** {product['description']}")

                # Ingredients Section
                st.subheader(f"🧪 {t('ingredients')}")
                st.write(", ".join(product['ingredients']))

                # Key Benefits
                st.subheader(f"✨ {t('key_benefits')}")
                for benefit in product['key_benefits']:
                    st.write(f"- {benefit}")

                # Usage Instructions
                st.subheader(f"📝 {t('how_to_use')}")

                # Handle usage_instructions as either string or dictionary
                if isinstance(product['usage_instructions'], dict):
                    # If it's a dictionary, access the keys
                    st.write(
                        f"**{t('frequency')}** {product['usage_instructions'].get('frequency', 'Not specified')}"
                    )
                    st.write(f"**{t('steps')}**")
                    for step in product['usage_instructions'].get('steps', []):
                        st.write(f"- {step}")
                    if product['usage_instructions'].get('warnings'):
                        st.warning(
                            f"⚠️ **{t('warning')}** {product['usage_instructions'].get('warnings', '')}"
                        )
                else:
                    # If it's a string, display it directly
                    st.write(product['usage_instructions'])

                # Skin Compatibility
                st.subheader(f"👥 {t('suitable_for')}")
                st.write(", ".join(product.get('suitable_for', product.get('skin_compatibility', []))))

                # Ingredient Analysis
                st.subheader(f"🔬 {t('ingredient_analysis')}")

                # Active Ingredients
                st.write(f"**{t('active_ingredients')}**")
                for ingredient, description in product['ingredient_analysis'][
                    'active_ingredients'].items():
                    st.markdown(f"**💊 {ingredient}**")
                    st.write(description)
                    st.divider()

                # Potential Irritants
                if product['ingredient_analysis']['potential_irritants']:
                    st.write(f"**⚠️ {t('potential_irritants')}**")
                    for ingredient, warning in product['ingredient_analysis'][
                        'potential_irritants'].items():
                        st.markdown(f"**⚠️ {ingredient}**")
                        st.warning(warning)

                # Additional Information
                st.write(f"**{t('additional_info')}**")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(
                        f"**{t('ph_level')}** {product['ingredient_analysis'].get('ph_level', 'Not specified')}"
                    )
                    st.write(
                        f"**{t('fragrance_free')}** {product['ingredient_analysis'].get('fragrance_free', 'Not specified')}"
                    )
                with col2:
                    st.write(
                        f"**{t('comedogenic')}** {product['ingredient_analysis']['comedogenic_rating']}/5"
                    )
                    st.caption(t("comedogenic_scale"))

                # Certifications
                if product['ingredient_analysis'].get('certifications'):
                    st.write(f"**🏆 {t('certifications')}**")
                    st.write(", ".join(
                        product['ingredient_analysis']['certifications']))

        # Doctor Directory section
        st.header(t("consult_professional"))

        # Get user location from session state
        user_location = st.session_state.user_location
        user_lat = user_location.get('latitude')
        user_lon = user_location.get('longitude')
        search_radius = st.session_state.search_radius

        # Get all doctors and filter by radius
        all_doctors = get_nearby_doctors()
        doctors_in_radius = get_doctors_in_radius(user_lat, user_lon, search_radius, all_doctors)

        # Display radius information
        st.info(t("doctors_in_radius").format(radius=search_radius))

        if not doctors_in_radius:
            st.warning(f"No doctors found within {search_radius}km. Try increasing your search radius in settings.")

        # Display doctors within radius
        for doctor in doctors_in_radius:
            with st.expander(
                    f"👨‍⚕️ Dr. {doctor['name']} - {doctor['speciality']} ({doctor['distance']} km)",
                    expanded=False):

                col1, col2 = st.columns([2, 1])

                with col1:
                    st.write(f"**{t('experience')}** {doctor['experience']} years")
                    st.write(f"**{t('location')}** {doctor['location']}")
                    st.write(f"**{t('contact')}** {doctor['contact']}")
                    st.write(f"**{t('email')}** {doctor['email']}")
                    st.write(f"**{t('available_hours')}** {doctor['available_hours']}")
                    st.write(f"**{t('distance')}** {doctor['distance']} km")

                with col2:
                    # Add button to open chat with this doctor
                    if st.button(t("send_message"), key=f"msg_btn_{doctor['name']}"):
                        # Start chat with this doctor
                        start_chat_with_doctor(doctor)
                        st.rerun()  # Используем st.rerun() вместо st.experimental_rerun()


if __name__ == "__main__":
    main()