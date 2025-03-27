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
from utils.ml_ensemble import predict_with_ensemble
from utils.specialized_models import predict_severe_condition, predict_child_skin_condition, predict_skin_problems, predict_skin_care
from utils.ai_analysis import get_image_analysis # Modification needed here to remove Anthropic functions from this file.
from utils.language import get_translation, init_language, change_language, LANGUAGES
from utils.location import get_user_location_from_ip, get_doctors_in_radius, get_all_doctors
from utils.chat import init_chat, start_chat_with_doctor, render_chat_interface
from utils.database import init_db, User, db
from utils.auth import init_auth, register_user, login_user_with_credentials
from utils.product_recommendation import get_database_products
from data.doctors import get_nearby_doctors

# Initialize session state variables if they don't exist
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'registered' not in st.session_state:
    st.session_state.registered = False
if 'api_keys_checked' not in st.session_state:
    st.session_state.api_keys_checked = False
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'username' not in st.session_state:
    st.session_state.username = None
if 'language' not in st.session_state:
    st.session_state.language = "en"
if 'chat_active' not in st.session_state:
    st.session_state.chat_active = False
if 'selected_model_type' not in st.session_state:
    st.session_state.selected_model_type = "Standard Skin Analysis"

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
        login_password = st.text_input("Пароль",
                                       type="password",
                                       key="login_password")

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
                        st.rerun(
                        )  # Reload the page to display the main program
                    else:
                        st.error("Неверный email или пароль")
            except Exception as e:
                st.error(f"Ошибка при входе: {str(e)}")
                import traceback
                st.error(traceback.format_exc())

    with (tab2):
        st.header("Регистрация")
        st.write("Пожалуйста, заполните форму ниже для регистрации.")

        # Input fields for registration
        email = st.text_input("Электронная почта", key="reg_email")
        username = st.text_input("Имя пользователя", key="reg_username")
        password = st.text_input("Пароль", type="password", key="reg_password")
        confirm_password = st.text_input("Подтвердите пароль",
                                         type="password",
                                         key="reg_confirm_password")
        user_type = st.selectbox("Тип пользователя",
                                 ["Пациент", "Врач", "Косметическая фирма"],
                                 key="reg_user_type")

        if user_type in ["Врач", "Doctor"]:
            speciality = st.text_input("Специальность", key="reg_speciality")
            experience = st.text_input("Опыт", key="reg_experience")
            license_number = st.text_input("Лицензия",
                                           key="reg_license_number")
            address = st.text_input("Лицензия", key="reg_address")
            phone = st.text_input("Номер телефона", key="reg_phone")
            affordable_hours = st.text_input("Рабочие часы", key="reg_hours")
            doctor_data = {
                'speciality': speciality.strip(),
                'license_number': license_number.strip(),
                'phone': phone.strip(),
                'affordable_hours': affordable_hours.strip(),
                'experience': int(experience) if experience.isdigit() else 0,
                'address': address.strip() if address else ''
            }

        # Register button
        if st.button("Зарегистрироваться"):
            if password != confirm_password:
                st.error("Пароли не совпадают!")
            else:
                # Call the register_user function from utils.auth
                try:
                    if user_type in ["Врач", "Doctor"]:
                        success, message = register_user(
                            email, username, password, user_type, doctor_data)
                    else:
                        success, message = register_user(
                            email, username, password, user_type)
                    if success:
                        # st.success(message)
                        st.session_state.registered = True  # Mark user as registered
                        st.rerun(
                        )  # Reload the page to display the main program
                    else:
                        st.error(message)
                except Exception as e:
                    st.error(f"Ошибка при регистрации: {str(e)}")


# Page configuration
st.set_page_config(page_title="SkinHealth AI - Professional Skin Analysis",
                   page_icon="🏥",
                   layout="wide")


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
            st.rerun()
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
            current_lang_name = [
                name for name, code in LANGUAGES.items()
                if code == st.session_state.language
            ][0]
            selected_language = st.selectbox(
                t("select_language"),
                options=language_options,
                index=language_options.index(current_lang_name))

            # Change language if user selects a different one
            selected_lang_code = LANGUAGES[selected_language]
            if selected_lang_code != st.session_state.language:
                change_language(selected_lang_code)

            # Location settings
            st.write(f"**{t('location_settings')}**")

            # Display current location
            user_location = st.session_state.user_location
            st.write(
                f"{t('your_location')}: {user_location.get('city', '')}, {user_location.get('state', '')}"
            )

            # Option to enter location manually (simplified for demo)
            if st.button(t("detect_location")):
                st.session_state.user_location = get_user_location_from_ip()
                st.rerun()

            # Search radius setting
            st.write(f"**{t('search_radius')}**")
            radius_options = list(range(0, 100 + 1))
            selected_radius = st.select_slider(
                "",
                options=radius_options,
                value=st.session_state.search_radius)

            # Update search radius if changed
            if selected_radius != st.session_state.search_radius:
                st.session_state.search_radius = selected_radius

    st.title(t("app_title"))
    st.subheader(t("app_subtitle"))
    # st.image("data/logo/1.png", width=100, caption=t("clinic_image_1"), use_container_width=True)
    left_co, cent_co, last_co = st.columns(3)
    with cent_co:
        st.image("data/logo/1.png", width=400, caption=t("clinic_image_1"))


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

        # Check if image contains a body part
        from utils.body_part_detection import is_body_part

        if not is_body_part(image):
            st.error(
                "Please upload a photo of a body part (skin, face, hand, etc.). The uploaded image doesn't appear to contain a body part."
            )
            st.stop()

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
            most_reliable_model = ensemble_predictions.get(
                "most_reliable_model", "Random Forest")
            ml_prediction = predict_skin_condition(skin_features) if most_reliable_model == "Random Forest" else \
                ensemble_predictions[most_reliable_model]

            # Get model type from radio button selection (defaults to standard if not set)
            selected_model_type = st.session_state.get(
                'selected_model_type', "Standard Skin Analysis")

            # Get AI analysis for the selected model type
            ai_analysis = get_image_analysis(processed_image,
                                             selected_model_type)

            # Store problem areas in session state for display
            st.session_state.problem_areas = problem_areas
            st.session_state.marked_image = marked_image

        # Display Results
        st.header(t("analysis_results"))

        # Add model selection for different use cases
        model_type = st.radio("Select analysis mode:", [
            "Standard Skin Analysis", "Severe Skin Conditions",
            "Child Skin Analysis"
        ],
                              horizontal=True,
                              key="analysis_mode_selection")

        # Store selected model type in session state and trigger update
        if "analysis_mode_selection" in st.session_state:
            st.session_state.selected_model_type = st.session_state.analysis_mode_selection

        # Display appropriate explanation based on selected model
        if model_type == "Standard Skin Analysis":
            st.info(
                "Standard analysis model focuses on common skin conditions like acne, uneven tone, and dullness."
            )
        elif model_type == "Severe Skin Conditions":
            st.info(
                "This specialized model analyzes severe skin conditions like psoriasis, eczema, rosacea, and severe acne."
            )
        else:  # Child Skin Analysis
            st.info(
                "Child skin analysis is optimized for detecting conditions common in children like atopic dermatitis, infantile acne, and cradle cap."
            )

        # Display the original and marked images side by side
        col_img1, col_img2 = st.columns(2)

        with col_img1:
            st.subheader(t("original_image"))
            st.image(processed_image,
                     caption=t("original_upload"),
                     use_container_width=True)

        with col_img2:
            st.subheader(t("analysis_detected"))
            st.image(marked_image, use_container_width=True)

        # Create two columns for different analyses
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader(t("ml_model_analysis"))

            # Show which model is being used based on user selection
            if 'ensemble_predictions' in st.session_state:
                ensemble_preds = st.session_state.ensemble_predictions

                # Determine which model to show based on user selection
                selected_model_type = st.session_state.get(
                    'selected_model_type', "Standard Skin Analysis")

                if selected_model_type == "Standard Skin Analysis":
                    # Use standard models (Random Forest, Gradient Boosting, SVM)
                    standard_models = [
                        name for name in ensemble_preds.keys() if name not in [
                            "most_reliable_model", "Severe Conditions",
                            "Child Skin"
                        ]
                    ]

                    if standard_models:
                        # Find model with highest confidence
                        highest_confidence_model = None
                        highest_confidence = 0

                        for model_name in standard_models:
                            if model_name in ensemble_preds:
                                confidence = float(ensemble_preds[model_name]['confidence'].rstrip('%'))
                                if confidence > highest_confidence:
                                    highest_confidence = confidence
                                    highest_confidence_model = model_name

                        if highest_confidence_model:
                            model_result = ensemble_preds[highest_confidence_model]
                            st.markdown(f"### Standard Skin Analysis")
                            st.write(f"**{t('condition')}** {model_result['condition']}")
                            st.write(f"**{t('confidence')}** {model_result['confidence']}")

                            if 'key_factors' in model_result:
                                st.write(f"**{t('key_factors')}**")
                                for factor, importance in model_result['key_factors']:
                                    st.write(f"- {factor}: {importance:.2%}")

                elif selected_model_type == "Severe Skin Conditions":
                    # Display severe conditions model results
                    if "Severe Conditions" in ensemble_preds:
                        st.markdown("### Severe Conditions Analysis")
                        model_result = ensemble_preds["Severe Conditions"]
                        st.write(f"**{t('condition')}** {model_result['condition']}")
                        st.write(f"**{t('confidence')}** {model_result['confidence']}")
                        if 'key_factors' in model_result:
                            st.write(f"**{t('key_factors')}**")
                            for factor, importance in model_result['key_factors']:
                                st.write(f"- {factor}: {importance:.2%}")
                    else:
                        st.warning("Severe conditions model not available")

                else:  # Child Skin Analysis
                    # Display child skin model results
                    if "Child Skin" in ensemble_preds:
                        st.markdown("### Child Skin Analysis")
                        model_result = ensemble_preds["Child Skin"]
                        st.write(f"**{t('condition')}** {model_result['condition']}")
                        st.write(f"**{t('confidence')}** {model_result['confidence']}")
                        if 'key_factors' in model_result:
                            st.write(f"**{t('key_factors')}**")
                            for factor, importance in model_result['key_factors']:
                                st.write(f"- {factor}: {importance:.2%}")
                    else:
                        st.warning("Child skin model not available")
            else:
                st.write(f"**{t('condition')}** {ml_prediction['condition']}")
                st.write(
                    f"**{t('confidence')}** {ml_prediction['confidence']}")

                st.write(f"**{t('key_factors')}**")
                if 'key_factors' in ml_prediction:
                    for factor, importance in ml_prediction['key_factors']:
                        st.write(f"- {factor}: {importance:.2%}")

                with st.expander(t("detailed_probabilities")):
                    for condition, prob in ml_prediction[
                        'probabilities'].items():
                        st.write(f"- {condition}: {prob}")

        # with col2:
        #     st.subheader("Additional Analysis")
        #
        #     # Original ML Analysis
        #     # if ai_analysis and ai_analysis.get('ml_analysis'):
        #     #     st.markdown(ai_analysis['ml_analysis'])
        #     # else:
        #     #     st.info("ML analysis unavailable")
        #
        #     # New Neural Networks Analysis
        #     with st.expander("Нейросетевой анализ проблем кожи", expanded=True):
        #         problems_prediction = predict_skin_problems(skin_features)
        #         st.write(f"**Тип анализа:** {problems_prediction['analysis_type']}")
        #         st.write(f"**Состояние:** {problems_prediction['condition']}")
        #         st.write(f"**Уверенность:** {problems_prediction['confidence']}")
        #         st.write("**Детальный анализ проблем:**")
        #         for problem in problems_prediction['detailed_problems']:
        #             st.write(f"- {problem}")

        # with col3:
        #     st.subheader("Recommendation Analysis")
        #
        #     with st.expander("Расширенные рекомендации по уходу", expanded=True):
        #         care_prediction = predict_skin_care(skin_features)
        #         st.write(f"**Тип анализа:** {care_prediction['analysis_type']}")
        #         st.write(f"**Тип ухода:** {care_prediction['care_type']}")
        #         st.write(f"**Уверенность:** {care_prediction['confidence']}")
        #         st.write("**Рекомендации по уходу:**")
        #         for rec in care_prediction['recommendations']:
        #             st.write(f"- {rec}")

        # Handle publish button functionality
        publish_button = st.button("Publish")
        if publish_button:
            print(
                f"st.session_state.get('authenticated', False):{st.session_state.get('authenticated', False)}"
            )
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
                    st.session_state.user_id, analysis_data)

                # Show success message with ID
                st.success(
                    f"Analysis published successfully! Share ID: {analysis_id}"
                )
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
            "Tone Uniformity": {
                t("uneven"): (0, 65),
                t("moderate"): (65, 85),
                t("even"): (85, 100)
            },
            "Brightness": {
                t("dull"): (0, 50),
                t("moderate"): (50, 75),
                t("bright"): (75, 100)
            },
            "Texture": {
                t("smooth"): (0, 30),
                t("moderate"): (30, 60),
                t("rough"): (60, 100)
            },
            "Spots Detected": {
                t("few"): (0, 3),
                t("moderate"): (3, 7),
                t("many"): (7, 100)
            },
            "Redness": {
                t("low"): (0, 30),
                t("moderate"): (30, 60),
                t("high"): (60, 100)
            },
            "Pigmentation": {
                t("even"): (0, 25),
                t("moderate"): (25, 50),
                t("uneven"): (50, 100)
            }
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
            numeric_value = float(value) if isinstance(value,
                                                       (float, int)) else 0
            interpretation = get_interpretation(feature, numeric_value)

            with st.expander(
                    f"{feature}: {numeric_value:.1f} ({interpretation})",
                    expanded=False):
                # Description
                st.markdown(
                    f"**{t('description')}** {metric_descriptions.get(feature, 'No description available')}"
                )

                # Reference range visualization
                if feature in reference_ranges:
                    st.markdown(f"**{t('reference_range')}**")
                    col1, col2, col3 = st.columns(3)

                    ranges = reference_ranges[feature]
                    categories = list(ranges.keys())

                    with col1:
                        st.markdown(f"**{categories[0]}**")
                        st.markdown(
                            f"{ranges[categories[0]][0]} - {ranges[categories[0]][1]}"
                        )
                    with col2:
                        st.markdown(f"**{categories[1]}**")
                        st.markdown(
                            f"{ranges[categories[1]][0]} - {ranges[categories[1]][1]}"
                        )
                    with col3:
                        st.markdown(f"**{categories[2]}**")
                        st.markdown(
                            f"{ranges[categories[2]][0]} - {ranges[categories[2]][1]}"
                        )

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

        # Model Comparison Section
        st.header(t("preliminary_result"))
        if 'ensemble_predictions' in st.session_state:
            ensemble_preds = st.session_state.ensemble_predictions
            selected_model_type = st.session_state.get('selected_model_type', "Standard Skin Analysis")

            # Define which models to show based on selected type
            standard_models = ["Random Forest", "Gradient Boosting", "SVM"]
            if selected_model_type == "Standard Skin Analysis":
                models_to_show = standard_models
            elif selected_model_type == "Severe Skin Conditions":
                models_to_show = standard_models + ["Severe Conditions"]
            else:  # Child Skin Analysis
                models_to_show = standard_models + ["Child Skin"]

            # Create a comparison table
            import pandas as pd

            comparison_data = []
            for model_name in models_to_show:
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
                if most_reliable in models_to_show:
                    st.success(t("most_reliable").format(model=most_reliable))

        # Product Recommendations section
        st.header(t("recommended_products"))
        import os
        from PIL import Image

        try:
            # Get all products from database
            all_products = get_database_products()

            # Limit to 6 products maximum
            max_products = 6
            products_to_display = all_products[:max_products]

            # Show message if there are more products than displayed
            if len(all_products) > max_products:
                st.info(
                    f"Showing {max_products} of {len(all_products)} total products."
                )

            if not products_to_display:
                st.warning("No products found in the database.")

            # Create rows of two columns for products
            # Create rows with 2 products each
            for i in range(0, len(products_to_display), 2):
                cols = st.columns(2)

                # Add products to columns
                for j in range(2):
                    if i + j < len(products_to_display):
                        product = products_to_display[i + j]
                        with cols[j]:
                            with st.expander(
                                    f"🏥 {product['name']} - ${product['price']}",
                                    expanded=False):
                                # Create two columns for product info and image
                                col_info, col_img = st.columns([3, 1])

                                with col_info:
                                    # Basic Information
                                    st.write(
                                        f"**{t('description')}** {product['description']}"
                                    )

                                    # Ingredients Section
                                    st.write(
                                        f"**🧪 {t('ingredients')}:** {', '.join(product['ingredients'])}"
                                    )

                                    # Key Benefits
                                    st.write(f"**✨ {t('key_benefits')}:**")
                                    for benefit in product['key_benefits']:
                                        st.write(f"- {benefit}")

                                    # Price and product type
                                    st.write(
                                        f"**💰 Price:** ${product['price']}")
                                    if 'product_type' in product and product[
                                        'product_type']:
                                        st.write(
                                            f"**📦 Type:** {product['product_type']}"
                                        )

                                    # Other database fields
                                    if 'ph_level' in product and product[
                                        'ph_level']:
                                        st.write(
                                            f"**🧪 pH Level:** {product['ph_level']}"
                                        )
                                    if 'comedogenic_rating' in product and product[
                                        'comedogenic_rating']:
                                        st.write(
                                            f"**⚖️ Comedogenic Rating:** {product['comedogenic_rating']}/5"
                                        )
                                    if 'fragrance' in product and product[
                                        'fragrance']:
                                        st.write(
                                            f"**🌸 Fragrance:** {product['fragrance']}"
                                        )

                                # Display product image in the right column
                                with col_img:
                                    # Check if image path exists in product data
                                    if 'image_path' in product and product[
                                        'image_path'] and os.path.exists(
                                        product['image_path']):
                                        try:
                                            img = Image.open(
                                                product['image_path'])
                                            st.image(img,
                                                     caption=product['name'],
                                                     use_container_width=True)
                                        except Exception:
                                            # Fallback if image cannot be loaded
                                            st.image(
                                                "https://via.placeholder.com/200x300?text=Product",
                                                caption=product['name'],
                                                use_container_width=True)
                                    else:
                                        # Display placeholder if no image available
                                        st.image(
                                            "https://via.placeholder.com/200x300?text=Product",
                                            caption=product['name'],
                                            use_container_width=True)

                                    # Usage Instructions
                                    st.write(f"**📝 {t('how_to_use')}:**")
                                    if isinstance(
                                            product['usage_instructions'],
                                            dict):
                                        # If it's a dictionary, access the keys
                                        st.write(
                                            f"**{t('frequency')}:** {product['usage_instructions'].get('frequency', 'Not specified')}"
                                        )
                                        if product['usage_instructions'].get(
                                                'steps'):
                                            st.write(f"**{t('steps')}:**")
                                            for step in product[
                                                'usage_instructions'].get(
                                                'steps', []):
                                                st.write(f"- {step}")
                                        if product['usage_instructions'].get(
                                                'warnings'):
                                            st.warning(
                                                f"⚠️ **{t('warning')}** {product['usage_instructions'].get('warnings', '')}"
                                            )
                                    else:
                                        # If it's a string, display it directly
                                        st.write(product['usage_instructions'])

                                    # Skin Compatibility
                                    suitable_for = ", ".join(
                                        product.get(
                                            'suitable_for',
                                            product.get(
                                                'skin_compatibility', [])))
                                    if suitable_for:
                                        st.write(
                                            f"**👥 {t('suitable_for')}:** {suitable_for}"
                                        )

                                    # Display warnings if available
                                    warnings = product.get('warnings')
                                    if warnings:
                                        st.warning(
                                            f"⚠️ **{t('warnings')}:** {warnings}"
                                        )

        except Exception as e:
            st.error(f"Error loading products: {str(e)}")
            import traceback
            st.error(traceback.format_exc())

        # Doctor Directory section
        st.header(t("consult_professional"))

        # Get user location from session state
        user_location = st.session_state.user_location
        user_lat = user_location.get('latitude')
        user_lon = user_location.get('longitude')
        search_radius = st.session_state.search_radius

        # Get all doctors and filter by radius
        all_doctors = get_all_doctors()
        doctors_in_radius = get_doctors_in_radius(user_lat, user_lon,
                                                  search_radius, all_doctors)

        # Display radius information
        st.info(t("doctorsin_radius").format(radius=search_radius))

        if not doctors_in_radius:
            st.warning(
                f"No doctors found within {search_radius}km. Try increasing your search radius in settings."
            )

        # Display doctors within radius
        for doctor in doctors_in_radius:
            with st.expander(
                    f"👨‍⚕️ Dr. {doctor['name']} - {doctor['speciality']} ({doctor['distance']} km)",
                    expanded=False):

                col1, col2 = st.columns([2, 1])
                with col1:
                    st.write(
                        f"**{t('experience')}** {doctor['experience']} years")
                    st.write(f"**{t('location')}** {doctor['location']}")
                    st.write(f"**{t('contact')}** {doctor['contact']}")
                    st.write(f"**{t('email')}** {doctor['email']}")
                    st.write(
                        f"**{t('available_hours')}** {doctor['available_hours']}"
                    )
                    st.write(f"**{t('distance')}** {doctor['distance']} km")

                with col2:
                    # Add button to open chat with this doctor
                    if st.button(t("send_message"),
                                 key=f"msg_btn_{doctor['name']}"):
                        # Start chat with this doctor
                        start_chat_with_doctor(doctor)
                        st.rerun()


if __name__ == "__main__":
    main()