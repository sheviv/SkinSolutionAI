import streamlit as st

# Available languages
LANGUAGES = {
    "English": "en",
    "Русский": "ru"  # Russian
}

# Translations dictionary
TRANSLATIONS = {
    "en": {
        # App title and headers
        "app_title": "SkinHealth AI",
        "app_subtitle": "Professional Skin Analysis & Recommendations",
        "clinic_image_1": "Professional Dermatology Care",
        "clinic_image_2": "Expert Consultation",

        # Upload section
        "upload_header": "Upload Your Skin Photo",
        "upload_info": "For best results, please ensure good lighting and a clear view of the skin area.",
        "upload_button": "Choose an image...",

        # Analysis results
        "analysis_results": "Analysis Results",
        "original_image": "Original Image",
        "original_upload": "Original Upload",
        "analysis_detected": "Analysis with Detected Areas",
        "problem_areas": "Detected Problem Areas:",
        "select_area": "Select an area to view details:",
        "area_type": "Type:",
        "area_severity": "Severity:",
        "area_size": "Size:",
        "area_description": "Description:",

        # ML model analysis
        "ml_model_analysis": "ML Model Analysis",
        "primary_model": "Primary Model:",
        "used_for_recommendations": "(used for recommendations)",
        "view_other_models": "View other model predictions:",
        "condition": "Condition:",
        "confidence": "Confidence:",
        "key_factors": "Key Factors:",
        "detailed_probabilities": "View Detailed Probabilities",

        # AI analysis
        "anthropic_analysis": "Anthropic Analysis",
        "anthropic_unavailable": "Anthropic analysis unavailable",
        "openai_analysis": "OpenAI Analysis",
        "openai_unavailable": "OpenAI analysis unavailable",

        # Skin metrics
        "skin_metrics": "Detailed Skin Metrics:",
        "description": "Description:",
        "reference_range": "Reference Range:",
        "recommendations": "Recommendations:",

        # Reference range categories
        "uneven": "Uneven",
        "moderate": "Moderate",
        "even": "Even",
        "dull": "Dull",
        "bright": "Bright",
        "smooth": "Smooth",
        "rough": "Rough",
        "few": "Few",
        "many": "Many",
        "low": "Low",
        "high": "High",

        # Metric descriptions
        "tone_uniformity_desc": "Measures how even your skin color is across the analyzed area. Higher values indicate more even tone.",
        "brightness_desc": "Indicates the overall luminosity of your skin. Higher values suggest a brighter complexion.",
        "texture_desc": "Evaluates the smoothness of your skin surface. Lower values indicate smoother skin.",
        "spots_desc": "Number of dark spots or hyperpigmentation areas detected in the image.",
        "redness_desc": "Measures inflammation or irritation. Lower values indicate less redness.",
        "pigmentation_desc": "Assesses the variation in melanin distribution. Higher values indicate more uneven pigmentation.",

        # Recommendations
        "rec_niacinamide": "Consider products with niacinamide or vitamin C",
        "rec_sunscreen": "Use sunscreen daily to prevent further uneven pigmentation",
        "rec_exfoliants": "Try chemical exfoliants like AHAs",
        "rec_vitamin_c": "Add vitamin C serum to your routine",
        "rec_gentle_exfoliation": "Use gentle exfoliation 2-3 times weekly",
        "rec_retinol": "Consider retinol products to improve cell turnover",
        "rec_targeted_treatments": "Targeted treatments with tranexamic acid or alpha arbutin",
        "rec_sun_protection": "Protect from sun exposure",
        "rec_centella": "Look for products with centella asiatica or green tea",
        "rec_avoid_hot": "Avoid hot water and harsh cleansers",
        "rec_licorice": "Consider products with licorice extract or kojic acid",
        "rec_consistent_sunscreen": "Ensure consistent sunscreen use",
        "rec_continue": "Continue with your current skincare routine",
        "rec_prevention": "Focus on prevention and maintenance",

        # Problem area analysis
        "comprehensive_analysis": "Comprehensive Problem Area Analysis",
        "no_problem_areas": "No specific problem areas were detected in the image.",
        "details": "Details",
        "recommended_actions": "Recommended Actions:",

        # ML model comparison
        "model_comparison": "Machine Learning Model Comparison",
        "most_reliable": "The {model} model has been selected as the most reliable based on confidence scores and is being used for final recommendations.",

        # Products and doctors
        "recommended_products": "Recommended Products",
        "ingredients": "Ingredients",
        "key_benefits": "Key Benefits",
        "how_to_use": "How to Use",
        "frequency": "Frequency:",
        "steps": "Steps:",
        "warning": "Warning:",
        "suitable_for": "Suitable For",
        "ingredient_analysis": "Ingredient Analysis",
        "active_ingredients": "Active Ingredients:",
        "potential_irritants": "Potential Irritants:",
        "additional_info": "Additional Information:",
        "ph_level": "pH Level:",
        "fragrance_free": "Fragrance Free:",
        "comedogenic": "Comedogenic Rating:",
        "comedogenic_scale": "(0 = non-comedogenic, 5 = highly comedogenic)",
        "certifications": "Certifications:",

        # Location and doctor settings
        "location_settings": "Location Settings",
        "your_location": "Your Location",
        "detect_location": "Detect My Location",
        "enter_location": "Enter Location Manually",
        "search_radius": "Search Radius (km)",
        "doctors_in_radius": "Doctors within {radius}km radius",
        "distance": "Distance:",
        "km_away": "{distance} km away",
        "send_message": "Send Message",
        "message_sent": "Message sent successfully!",
        "available_hours": "Available Hours:",
        "email": "Email:",
        "message_doctor": "Message Doctor",
        "message_placeholder": "Type your message here...",
        "send": "Send",
        "cancel": "Cancel",

        "consult_professional": "Consult a Professional",
        "experience": "Experience:",
        "location": "Location:",
        "contact": "Contact:"
    },
    "ru": {
        # App title and headers
        "app_title": "SkinHealth AI",
        "app_subtitle": "Профессиональный анализ кожи и рекомендации",
        "clinic_image_1": "Профессиональный дерматологический уход",
        "clinic_image_2": "Консультация специалиста",

        # Upload section
        "upload_header": "Загрузите фотографию вашей кожи",
        "upload_info": "Для достижения наилучших результатов, пожалуйста, убедитесь в хорошем освещении и четком изображении участка кожи.",
        "upload_button": "Выберите изображение...",

        # Analysis results
        "analysis_results": "Результаты анализа",
        "original_image": "Исходное изображение",
        "original_upload": "Исходная загрузка",
        "analysis_detected": "Анализ с обнаруженными областями",
        "problem_areas": "Обнаруженные проблемные области:",
        "select_area": "Выберите область для просмотра деталей:",
        "area_type": "Тип:",
        "area_severity": "Степень тяжести:",
        "area_size": "Размер:",
        "area_description": "Описание:",

        # ML model analysis
        "ml_model_analysis": "Анализ ML модели",
        "primary_model": "Основная модель:",
        "used_for_recommendations": "(используется для рекомендаций)",
        "view_other_models": "Посмотреть прогнозы других моделей:",
        "condition": "Состояние:",
        "confidence": "Уверенность:",
        "key_factors": "Ключевые факторы:",
        "detailed_probabilities": "Просмотр подробных вероятностей",

        # AI analysis
        "anthropic_analysis": "Анализ Anthropic",
        "anthropic_unavailable": "Анализ Anthropic недоступен",
        "openai_analysis": "Анализ OpenAI",
        "openai_unavailable": "Анализ OpenAI недоступен",

        # Skin metrics
        "skin_metrics": "Подробные показатели кожи:",
        "description": "Описание:",
        "reference_range": "Референсный диапазон:",
        "recommendations": "Рекомендации:",

        # Reference range categories
        "uneven": "Неровный",
        "moderate": "Средний",
        "even": "Ровный",
        "dull": "Тусклый",
        "bright": "Яркий",
        "smooth": "Гладкий",
        "rough": "Шероховатый",
        "few": "Мало",
        "many": "Много",
        "low": "Низкий",
        "high": "Высокий",

        # Metric descriptions
        "tone_uniformity_desc": "Измеряет насколько равномерен цвет вашей кожи по всей анализируемой области. Более высокие значения указывают на более равномерный тон.",
        "brightness_desc": "Указывает на общую яркость вашей кожи. Более высокие значения предполагают более яркий цвет лица.",
        "texture_desc": "Оценивает гладкость поверхности вашей кожи. Более низкие значения указывают на более гладкую кожу.",
        "spots_desc": "Количество темных пятен или областей гиперпигментации, обнаруженных на изображении.",
        "redness_desc": "Измеряет воспаление или раздражение. Более низкие значения указывают на меньшее покраснение.",
        "pigmentation_desc": "Оценивает вариацию в распределении меланина. Более высокие значения указывают на более неравномерную пигментацию.",

        # Recommendations
        "rec_niacinamide": "Рассмотрите продукты с ниацинамидом или витамином C",
        "rec_sunscreen": "Используйте солнцезащитный крем ежедневно для предотвращения дальнейшей неравномерной пигментации",
        "rec_exfoliants": "Попробуйте химические эксфолианты, такие как AHA",
        "rec_vitamin_c": "Добавьте в свой уход сыворотку с витамином C",
        "rec_gentle_exfoliation": "Используйте мягкое отшелушивание 2-3 раза в неделю",
        "rec_retinol": "Рассмотрите продукты с ретинолом для улучшения обновления клеток",
        "rec_targeted_treatments": "Целевое лечение с транексамовой кислотой или альфа-арбутином",
        "rec_sun_protection": "Защитите от воздействия солнца",
        "rec_centella": "Ищите продукты с центеллой азиатской или зеленым чаем",
        "rec_avoid_hot": "Избегайте горячей воды и агрессивных очищающих средств",
        "rec_licorice": "Рассмотрите продукты с экстрактом солодки или койевой кислотой",
        "rec_consistent_sunscreen": "Обеспечьте постоянное использование солнцезащитного крема",
        "rec_continue": "Продолжайте свой текущий уход за кожей",
        "rec_prevention": "Сосредоточьтесь на профилактике и поддержании",

        # Problem area analysis
        "comprehensive_analysis": "Комплексный анализ проблемных областей",
        "no_problem_areas": "На изображении не обнаружено конкретных проблемных областей.",
        "details": "Детали",
        "recommended_actions": "Рекомендуемые действия:",

        # ML model comparison
        "model_comparison": "Сравнение моделей машинного обучения",
        "most_reliable": "Модель {model} была выбрана как наиболее надежная на основе показателей уверенности и используется для окончательных рекомендаций.",

        # Products and doctors
        "recommended_products": "Рекомендуемые продукты",
        "ingredients": "Ингредиенты",
        "key_benefits": "Ключевые преимущества",
        "how_to_use": "Как использовать",
        "frequency": "Частота:",
        "steps": "Шаги:",
        "warning": "Предупреждение:",
        "suitable_for": "Подходит для",
        "ingredient_analysis": "Анализ ингредиентов",
        "active_ingredients": "Активные ингредиенты:",
        "potential_irritants": "Потенциальные раздражители:",
        "additional_info": "Дополнительная информация:",
        "ph_level": "Уровень pH:",
        "fragrance_free": "Без отдушек:",
        "comedogenic": "Комедогенный рейтинг:",
        "comedogenic_scale": "(0 = некомедогенный, 5 = высококомедогенный)",
        "certifications": "Сертификаты:",

        "consult_professional": "Проконсультируйтесь с профессионалом",
        "experience": "Опыт:",
        "location": "Местоположение:",
        "contact": "Контакт:",
        "chat": "Чат",
        "back_to_analysis": "Вернуться к анализу кожи",
        "chat_with": "Чат с доктором {doctor}",
        "type_message": "Введите сообщение...",
        "send_message_btn": "Отправить",
        "exit_chat": "Выйти из чата",
        "messages": "Сообщения"
    }
}


def get_translation(key, lang=None):
    """Get translation for a key in the specified language"""
    if lang is None:
        lang = st.session_state.get('language', 'ru')

    # Return the translation or the key itself if not found
    return TRANSLATIONS.get(lang, {}).get(key, key)


def init_language():
    """Initialize language in session state if not already set"""
    if 'language' not in st.session_state:
        st.session_state.language = 'ru'


def change_language(lang):
    """Change the language in session state"""
    if lang in [code for _, code in LANGUAGES.items()]:
        st.session_state.language = lang
        # Force refresh
        st.rerun()