import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Create directory for models if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

# Initialize scalers for different specialized models
severe_scaler = StandardScaler()
child_scaler = StandardScaler()

# Model for severe skin conditions
severe_model = GradientBoostingClassifier(
    n_estimators=600,
    learning_rate=0.05,
    max_depth=12,
    min_samples_split=3,
    min_samples_leaf=2,
    subsample=0.8,
    random_state=42
)

# Neural network model for children's skin conditions
child_model = MLPClassifier(
    hidden_layer_sizes=(100, 75, 50),
    activation='relu',
    solver='adam',
    alpha=0.0001,
    batch_size=64,
    learning_rate='adaptive',
    max_iter=1000,
    random_state=42
)


def generate_severe_condition_data():
    """Generate training data for severe skin conditions."""
    np.random.seed(42)
    n_samples = 1800

    # Generate feature data for severe conditions
    tone_uniformity = np.random.normal(60, 18, n_samples)
    brightness = np.random.normal(55, 17, n_samples)
    texture = np.random.normal(60, 15, n_samples)
    spots = np.random.poisson(7, n_samples)
    redness = np.random.normal(55, 18, n_samples)
    pigmentation = np.random.normal(50, 15, n_samples)

    X = np.column_stack([
        tone_uniformity, brightness, texture,
        spots, redness, pigmentation
    ])

    # Generate labels for severe conditions
    y = []
    for i in range(n_samples):
        # Severe acne
        if spots[i] > 10 and redness[i] > 60:
            y.append("Severe Acne")
        # Rosacea
        elif redness[i] > 70 and tone_uniformity[i] < 60:
            y.append("Rosacea")
        # Psoriasis
        elif texture[i] > 75 and brightness[i] < 50:
            y.append("Psoriasis")
        # Eczema
        elif texture[i] > 65 and redness[i] > 50 and brightness[i] < 60:
            y.append("Eczema")
        # Melasma
        elif pigmentation[i] > 65 and tone_uniformity[i] < 55:
            y.append("Melasma")
        # Dermatitis
        else:
            y.append("Dermatitis")

    return X, np.array(y)


def generate_child_skin_data():
    """Generate training data for children's skin conditions."""
    np.random.seed(43)  # Different seed
    n_samples = 1500

    # Generate feature data for children's skin - different distributions
    tone_uniformity = np.random.normal(85, 10, n_samples)  # Children usually have more even tone
    brightness = np.random.normal(80, 12, n_samples)  # Brighter skin typically
    texture = np.random.normal(30, 12, n_samples)  # Smoother texture
    spots = np.random.poisson(2, n_samples)  # Fewer spots generally
    redness = np.random.normal(35, 15, n_samples)  # Can be sensitive
    pigmentation = np.random.normal(25, 8, n_samples)  # Less pigmentation issues

    X = np.column_stack([
        tone_uniformity, brightness, texture,
        spots, redness, pigmentation
    ])

    # Generate labels for children's skin conditions
    y = []
    for i in range(n_samples):
        # Atopic Dermatitis (common in children)
        if redness[i] > 55 and texture[i] > 45:
            y.append("Atopic Dermatitis")
        # Diaper Rash patterns
        elif redness[i] > 50 and brightness[i] < 70:
            y.append("Diaper Rash Related")
        # Infantile Acne
        elif spots[i] > 4 and redness[i] > 40:
            y.append("Infantile Acne")
        # Cradle Cap
        elif texture[i] > 50 and tone_uniformity[i] < 75:
            y.append("Cradle Cap")
        # Contact Dermatitis
        elif redness[i] > 45 and pigmentation[i] > 35:
            y.append("Contact Dermatitis")
        # Healthy Child Skin
        else:
            y.append("Healthy Child Skin")

    return X, np.array(y)


def train_severe_model():
    """Train the model for severe skin conditions."""
    X, y = generate_severe_condition_data()

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    X_train_scaled = severe_scaler.fit_transform(X_train)
    X_test_scaled = severe_scaler.transform(X_test)

    # Train model
    severe_model.fit(X_train_scaled, y_train)

    # Evaluate model
    from sklearn.metrics import accuracy_score, classification_report
    y_pred = severe_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Severe Conditions Model Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(report)

    # Save model and scaler
    joblib.dump(severe_model, 'models/severe_skin_model.joblib')
    joblib.dump(severe_scaler, 'models/severe_scaler.joblib')

    return severe_model, severe_scaler


def train_child_model():
    """Train the model for children's skin conditions."""
    X, y = generate_child_skin_data()

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    X_train_scaled = child_scaler.fit_transform(X_train)
    X_test_scaled = child_scaler.transform(X_test)

    # Train model
    child_model.fit(X_train_scaled, y_train)

    # Evaluate model
    from sklearn.metrics import accuracy_score, classification_report
    y_pred = child_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Children's Skin Model Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(report)

    # Save model and scaler
    joblib.dump(child_model, 'models/child_skin_model.joblib')
    joblib.dump(child_scaler, 'models/child_scaler.joblib')

    return child_model, child_scaler


def load_specialized_models():
    """Load all specialized models if they exist, otherwise train them."""
    severe_classifier = None
    child_classifier = None

    try:
        # Try to load severe condition model
        if os.path.exists('models/severe_skin_model.joblib'):
            severe_classifier = joblib.load('models/severe_skin_model.joblib')
            severe_scaler = joblib.load('models/severe_scaler.joblib')
        else:
            print("Training severe skin condition model...")
            severe_classifier, severe_scaler = train_severe_model()

        # Try to load children's skin model
        if os.path.exists('models/child_skin_model.joblib'):
            child_classifier = joblib.load('models/child_skin_model.joblib')
            child_scaler = joblib.load('models/child_scaler.joblib')
        else:
            print("Training children's skin model...")
            child_classifier, child_scaler = train_child_model()

    except Exception as e:
        print(f"Error loading specialized models: {str(e)}")
        # If loading fails, train models
        if not severe_classifier:
            severe_classifier, severe_scaler = train_severe_model()
        if not child_classifier:
            child_classifier, child_scaler = train_child_model()

    return {
        'severe_model': severe_classifier,
        'severe_scaler': severe_scaler,
        'child_model': child_classifier,
        'child_scaler': child_scaler
    }


def predict_severe_condition(features):
    """Predict severe skin conditions."""
    try:
        # Make sure models are loaded
        if not os.path.exists('models/severe_skin_model.joblib'):
            train_severe_model()

        # Load model if not already loaded
        model = joblib.load('models/severe_skin_model.joblib')
        scaler = joblib.load('models/severe_scaler.joblib')

        # Convert features to numerical values
        feature_values = np.array([
            float(features["Tone Uniformity"]),
            float(features["Brightness"]),
            float(features["Texture"]),
            float(features["Spots Detected"]),
            float(features["Redness"]),
            float(features["Pigmentation"])
        ]).reshape(1, -1)

        # Scale features
        scaled_features = scaler.transform(feature_values)

        # Make prediction
        probabilities = model.predict_proba(scaled_features)[0]
        confidence = max(probabilities) * 100
        prediction = model.predict(scaled_features)[0]

        # Get feature importance for GradientBoostingClassifier
        feature_importance = dict(zip(
            ["Tone", "Brightness", "Texture", "Spots", "Redness", "Pigmentation"],
            model.feature_importances_
        ))

        return {
            'condition': prediction,
            'confidence': f"{confidence:.1f}%",
            'probabilities': {
                class_name: f"{prob * 100:.1f}%"
                for class_name, prob in zip(model.classes_, probabilities)
            },
            'key_factors': sorted(
                feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:3],
            'model_type': 'severe_conditions'
        }
    except Exception as e:
        print(f"Severe condition prediction error: {str(e)}")
        return {
            'condition': "Unknown Severe Condition",
            'confidence': "N/A",
            'probabilities': {},
            'key_factors': [],
            'model_type': 'severe_conditions'
        }


def predict_child_skin_condition(features):
    """Predict children's skin conditions."""
    try:
        # Make sure models are loaded
        if not os.path.exists('models/child_skin_model.joblib'):
            train_child_model()

        # Load model if not already loaded
        model = joblib.load('models/child_skin_model.joblib')
        scaler = joblib.load('models/child_scaler.joblib')

        # Convert features to numerical values
        feature_values = np.array([
            float(features["Tone Uniformity"]),
            float(features["Brightness"]),
            float(features["Texture"]),
            float(features["Spots Detected"]),
            float(features["Redness"]),
            float(features["Pigmentation"])
        ]).reshape(1, -1)

        # Scale features
        scaled_features = scaler.transform(feature_values)

        # Make prediction
        probabilities = model.predict_proba(scaled_features)[0]
        confidence = max(probabilities) * 100
        prediction = model.predict(scaled_features)[0]

        # MLPClassifier doesn't have feature_importances_
        # Use coefficients of features for the first layer as a proxy
        coefs = model.coefs_[0]
        importances = np.sum(np.abs(coefs), axis=1)
        importances = importances / np.sum(importances)  # Normalize

        feature_importance = dict(zip(
            ["Tone", "Brightness", "Texture", "Spots", "Redness", "Pigmentation"],
            importances
        ))

        return {
            'condition': prediction,
            'confidence': f"{confidence:.1f}%",
            'probabilities': {
                class_name: f"{prob * 100:.1f}%"
                for class_name, prob in zip(model.classes_, probabilities)
            },
            'key_factors': sorted(
                feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:3],
            'model_type': 'child_skin'
        }
    except Exception as e:
        print(f"Child skin prediction error: {str(e)}")
        return {
            'condition': "Unknown Child Skin Condition",
            'confidence': "N/A",
            'probabilities': {},
            'key_factors': [],
            'model_type': 'child_skin'
        }


def generate_skin_problems_data():
    """Generate training data for skin problems analysis."""
    np.random.seed(44)
    n_samples = 2000

    features = np.random.rand(n_samples, 6) * 100  # Generate random features

    # Generate labels based on feature combinations
    labels = []
    for features_row in features:
        if features_row[3] > 70:  # High number of spots
            labels.append("Acne-Related Issues")
        elif features_row[4] > 65:  # High redness
            labels.append("Sensitive Skin Issues")
        elif features_row[0] < 40:  # Low tone uniformity
            labels.append("Hyperpigmentation")
        elif features_row[2] > 60:  # High texture
            labels.append("Texture Issues")
        else:
            labels.append("Normal Skin")

    return features, np.array(labels)


def generate_skin_care_data():
    """Generate training data for skin care recommendations."""
    np.random.seed(45)
    n_samples = 2000

    features = np.random.rand(n_samples, 6) * 100

    # Generate care type labels based on features
    labels = []
    for features_row in features:
        if features_row[1] < 50:  # Low brightness
            labels.append("Intensive Care")
        elif features_row[4] > 60:  # High redness
            labels.append("Calming Care")
        elif features_row[3] > 50:  # Moderate to high spots
            labels.append("Treatment Care")
        else:
            labels.append("Maintenance Care")

    return features, np.array(labels)


# Neural network for detailed skin problems analysis
skin_problems_model = MLPClassifier(
    hidden_layer_sizes=(150, 100, 50),
    activation='relu',
    solver='adam',
    alpha=0.001,
    batch_size=32,
    learning_rate='adaptive',
    max_iter=1000,
    random_state=42
)

# Neural network for extended skin care recommendations
skin_care_model = MLPClassifier(
    hidden_layer_sizes=(200, 150, 100),
    activation='relu',
    solver='adam',
    alpha=0.0005,
    batch_size=32,
    learning_rate='adaptive',
    max_iter=1000,
    random_state=42
)

# Train models on initialization
X_problems, y_problems = generate_skin_problems_data()
skin_problems_model.fit(X_problems, y_problems)

X_care, y_care = generate_skin_care_data()
skin_care_model.fit(X_care, y_care)


def predict_skin_problems(features):
    """Predict detailed skin problems using neural network."""
    try:
        # Convert features to numerical values
        feature_values = np.array([
            float(features["Tone Uniformity"]),
            float(features["Brightness"]),
            float(features["Texture"]),
            float(features["Spots Detected"]),
            float(features["Redness"]),
            float(features["Pigmentation"])
        ]).reshape(1, -1)

        # Make prediction
        probabilities = skin_problems_model.predict_proba(feature_values)[0]
        confidence = max(probabilities) * 100
        prediction = skin_problems_model.predict(feature_values)[0]

        return {
            'condition': prediction,
            'confidence': f"{confidence:.1f}%",
            'analysis_type': 'Deep Neural Analysis',
            'detailed_problems': [
                'Dehydration Level',
                'Barrier Damage',
                'Sensitivity Level',
                'Environmental Damage'
            ]
        }
    except Exception as e:
        print(f"Skin problems prediction error: {str(e)}")
        return {
            'condition': "Analysis Unavailable",
            'confidence': "N/A",
            'analysis_type': 'Deep Neural Analysis',
            'detailed_problems': []
        }


def predict_skin_care(features):
    """Predict extended skin care recommendations."""
    try:
        # Convert features to numerical values
        feature_values = np.array([
            float(features["Tone Uniformity"]),
            float(features["Brightness"]),
            float(features["Texture"]),
            float(features["Spots Detected"]),
            float(features["Redness"]),
            float(features["Pigmentation"])
        ]).reshape(1, -1)

        # Make prediction
        probabilities = skin_care_model.predict_proba(feature_values)[0]
        confidence = max(probabilities) * 100
        prediction = skin_care_model.predict(feature_values)[0]

        return {
            'care_type': prediction,
            'confidence': f"{confidence:.1f}%",
            'analysis_type': 'Advanced Care Analysis',
            'recommendations': [
                'Active Ingredients',
                'Treatment Frequency',
                'Environmental Protection',
                'Lifestyle Adjustments'
            ]
        }
    except Exception as e:
        print(f"Skin care prediction error: {str(e)}")
        return {
            'care_type': "Analysis Unavailable",
            'confidence': "N/A",
            'analysis_type': 'Advanced Care Analysis',
            'recommendations': []
        }


# Initialize models on module load
specialized_models = load_specialized_models()
