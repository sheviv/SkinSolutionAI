import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Initialize scaler and model with optimized parameters
scaler = StandardScaler()
model = RandomForestClassifier(
    n_estimators=500,  # Increased number of trees
    max_depth=15,      # Increased depth for more complex patterns
    min_samples_split=4,
    min_samples_leaf=2,
    class_weight='balanced',  # Handle imbalanced classes
    random_state=42
)

def generate_sample_data():
    """Generate enhanced sample data for initial model training."""
    np.random.seed(42)
    n_samples = 2000  # Increased sample size

    # Generate feature data with more realistic distributions
    tone_uniformity = np.random.normal(75, 12, n_samples)
    brightness = np.random.normal(65, 15, n_samples)
    texture = np.random.normal(45, 8, n_samples)
    spots = np.random.poisson(4, n_samples)
    redness = np.random.normal(40, 10, n_samples)
    pigmentation = np.random.normal(35, 8, n_samples)

    X = np.column_stack([
        tone_uniformity, brightness, texture,
        spots, redness, pigmentation
    ])

    # Generate labels with more sophisticated conditions
    y = []
    for i in range(n_samples):
        # Acne-Prone Skin: High spot count OR (moderate spots AND high redness)
        if spots[i] > 7 or (spots[i] > 4 and redness[i] > 55):
            y.append("Acne-Prone Skin")
        # Uneven Skin Tone: Low uniformity AND high pigmentation variation
        elif tone_uniformity[i] < 65 and pigmentation[i] > 40:
            y.append("Uneven Skin Tone")
        # Dull Skin: Low brightness OR rough texture
        elif brightness[i] < 50 or texture[i] > 55:
            y.append("Dull Skin")
        # Healthy Skin: Good metrics across all features
        else:
            y.append("Healthy Skin")

    return X, np.array(y)

def train_model():
    """Train the skin analysis model with enhanced sample data."""
    X, y = generate_sample_data()

    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model with cross-validation
    model.fit(X_train_scaled, y_train)

    # Evaluate model
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Model Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(report)

    # Save model and scaler
    if not os.path.exists('models'):
        os.makedirs('models')
    joblib.dump(model, 'models/skin_analysis_model.joblib')
    joblib.dump(scaler, 'models/scaler.joblib')

def predict_skin_condition(features):
    """Predict skin condition with enhanced feature processing."""
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

        # Scale features
        scaled_features = scaler.transform(feature_values)

        # Make prediction with confidence threshold
        probabilities = model.predict_proba(scaled_features)[0]
        confidence = max(probabilities) * 100

        # Only predict a condition if confidence is high enough
        prediction = model.predict(scaled_features)[0]
        if confidence < 70:  # Increased confidence threshold
            # Use feature-based fallback logic for low confidence cases
            if feature_values[0][3] > 6:  # High spot count
                prediction = "Acne-Prone Skin"
            elif feature_values[0][0] < 65:  # Low uniformity
                prediction = "Uneven Skin Tone"
            elif feature_values[0][1] < 50:  # Low brightness
                prediction = "Dull Skin"
            else:
                prediction = "Healthy Skin"

        # Get feature importance
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
            )[:3]
        }
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return {
            'condition': "Healthy Skin",
            'confidence': "N/A",
            'probabilities': {},
            'key_factors': []
        }

# Train model on startup
train_model()