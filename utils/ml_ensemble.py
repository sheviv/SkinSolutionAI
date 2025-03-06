import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Initialize different models for the ensemble
random_forest = RandomForestClassifier(n_estimators=500, max_depth=15, random_state=42)
gradient_boosting = GradientBoostingClassifier(n_estimators=300, learning_rate=0.1, random_state=42)
svm_model = SVC(probability=True, random_state=42)

# Initialize scaler
ensemble_scaler = StandardScaler()

def train_ensemble_models():
    """Train all models in the ensemble with the same training data."""
    # Import here to avoid circular import
    from utils.ml_model import generate_sample_data

    X, y = generate_sample_data()

    # Split and scale data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train_scaled = ensemble_scaler.fit_transform(X_train)
    X_test_scaled = ensemble_scaler.transform(X_test)

    # Train all models
    models = [random_forest, gradient_boosting, svm_model]
    model_names = ["Random Forest", "Gradient Boosting", "SVM"]

    best_accuracy = 0
    most_reliable_model = "Random Forest"  # Default

    for model, name in zip(models, model_names):
        model.fit(X_train_scaled, y_train)

        # Save model
        joblib.dump(model, f'models/{name.lower().replace(" ", "_")}_model.joblib')

        # Evaluate on test set
        accuracy = model.score(X_test_scaled, y_test)
        print(f"{name} Accuracy: {accuracy:.4f}")

        # Track best performer
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            most_reliable_model = name

    # Save scaler
    joblib.dump(ensemble_scaler, 'models/ensemble_scaler.joblib')

    # Train specialized models
    from utils.specialized_models import train_severe_model, train_child_model
    train_severe_model()
    train_child_model()

    print(f"Most reliable model: {most_reliable_model}")
    return most_reliable_model

def load_ensemble_models():
    """Load all models in the ensemble from disk."""
    models = {}
    model_names = ["Random Forest", "Gradient Boosting", "SVM"]

    try:
        for name in model_names:
            filename = f'models/{name.lower().replace(" ", "_")}_model.joblib'
            if os.path.exists(filename):
                models[name] = joblib.load(filename)

        # Load scaler
        if os.path.exists('models/ensemble_scaler.joblib'):
            models['scaler'] = joblib.load('models/ensemble_scaler.joblib')

        # Load specialized models
        if os.path.exists('models/severe_skin_model.joblib'):
            models['Severe Conditions'] = joblib.load('models/severe_skin_model.joblib')
            models['severe_scaler'] = joblib.load('models/severe_scaler.joblib')

        if os.path.exists('models/child_skin_model.joblib'):
            models['Child Skin'] = joblib.load('models/child_skin_model.joblib')
            models['child_scaler'] = joblib.load('models/child_scaler.joblib')

        return models
    except Exception as e:
        print(f"Error loading ensemble models: {str(e)}")
        return {}

def predict_with_ensemble(features):
    """Get predictions from all ensemble models."""
    # Ensure models are trained
    if not os.path.exists('models/random_forest_model.joblib'):
        most_reliable = train_ensemble_models()
    else:
        # Default to Random Forest if we can't determine
        most_reliable = "Random Forest"

    # Load models if not already loaded
    models = load_ensemble_models()

    if not models:
        print("Models could not be loaded, training...")
        most_reliable = train_ensemble_models()
        models = load_ensemble_models()

    # Convert features to the right format
    feature_values = np.array([
        float(features["Tone Uniformity"]),
        float(features["Brightness"]),
        float(features["Texture"]),
        float(features["Spots Detected"]),
        float(features["Redness"]),
        float(features["Pigmentation"])
    ]).reshape(1, -1)

    # Scale features for standard models
    if 'scaler' in models:
        scaled_features = models['scaler'].transform(feature_values)
    else:
        # If scaler not found, just use raw features
        scaled_features = feature_values

    # Get predictions from each model
    results = {}
    highest_confidence = 0
    most_confident_model = most_reliable

    # Process standard models
    for name, model in models.items():
        if name in ['scaler', 'severe_scaler', 'child_scaler']:
            continue

        try:
            # Use appropriate scaler for specialized models
            if name == 'Severe Conditions' and 'severe_scaler' in models:
                model_scaled_features = models['severe_scaler'].transform(feature_values)
            elif name == 'Child Skin' and 'child_scaler' in models:
                model_scaled_features = models['child_scaler'].transform(feature_values)
            else:
                model_scaled_features = scaled_features

            probabilities = model.predict_proba(model_scaled_features)[0]
            confidence = max(probabilities) * 100
            prediction = model.predict(model_scaled_features)[0]

            # If this model is more confident, update most confident model
            if confidence > highest_confidence:
                highest_confidence = confidence
                most_confident_model = name

            # Try to get feature importances (not all models support this)
            key_factors = []
            if hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(
                    ["Tone", "Brightness", "Texture", "Spots", "Redness", "Pigmentation"],
                    model.feature_importances_
                ))
                key_factors = sorted(
                    feature_importance.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:3]
            elif name == 'Child Skin':
                # For neural network models, use a different approach
                coefs = model.coefs_[0]
                importances = np.sum(np.abs(coefs), axis=1)
                importances = importances / np.sum(importances)  # Normalize

                feature_importance = dict(zip(
                    ["Tone", "Brightness", "Texture", "Spots", "Redness", "Pigmentation"],
                    importances
                ))
                key_factors = sorted(
                    feature_importance.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:3]

            # Model type identifier helps with UI display
            model_type = "standard"
            if name == "Severe Conditions":
                model_type = "severe_conditions"
            elif name == "Child Skin":
                model_type = "child_skin"

            results[name] = {
                'condition': prediction,
                'confidence': f"{confidence:.1f}%",
                'probabilities': {
                    class_name: f"{prob * 100:.1f}%"
                    for class_name, prob in zip(model.classes_, probabilities)
                },
                'key_factors': key_factors,
                'model_type': model_type
            }

        except Exception as e:
            print(f"Error getting prediction from {name}: {str(e)}")

    # Add specialized model predictions directly from their modules if needed
    if 'Severe Conditions' not in results:
        try:
            from utils.specialized_models import predict_severe_condition
            results['Severe Conditions'] = predict_severe_condition(features)
        except Exception as e:
            print(f"Error loading severe conditions model: {str(e)}")

    if 'Child Skin' not in results:
        try:
            from utils.specialized_models import predict_child_skin_condition
            results['Child Skin'] = predict_child_skin_condition(features)
        except Exception as e:
            print(f"Error loading child skin model: {str(e)}")

    # Use higher confidence model instead of predefined "most reliable"
    results["most_reliable_model"] = most_confident_model

    return results