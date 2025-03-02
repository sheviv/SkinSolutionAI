import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import joblib
from utils.ml_model import scaler, generate_sample_data

# Initialize models with optimized parameters
models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=500,
        max_depth=15,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=8,
        min_samples_split=5,
        random_state=42
    ),
    "Support Vector Machine": SVC(
        C=10.0,
        kernel='rbf',
        gamma='scale',
        probability=True,
        class_weight='balanced',
        random_state=42
    ),
    "K-Nearest Neighbors": KNeighborsClassifier(
        n_neighbors=7,
        weights='distance',
        algorithm='auto',
        leaf_size=30,
        p=2
    )
}


# Train all models
def train_ensemble():
    """Train all models in the ensemble using the same data."""
    X, y = generate_sample_data()
    X_scaled = scaler.fit_transform(X)

    for name, model in models.items():
        model.fit(X_scaled, y)
        print(f"Trained {name} model")


# Predict with all models
def predict_with_ensemble(features):
    """Get predictions from all models in the ensemble."""
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

        results = {}
        for name, model in models.items():
            # Make prediction
            prediction = model.predict(scaled_features)[0]
            probabilities = model.predict_proba(scaled_features)[0]
            confidence = max(probabilities) * 100

            results[name] = {
                'condition': prediction,
                'confidence': f"{confidence:.1f}%",
                'probabilities': {
                    class_name: f"{prob * 100:.1f}%"
                    for class_name, prob in zip(model.classes_, probabilities)
                }
            }

        # Determine most reliable model based on confidence
        most_reliable = max(results.items(), key=lambda x: float(x[1]['confidence'].rstrip('%')))

        # Add ensemble voting result (highest average prediction across models)
        all_probs = {}
        for name, result in results.items():
            for condition, prob_str in result['probabilities'].items():
                prob = float(prob_str.rstrip('%'))
                if condition not in all_probs:
                    all_probs[condition] = []
                all_probs[condition].append(prob)

        # Calculate average probability for each condition
        avg_probs = {cond: sum(probs) / len(probs) for cond, probs in all_probs.items()}
        ensemble_prediction = max(avg_probs.items(), key=lambda x: x[1])

        results["Ensemble Vote"] = {
            'condition': ensemble_prediction[0],
            'confidence': f"{ensemble_prediction[1]:.1f}%",
            'probabilities': {cond: f"{prob:.1f}%" for cond, prob in avg_probs.items()}
        }

        # Mark which model is used for final recommendations
        results["most_reliable_model"] = most_reliable[0]

        return results

    except Exception as e:
        print(f"Ensemble prediction error: {str(e)}")
        return {
            "error": f"Prediction failed: {str(e)}",
            "most_reliable_model": "Random Forest"  # Default fallback
        }


# Train models on import
train_ensemble()
