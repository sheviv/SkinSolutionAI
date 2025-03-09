import streamlit as st
import datetime
import uuid
import os
import cv2
import numpy as np
import json
from utils.database import db, User, PublishedAnalysis as PublishedAnalysisModel


class PublishedAnalysis:
    """Handles saving and retrieving published skin analyses"""

    @staticmethod
    def save_analysis(user_id, analysis_data):
        """Save analysis data to database or file system"""
        # Generate unique ID for the analysis
        analysis_id = str(uuid.uuid4())[:8]

        # Save image to file system
        image_path = None
        if 'image' in analysis_data and analysis_data['image'] is not None:
            image_path = f"data/published_images/{analysis_id}.jpg"
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            cv2.imwrite(image_path, analysis_data['image'])

        # Convert features to JSON string
        features_json = None
        if 'features' in analysis_data:
            features_json = json.dumps(analysis_data['features'])

        # Create database record
        try:
            new_analysis = PublishedAnalysisModel(
                id=analysis_id,
                user_id=user_id,
                condition=analysis_data.get('condition', 'Unknown'),
                features=features_json,
                image_path=image_path,
                timestamp=datetime.datetime.now(),
                is_public=True
            )

            # Add to database and commit
            db.session.add(new_analysis)
            db.session.commit()

            # For backward compatibility, also store in session state
            if 'published_analyses' not in st.session_state:
                st.session_state.published_analyses = {}

            st.session_state.published_analyses[analysis_id] = {
                'id': analysis_id,
                'user_id': user_id,
                'condition': analysis_data.get('condition', 'Unknown'),
                'features': analysis_data.get('features', {}),
                'image_path': image_path,
                'timestamp': datetime.datetime.now().isoformat(),
                'is_public': True
            }

            return analysis_id
        except Exception as e:
            # If database operation fails, rollback
            db.session.rollback()
            st.error(f"Error saving analysis: {str(e)}")
            return None

    @staticmethod
    def get_analysis(analysis_id):
        """Retrieve a published analysis by ID"""
        # First try to get from database
        from flask import current_app

        # Check if we're in an application context
        if current_app:
            # Already in app context
            analysis = PublishedAnalysisModel.query.filter_by(id=analysis_id).first()
        else:
            # Create app context
            from flask import Flask
            from utils.database import init_db

            app = Flask(__name__)
            init_db(app)

            with app.app_context():
                analysis = PublishedAnalysisModel.query.filter_by(id=analysis_id).first()

        if analysis:
            # Convert from database model to dictionary
            return {
                'id': analysis.id,
                'user_id': analysis.user_id,
                'condition': analysis.condition,
                'features': analysis.get_features_dict(),
                'image_path': analysis.image_path,
                'timestamp': analysis.timestamp.isoformat(),
                'is_public': analysis.is_public
            }

        # Fallback to session state for backward compatibility
        if 'published_analyses' in st.session_state:
            return st.session_state.published_analyses.get(analysis_id)

        return None

    @staticmethod
    def get_user_analyses(user_id):
        """Get all analyses published by a specific user"""
        from flask import current_app

        # Check if we're in an application context
        if current_app:
            # Already in app context
            analyses = PublishedAnalysisModel.query.filter_by(user_id=user_id).all()
        else:
            # Create app context
            from flask import Flask
            from utils.database import init_db

            app = Flask(__name__)
            init_db(app)

            with app.app_context():
                analyses = PublishedAnalysisModel.query.filter_by(user_id=user_id).all()

        if analyses:
            # Convert from database models to dictionary
            return {
                analysis.id: {
                    'id': analysis.id,
                    'user_id': analysis.user_id,
                    'condition': analysis.condition,
                    'features': analysis.get_features_dict(),
                    'image_path': analysis.image_path,
                    'timestamp': analysis.timestamp.isoformat(),
                    'is_public': analysis.is_public
                }
                for analysis in analyses
            }

        # Fallback to session state for backward compatibility
        if 'published_analyses' in st.session_state:
            return {k: v for k, v in st.session_state.published_analyses.items()
                    if v.get('user_id') == user_id}

        return {}

    @staticmethod
    def get_all_public_analyses():
        """Get all public analyses"""
        # Get from database
        from flask import current_app

        # Check if we're in an application context
        if current_app:
            # Already in an app context, proceed directly
            analyses = PublishedAnalysisModel.query.filter_by(is_public=True).order_by(
                PublishedAnalysisModel.timestamp.desc()
            ).all()

            if analyses:
                # Convert from database models to dictionary
                return {
                    analysis.id: {
                        'id': analysis.id,
                        'user_id': analysis.user_id,
                        'condition': analysis.condition,
                        'features': analysis.get_features_dict(),
                        'image_path': analysis.image_path,
                        'timestamp': analysis.timestamp.isoformat(),
                        'is_public': analysis.is_public
                    }
                    for analysis in analyses
                }
        else:
            # Not in app context, create one temporarily
            from flask import Flask
            from utils.database import init_db

            app = Flask(__name__)
            init_db(app)

            with app.app_context():
                analyses = PublishedAnalysisModel.query.filter_by(is_public=True).order_by(
                    PublishedAnalysisModel.timestamp.desc()
                ).all()

                if analyses:
                    # Convert from database models to dictionary
                    return {
                        analysis.id: {
                            'id': analysis.id,
                            'user_id': analysis.user_id,
                            'condition': analysis.condition,
                            'features': analysis.get_features_dict(),
                            'image_path': analysis.image_path,
                            'timestamp': analysis.timestamp.isoformat(),
                            'is_public': analysis.is_public
                        }
                        for analysis in analyses
                    }

        # Fallback to session state for backward compatibility
        if 'published_analyses' in st.session_state:
            return {k: v for k, v in st.session_state.published_analyses.items()
                    if v.get('is_public', True)}

        return {}
