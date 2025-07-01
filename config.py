# Configuration for Student Success Prediction System
#
# Place a .env file in your project root with:
# GEMINI_API_KEY=your-gemini-api-key-here
# SUPABASE_URL=https://your-project-id.supabase.co
# SUPABASE_KEY=your-anon-key-here

import os
from dotenv import load_dotenv
import streamlit as st

load_dotenv()  # Load environment variables from .env if present

# Supabase Configuration
# ⚠️ IMPORTANT: Replace these placeholder values with your actual Supabase credentials!
# Get these from: https://supabase.com/dashboard → Settings → API
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

# Gemini API Key
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

# App Configuration
APP_NAME = "Student Success Prediction System"
APP_VERSION = "1.0.0"

# Session State Keys
SESSION_KEYS = [
    'model_trained', 'model', 'explainer', 'feature_cols', 'completion_rate_dicts',
    'df', 'model_gemini', 'show_predictions', 'show_factors', 'show_recommendations',
    'prediction_results', 'data_source', 'uploaded_df', 'upload_action'
]

# Validation Rules
PASSWORD_MIN_LENGTH = 6
USERNAME_MIN_LENGTH = 3 