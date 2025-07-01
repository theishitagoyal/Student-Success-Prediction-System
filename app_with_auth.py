import streamlit as st
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import shap
import numpy as np
import google.generativeai as genai
import json
import copy
import io
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from auth_supabase import auth_ui, require_auth, get_current_user
from config import SESSION_KEYS, SUPABASE_URL, SUPABASE_KEY, GEMINI_API_KEY
from supabase import create_client
import time
from datetime import datetime, timedelta

# --- ADVANCED MODEL FUNCTIONS (moved up for visibility and to avoid NameError) ---
def create_advanced_features(df):
    """Create advanced features for better model performance"""
    df_enhanced = df.copy()
    # Ensure days_remaining is always >= 0
    df_enhanced['days_remaining'] = df_enhanced['days_remaining'].clip(lower=0)
    # Time-based features
    df_enhanced['is_weekend'] = df_enhanced['day_of_week'].apply(lambda x: 1 if x in ['Saturday', 'Sunday'] else 0)
    df_enhanced['is_monday'] = df_enhanced['day_of_week'].apply(lambda x: 1 if x == 'Monday' else 0)
    df_enhanced['is_friday'] = df_enhanced['day_of_week'].apply(lambda x: 1 if x == 'Friday' else 0)
    # Time of day features
    df_enhanced['is_morning'] = df_enhanced['time_of_day'].apply(lambda x: 1 if x == 'Morning' else 0)
    df_enhanced['is_evening'] = df_enhanced['time_of_day'].apply(lambda x: 1 if x == 'Evening' else 0)
    df_enhanced['is_night'] = df_enhanced['time_of_day'].apply(lambda x: 1 if x == 'Night' else 0)
    # Task complexity features
    df_enhanced['task_complexity'] = df_enhanced['task_difficulty'] * df_enhanced['days_remaining']
    df_enhanced['urgency_score'] = df_enhanced['overdue_tasks'] + (1 / (df_enhanced['days_remaining'] + 1))
    # Interaction features
    df_enhanced['difficulty_urgency'] = df_enhanced['task_difficulty'] * df_enhanced['urgency_score']
    df_enhanced['weekend_difficulty'] = df_enhanced['is_weekend'] * df_enhanced['task_difficulty']
    return df_enhanced

def train_ensemble_model(X, y):
    """Train an ensemble model combining XGBoost and Random Forest"""
    # XGBoost with hyperparameter tuning
    xgb_params = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.1, 0.2],
        'subsample': [0.8, 0.9]
    }
    xgb_model = xgb.XGBClassifier(random_state=42)
    xgb_grid = GridSearchCV(xgb_model, xgb_params, cv=3, scoring='accuracy', n_jobs=-1)
    xgb_grid.fit(X, y)
    # Random Forest with hyperparameter tuning
    rf_params = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    rf_model = RandomForestClassifier(random_state=42)
    rf_grid = GridSearchCV(rf_model, rf_params, cv=3, scoring='accuracy', n_jobs=-1)
    rf_grid.fit(X, y)
    # Create ensemble
    ensemble = VotingClassifier(
        estimators=[
            ('xgb', xgb_grid.best_estimator_),
            ('rf', rf_grid.best_estimator_)
        ],
        voting='soft'
    )
    ensemble.fit(X, y)
    return ensemble, xgb_grid.best_estimator_, rf_grid.best_estimator_

def evaluate_model_performance(model, X, y, model_name="Model"):
    """Comprehensive model evaluation"""
    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    # Predictions
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else None
    # Metrics
    accuracy = accuracy_score(y, y_pred)
    return {
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'accuracy': accuracy,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }

def get_feature_importance(model, feature_names):
    """Extract feature importance from model"""
    if hasattr(model, 'feature_importances_'):
        return dict(zip(feature_names, model.feature_importances_))
    elif hasattr(model, 'named_estimators_'):
        # For ensemble models, average the feature importances
        importances = {}
        for name, estimator in model.named_estimators_.items():
            if hasattr(estimator, 'feature_importances_'):
                for feat, imp in zip(feature_names, estimator.feature_importances_):
                    importances[feat] = importances.get(feat, 0) + imp
        # Average the importances
        n_estimators = len(model.named_estimators_)
        return {k: v/n_estimators for k, v in importances.items()}
    return {} 

# Supabase client for tasks
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Configure page
st.set_page_config(
    page_title="Student Success Predictor",
    page_icon="🎓",
    layout="wide"
)

# Authentication Check - Must be logged in to access the app
if not auth_ui():
    st.stop()

current_user = get_current_user()
if not current_user:
    st.error("User not found. Please log in again.")
    st.stop()

user_id = current_user['id']

# Fetch user tasks and count for all pages
def fetch_user_tasks(user_id):
    try:
        response = supabase.table('user_tasks').select('*').eq('user_id', user_id).order('created_at', desc=False).limit(2000).execute()
        if response.data:
            return response.data
        return []
    except Exception as e:
        st.error(f"Error fetching tasks: {e}")
        return []

def add_user_task(user_id, task):
    try:
        task['user_id'] = user_id
        result = supabase.table('user_tasks').insert(task).execute()
        return bool(result.data)
    except Exception as e:
        st.error(f"Error adding task: {e}")
        return False

user_tasks = fetch_user_tasks(user_id)
task_count = len(user_tasks)

# User is now authenticated, show the main app
st.title("🎓 Student Success Prediction System")
st.markdown("---")

# Show current user info
st.sidebar.success(f"👤 Logged in as: {current_user['username']}")

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'explainer' not in st.session_state:
    st.session_state.explainer = None
if 'feature_cols' not in st.session_state:
    st.session_state.feature_cols = None
if 'completion_rate_dicts' not in st.session_state:
    st.session_state.completion_rate_dicts = {}
if 'df' not in st.session_state:
    st.session_state.df = None
if 'model_gemini' not in st.session_state:
    st.session_state.model_gemini = None
if 'show_predictions' not in st.session_state:
    st.session_state.show_predictions = False
if 'show_factors' not in st.session_state:
    st.session_state.show_factors = False
if 'show_recommendations' not in st.session_state:
    st.session_state.show_recommendations = False
if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = None

# Data source selection state
if 'data_source' not in st.session_state:
    st.session_state['data_source'] = 'supabase'  # 'supabase' or 'uploaded'
if 'uploaded_df' not in st.session_state:
    st.session_state['uploaded_df'] = None

# Hardcoded API Key
try:
    genai.configure(api_key=GEMINI_API_KEY)
    st.session_state.model_gemini = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    st.error(f"❌ API Key error: {str(e)}")

# Sidebar for Data Upload
st.sidebar.header("📁 Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'], key="csv_uploader")

# Action descriptions for tooltips and messages
action_options = [
    ("Use for this session only", "Temporarily use this data. It will be lost when you log out or refresh."),
    ("Replace tracked tasks", "Permanently replace all your tracked tasks with this uploaded data. This will be saved in your account and available in all future sessions."),
    ("Merge with tracked tasks", "Permanently add this uploaded data to your tracked tasks, avoiding duplicates. This will be saved in your account and available in all future sessions.")
]

action_labels = [opt[0] for opt in action_options]
# Prepare selectbox options with an info icon
action_labels_with_info = [f"{label} ℹ️" for label in action_labels]
label_to_desc = dict(zip(action_labels_with_info, [desc for _, desc in action_options]))
label_to_action = dict(zip(action_labels_with_info, action_labels))

if uploaded_file is not None:
    uploaded_df = pd.read_csv(uploaded_file)
    st.session_state['uploaded_df'] = uploaded_df
    st.sidebar.success(f"✅ File uploaded! Shape: {uploaded_df.shape}")
    # Show data preview
    with st.sidebar.expander("📊 Uploaded Data Preview"):
        st.dataframe(uploaded_df.head())
    # Show action selection only if a file is uploaded
    st.sidebar.markdown("**How do you want to use this data?**")
    selected_label = st.sidebar.selectbox(
        "Choose action:",
        action_labels_with_info,
        index=0,
        key="upload_action"
    )
    if selected_label not in label_to_desc:
        selected_label = action_labels_with_info[0]
    st.sidebar.markdown(f'<span style="color:#888;">{label_to_desc[selected_label]}</span>', unsafe_allow_html=True)
    action = label_to_action[selected_label]
    if st.sidebar.button("Apply Dataset Action", key="apply_dataset_action"):
        if action == "Use for this session only":
            st.session_state['data_source'] = 'uploaded'
            st.sidebar.info("Using uploaded data for this session only. Your tracked tasks are unchanged.")
        elif action == "Replace tracked tasks":
            # Clear user's tasks in Supabase and insert uploaded data
            try:
                # Delete all user_tasks for this user
                supabase.table('user_tasks').delete().eq('user_id', user_id).execute()
                # Insert uploaded data
                for _, row in uploaded_df.iterrows():
                    task = row.to_dict()
                    task['user_id'] = user_id
                    supabase.table('user_tasks').insert(task).execute()
                st.session_state['data_source'] = 'supabase'
                st.sidebar.success("Replaced tracked tasks with uploaded data.")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Error replacing tasks: {e}")
        elif action == "Merge with tracked tasks":
            # Add uploaded data to Supabase, skipping duplicates
            try:
                existing_tasks = fetch_user_tasks(user_id)
                existing_set = set(tuple(row.items()) for row in pd.DataFrame(existing_tasks).drop(columns=['id','user_id','created_at'], errors='ignore').to_dict('records'))
                new_rows = 0
                for _, row in uploaded_df.iterrows():
                    task_tuple = tuple((k, row[k]) for k in row.index if k != 'id' and k != 'user_id' and k != 'created_at')
                    if task_tuple not in existing_set:
                        task = row.to_dict()
                        task['user_id'] = user_id
                        supabase.table('user_tasks').insert(task).execute()
                        new_rows += 1
                st.session_state['data_source'] = 'supabase'
                st.sidebar.success(f"Merged uploaded data. {new_rows} new tasks added.")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Error merging tasks: {e}")

# If the uploaded file is removed, clear all related session state and rerun
if 'uploaded_df' in st.session_state and st.session_state['uploaded_df'] is not None and uploaded_file is None:
    st.session_state['uploaded_df'] = None
    st.session_state['data_source'] = 'supabase'
    st.session_state['upload_action'] = action_labels_with_info[0]  # Reset selectbox to default label
    st.rerun()

# --- DATASET SELECTION FOR APP ---
# Ensure we never use uploaded data if it has been removed
if st.session_state['data_source'] == 'uploaded' and (('uploaded_df' not in st.session_state) or (st.session_state['uploaded_df'] is None)):
    st.session_state['data_source'] = 'supabase'

if st.session_state['data_source'] == 'uploaded' and st.session_state['uploaded_df'] is not None:
    selected_tasks = st.session_state['uploaded_df'].to_dict('records')
    selected_task_count = len(selected_tasks)
else:
    selected_tasks = user_tasks
    selected_task_count = task_count

# Helper Functions
def compute_completion_rate_dict(df, feature):
    rate_series = df.groupby(feature)['completed_on_time'].mean()
    return rate_series.to_dict()

def prepare_input_for_prediction(input_data, feature_cols_order):
    is_weekend = 1 if input_data['day_of_week'] in ['Saturday', 'Sunday'] else 0
    
    input_df = pd.DataFrame([{
        'overdue_tasks': input_data['overdue_tasks'],
        'task_difficulty': input_data['task_difficulty'],
        'days_remaining': input_data['days_remaining'],
        'is_weekend': is_weekend,
        'task_type': input_data['task_type'],
        'time_of_day': input_data['time_of_day']
    }])
    
    input_encoded = pd.get_dummies(input_df, columns=['task_type', 'time_of_day'])
    
    prepared_df = pd.DataFrame(columns=feature_cols_order)
    for col in feature_cols_order:
        if col in input_encoded.columns:
            prepared_df[col] = input_encoded[col]
        else:
            prepared_df[col] = 0
    
    return prepared_df.iloc[[0]]

def filter_shap_factors(shap_df, input_data):
    filtered_factors = []
    
    for _, row in shap_df.iterrows():
        feature_name = row['feature']
        
        if feature_name == 'task_difficulty' or feature_name.startswith('task_type_'):
            continue
        
        if feature_name.startswith('time_of_day_'):
            time_value = feature_name.replace('time_of_day_', '')
            if time_value != input_data['time_of_day']:
                continue
        
        filtered_factors.append(row)
    
    return pd.DataFrame(filtered_factors) if filtered_factors else pd.DataFrame(columns=['feature', 'shap_value'])

def generate_explanation_only(analysis_results):
    if not st.session_state.model_gemini:
        return "⚠️ Gemini API key required for detailed explanations"
    
    predicted_prob = analysis_results['predicted_probability']
    task_type_rate = analysis_results['task_type_completion_rate']
    task_difficulty_rate = analysis_results['task_difficulty_completion_rate']
    negative_factors = analysis_results['negative_factors']
    student_input = analysis_results['student_input']
    
    prompt = f"""
Write an explanation following this EXACT structure:

1. Start with: "The task has a [X]% success probability of getting completed on time"
2. Then compare: "which is [better/worse] than the [X]% rate for Level [difficulty] tasks and [better/worse] than the [X]% [task_type] completion rate"
3. Then explain what's causing the probability dip: "Your [specific student_input parameters] are creating [impact description] with [impact values]"

DO NOT include any recommendations or actions.
DO NOT use "**Actions:**" section.
Focus only on explanation of current situation.

Data to use:
- Predicted probability: {predicted_prob*100:.0f}%
- Task difficulty rate: {task_difficulty_rate*100:.0f}% for Level {student_input['task_difficulty']}
- Task type rate: {task_type_rate*100:.0f}% for {student_input['task_type']}
- Negative factors: {negative_factors}
- Student input: {student_input}

Write only the explanation paragraph.
"""
    
    try:
        response = st.session_state.model_gemini.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error generating explanation: {str(e)}"

def generate_recommendations_for_negative_factors(original_input, negative_shap_factors, feature_cols, df):
    recommendations = []
    original_input_prepared = prepare_input_for_prediction(original_input, feature_cols)
    original_prob = st.session_state.model.predict_proba(original_input_prepared)[0][1]

    # User-controllable features
    controllable = ['time_of_day']
    possible_values = {k: df[k].unique().tolist() for k in controllable if k in df.columns}

    for param in controllable:
        if param not in original_input or param not in possible_values:
            continue
        best_improvement = 0
        best_value = None
        best_new_prob = original_prob
        for test_value in possible_values[param]:
            if test_value == original_input[param]:
                continue
            modified_input = copy.deepcopy(original_input)
            modified_input[param] = test_value
            try:
                modified_input_prepared = prepare_input_for_prediction(modified_input, feature_cols)
                new_prob = st.session_state.model.predict_proba(modified_input_prepared)[0][1]
                improvement = new_prob - original_prob
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_value = test_value
                    best_new_prob = new_prob
            except Exception:
                continue
        if best_improvement > 0:
            recommendations.append({
                'parameter': param,
                'current_value': original_input[param],
                'recommended_value': best_value,
                'probability_increase': best_improvement,
                'new_probability': best_new_prob,
                'original_shap_impact': None
            })
    return sorted(recommendations, key=lambda x: x['probability_increase'], reverse=True)

def create_completion_rate_graphs():
    if st.session_state.df is None:
        return
    
    df = st.session_state.df
    
    # Create completion rate graphs
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Task Type completion rates
    task_type_rates = df.groupby('task_type')['completed_on_time'].mean()
    axes[0, 0].bar(task_type_rates.index, task_type_rates.values)
    axes[0, 0].set_title('Completion Rate by Task Type')
    axes[0, 0].set_ylabel('Completion Rate')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Task Difficulty completion rates
    difficulty_rates = df.groupby('task_difficulty')['completed_on_time'].mean()
    axes[0, 1].bar(difficulty_rates.index, difficulty_rates.values)
    axes[0, 1].set_title('Completion Rate by Task Difficulty')
    axes[0, 1].set_ylabel('Completion Rate')
    
    # Time of Day completion rates
    time_rates = df.groupby('time_of_day')['completed_on_time'].mean()
    axes[1, 0].bar(time_rates.index, time_rates.values)
    axes[1, 0].set_title('Completion Rate by Time of Day')
    axes[1, 0].set_ylabel('Completion Rate')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Days Remaining vs Completion Rate
    days_rates = df.groupby('days_remaining')['completed_on_time'].mean()
    axes[1, 1].scatter(days_rates.index, days_rates.values)
    axes[1, 1].set_title('Completion Rate by Days Remaining')
    axes[1, 1].set_xlabel('Days Remaining')
    axes[1, 1].set_ylabel('Completion Rate')
    
    plt.tight_layout()
    st.pyplot(fig)

# Sidebar navigation
st.sidebar.header("🧭 Navigation")
page = st.sidebar.selectbox(
    "Choose a page:",
    ["Task Tracker", "Prediction", "Statistics", "Model Insights"],
    index=0
)

# --- TASK TRACKER PAGE ---
if page == "Task Tracker":
    st.header("📝 Task Tracker")
    
    # Progress tracking with enhanced styling
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; margin: 20px 0;">
        <h2 style="color: white; text-align: center; margin: 0;">📊 Data Collection Progress</h2>
        <p style="color: white; text-align: center; margin: 10px 0 0 0;">Track your tasks to unlock AI predictions</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced progress display
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Tasks Logged", selected_task_count, help="Total number of tasks you've logged")
    
    with col2:
        if selected_task_count >= 100:
            st.success("✅ Predictions Unlocked!")
        elif selected_task_count >= 50:
            st.warning("⚠️ Early Access Available")
        else:
            st.info("📈 Collecting Data...")
    
    with col3:
        if selected_task_count < 100:
            progress = min(selected_task_count, 100) / 100
            st.progress(progress, text=f"{selected_task_count}/100 tasks")
        else:
            st.success("🎯 Target Reached!")

    # Task input form with enhanced styling
    st.subheader("➕ Add New Task")
    
    with st.form("task_form", clear_on_submit=True):
        st.markdown("**📋 Task Information**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            overdue_tasks = st.number_input("Overdue Tasks", min_value=0, max_value=20, value=0, help="How many tasks are currently overdue?")
            task_type = st.selectbox("Task Type", ["Quiz", "Assignment", "Project"], help="What type of task is this?")
            task_difficulty = st.selectbox("Task Difficulty", [1, 2, 3], help="Rate the difficulty level (1=Easy, 2=Medium, 3=Hard)")
        
        with col2:
            days_remaining = st.number_input("Days Remaining", min_value=0, max_value=30, value=1, help="How many days until the deadline?")
            day_of_week = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], help="What day of the week is this?")
            time_of_day = st.selectbox("Time of Day", ["Morning", "Afternoon", "Evening", "Night"], help="When do you plan to work on this?")
        
        completed_on_time = st.selectbox("Completed On Time", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", help="Did you complete this task on time?")
        
        submitted = st.form_submit_button("➕ Add Task", type="primary", use_container_width=True)
        
        if submitted:
            task = {
                'overdue_tasks': overdue_tasks,
                'task_type': task_type,
                'task_difficulty': task_difficulty,
                'days_remaining': days_remaining,
                'day_of_week': day_of_week,
                'time_of_day': time_of_day,
                'completed_on_time': completed_on_time
            }
            if add_user_task(user_id, task):
                st.success("✅ Task added successfully!")
                time.sleep(1)
                st.rerun()

    # Enhanced task display
    st.subheader("📋 Your Task History")
    
    if selected_tasks:
        # Summary statistics
        df_display = pd.DataFrame(selected_tasks)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            success_rate = df_display['completed_on_time'].mean()
            st.metric("Success Rate", f"{success_rate:.1%}")
        
        with col2:
            avg_difficulty = df_display['task_difficulty'].mean()
            st.metric("Avg Difficulty", f"{avg_difficulty:.1f}/3")
        
        with col3:
            most_common_type = df_display['task_type'].mode().iloc[0] if not df_display['task_type'].mode().empty else "N/A"
            st.metric("Most Common Type", most_common_type)
        
        with col4:
            avg_days = df_display['days_remaining'].mean()
            st.metric("Avg Days Remaining", f"{avg_days:.1f}")
        
        # Task table with enhanced styling
        st.dataframe(df_display, hide_index=True, use_container_width=True)
        
        # Quick insights
        st.subheader("💡 Quick Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Best performing task type
            type_success = df_display.groupby('task_type')['completed_on_time'].mean().sort_values(ascending=False)
            best_type = type_success.index[0]
            best_rate = type_success.iloc[0]
            st.info(f"🎯 **Best performing task type**: {best_type} ({best_rate:.1%} success rate)")
        
        with col2:
            # Best time of day
            time_success = df_display.groupby('time_of_day')['completed_on_time'].mean().sort_values(ascending=False)
            best_time = time_success.index[0]
            best_time_rate = time_success.iloc[0]
            st.info(f"⏰ **Best time to work**: {best_time} ({best_time_rate:.1%} success rate)")
        
    else:
        st.info("📝 No tasks logged yet. Start by adding your first task above!")
        
        # Quick start guide
        with st.expander("🚀 Quick Start Guide"):
            st.markdown("""
            **How to get started:**
            1. **Add your first task** using the form above
            2. **Log 50+ tasks** to unlock early predictions
            3. **Reach 100+ tasks** for reliable predictions
            4. **Track consistently** to improve model accuracy
            
            **Tips for better predictions:**
            - Be honest about completion status
            - Log tasks regularly
            - Include variety in task types and difficulties
            - Note your actual working patterns
            """)

# --- PREDICTION PAGE ---
elif page == "Prediction":
    st.header("🎯 Prediction")
    if selected_task_count < 50:
        st.warning("Prediction features will unlock after you log 50 tasks.")
        st.progress(min(selected_task_count, 50) / 50, text=f"{selected_task_count}/50 tasks logged to unlock predictions")
        st.stop()
    elif selected_task_count < 100:
        st.warning("You have early access to predictions, but results may be unreliable until you log at least 100 tasks.")
        st.progress(selected_task_count / 100, text=f"{selected_task_count}/100 tasks logged for accurate predictions")

    # Retrain model every 50 new tasks, up to 2000
    def should_retrain_model(task_count):
        return (task_count >= 100) and (task_count <= 2000) and (task_count % 50 == 0)

    if should_retrain_model(selected_task_count):
        st.info(f"Retraining model on {selected_task_count} tasks...")
        # (Retrain model code will run below, using selected_tasks as the dataset)

    # Main App Logic
    if selected_task_count < 100:
        st.warning("Prediction features will unlock after you log 100 tasks.")
        st.stop()

    # Ensure df is defined from the selected data source
    if st.session_state['data_source'] == 'uploaded' and st.session_state['uploaded_df'] is not None:
        df = st.session_state['uploaded_df'].copy()
    else:
        df = pd.DataFrame(selected_tasks)
    st.session_state.df = df  # Always set the current DataFrame in session state

    # Train model if not already trained
    if not st.session_state.model_trained:
        st.header("🤖 Enhanced Model Training")
        
        with st.spinner("Training advanced ensemble model..."):
            # Create enhanced features
            df_enhanced = create_advanced_features(df)
            
            # Prepare features (including new engineered features)
            base_features = ['overdue_tasks', 'task_difficulty', 'days_remaining']
            engineered_features = ['is_weekend', 'is_monday', 'is_friday', 'is_morning', 
                                 'is_evening', 'is_night', 'task_complexity', 'urgency_score',
                                 'difficulty_urgency', 'weekend_difficulty']
            
            # One-hot encode categorical variables
            df_encoded = pd.get_dummies(df_enhanced, columns=['task_type', 'time_of_day'])
            categorical_features = [col for col in df_encoded.columns if col.startswith(('task_type_', 'time_of_day_'))]
            # Clean inf/-inf/NaN
            df_encoded = df_encoded.replace([np.inf, -np.inf], np.nan)
            df_encoded = df_encoded.fillna(0)
            # Combine all features
            feature_cols = base_features + engineered_features + categorical_features
            
            # Prepare target
            y = df_encoded['completed_on_time']
            X = df_encoded[feature_cols]
            
            # Train ensemble model
            ensemble_model, xgb_model, rf_model = train_ensemble_model(X, y)
            
            # Evaluate models
            ensemble_eval = evaluate_model_performance(ensemble_model, X, y, "Ensemble")
            xgb_eval = evaluate_model_performance(xgb_model, X, y, "XGBoost")
            rf_eval = evaluate_model_performance(rf_model, X, y, "Random Forest")
            
            # Create SHAP explainer for ensemble
            explainer = shap.TreeExplainer(xgb_model)  # Use XGBoost for SHAP
            
            # Get feature importance
            feature_importance = get_feature_importance(ensemble_model, feature_cols)
            
            # Store in session state
            st.session_state.model = ensemble_model
            st.session_state.xgb_model = xgb_model
            st.session_state.rf_model = rf_model
            st.session_state.explainer = explainer
            st.session_state.feature_cols = feature_cols
            st.session_state.model_trained = True
            st.session_state.feature_importance = feature_importance
            st.session_state.model_evaluations = {
                'ensemble': ensemble_eval,
                'xgb': xgb_eval,
                'rf': rf_eval
            }
            
            # Compute completion rate dictionaries
            st.session_state.completion_rate_dicts = {
                'task_type': compute_completion_rate_dict(df, 'task_type'),
                'task_difficulty': compute_completion_rate_dict(df, 'task_difficulty'),
                'time_of_day': compute_completion_rate_dict(df, 'time_of_day')
            }
        
        st.success("✅ Advanced ensemble model trained successfully!")

    # Prediction Interface
    st.header("🎯 Make Predictions")

    # Modern dashboard layout
    st.markdown("""
    <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; margin: 20px 0;">
        <h2 style="color: white; text-align: center; margin: 0;">📊 Prediction Dashboard</h2>
        <p style="color: white; text-align: center; margin: 10px 0 0 0;">Enter your task details below to get AI-powered predictions</p>
    </div>
    """, unsafe_allow_html=True)

    # Input form with better styling
    with st.container():
        st.markdown("### 📝 Task Information")
        
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**📚 Task Details**")
            overdue_tasks = st.number_input("Number of Overdue Tasks", min_value=0, value=2, help="How many tasks are currently overdue?")
            task_difficulty = st.selectbox("Task Difficulty", sorted(df['task_difficulty'].unique()), help="Rate the difficulty level of this task")

        with col2:
            st.markdown("**⏰ Time Constraints**")
            days_remaining = st.number_input("Days Remaining", min_value=1, value=5, help="How many days until the deadline?")
            day_of_week = st.selectbox("Day of Week", df['day_of_week'].unique(), help="What day of the week is this?")

        with col3:
            st.markdown("**📋 Task Type & Schedule**")
            task_type = st.selectbox("Task Type", df['task_type'].unique(), help="What type of task is this?")
            time_of_day = st.selectbox("Time of Day", df['time_of_day'].unique(), help="When do you plan to work on this?")

    # Enhanced action buttons with better styling
    st.markdown("---")
    st.markdown("### 🚀 Analysis Actions")
    
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📊 Make Predictions and Analysis Results", type="primary", key="predict_btn", use_container_width=True):
            st.session_state.show_predictions = True
            st.session_state.show_factors = False
            st.session_state.show_recommendations = False
            
            # Create enhanced input for prediction
            test_input = {
                'overdue_tasks': overdue_tasks,
                'task_difficulty': task_difficulty,
                'days_remaining': days_remaining,
                'day_of_week': day_of_week,
                'task_type': task_type,
                'time_of_day': time_of_day
            }
            
            try:
                # Create enhanced features for prediction
                input_df = pd.DataFrame([test_input])
                input_enhanced = create_advanced_features(input_df)
                input_encoded = pd.get_dummies(input_enhanced, columns=['task_type', 'time_of_day'])
                
                # Ensure all features are present
                for col in st.session_state.feature_cols:
                    if col not in input_encoded.columns:
                        input_encoded[col] = 0
                
                input_encoded = input_encoded[st.session_state.feature_cols]
                
                # Make prediction with ensemble
                original_prob = st.session_state.model.predict_proba(input_encoded)[0][1]
                original_prediction = st.session_state.model.predict(input_encoded)[0]
                
                # Calculate SHAP values
                shap_values = st.session_state.explainer(input_encoded)
                shap_df = pd.DataFrame({
                    'feature': input_encoded.columns,
                    'shap_value': shap_values.values[0]
                })
                
                # Filter SHAP factors
                filtered_shap_df = filter_shap_factors(shap_df, test_input)
                negative_shap_factors = filtered_shap_df[filtered_shap_df['shap_value'] < 0].sort_values(by='shap_value')
                
                # Store results
                st.session_state.prediction_results = {
                    'test_input': test_input,
                    'original_prob': original_prob,
                    'original_prediction': original_prediction,
                    'negative_shap_factors': negative_shap_factors,
                    'original_input_prepared': input_encoded
                }
                
            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")

    with col2:
        if st.button("⚠️ Factors Reducing Chances and Probability Analysis", type="secondary", key="factors_btn", use_container_width=True):
            st.session_state.show_factors = True
            st.session_state.show_predictions = False
            st.session_state.show_recommendations = False

    with col3:
        if st.button("💡 Personalized Recommendations", type="secondary", key="recommendations_btn", use_container_width=True):
            st.session_state.show_recommendations = True
            st.session_state.show_predictions = False
            st.session_state.show_factors = False

    # Display results based on button clicks
    if st.session_state.show_predictions and st.session_state.prediction_results:
        results = st.session_state.prediction_results
        test_input = results['test_input']
        original_prob = results['original_prob']
        original_prediction = results['original_prediction']
        
        st.markdown("---")
        st.header("📈 Advanced Analysis Results")
        
        # Enhanced metrics display
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Success Probability", f"{original_prob:.1%}", 
                     delta=f"{original_prob - 0.5:.1%}" if original_prob > 0.5 else f"{original_prob - 0.5:.1%}")
        
        with col2:
            task_type_rate = st.session_state.completion_rate_dicts['task_type'].get(test_input['task_type'], 0.0)
            st.metric(f"{test_input['task_type']} Success Rate", f"{task_type_rate:.1%}")
        
        with col3:
            task_difficulty_rate = st.session_state.completion_rate_dicts['task_difficulty'].get(test_input['task_difficulty'], 0.0)
            st.metric(f"Level {test_input['task_difficulty']} Success Rate", f"{task_difficulty_rate:.1%}")
        
        with col4:
            time_rate = st.session_state.completion_rate_dicts['time_of_day'].get(test_input['time_of_day'], 0.0)
            st.metric(f"{test_input['time_of_day']} Success Rate", f"{time_rate:.1%}")
        
        # Enhanced prediction result with confidence levels
        confidence_level = "High" if abs(original_prob - 0.5) > 0.3 else "Medium" if abs(original_prob - 0.5) > 0.15 else "Low"
        confidence_color = "#28a745" if confidence_level == "High" else "#ffc107" if confidence_level == "Medium" else "#dc3545"
        
        if original_prediction == 1:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%); border: 3px solid #28a745; border-radius: 15px; padding: 30px; text-align: center; margin: 20px 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <h1 style="color: #155724; font-size: 2.5em; margin: 0;">✅ WILL COMPLETE ON TIME</h1>
                <p style="color: #155724; font-size: 1.2em; margin: 10px 0;">Confidence Level: <span style="color: {confidence_color}; font-weight: bold;">{confidence_level}</span></p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%); border: 3px solid #dc3545; border-radius: 15px; padding: 30px; text-align: center; margin: 20px 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <h1 style="color: #721c24; font-size: 2.5em; margin: 0;">❌ WILL NOT COMPLETE ON TIME</h1>
                <p style="color: #721c24; font-size: 1.2em; margin: 10px 0;">Confidence Level: <span style="color: {confidence_color}; font-weight: bold;">{confidence_level}</span></p>
            </div>
            """, unsafe_allow_html=True)
        
        # Enhanced probability gauge
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = original_prob * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Success Probability (%)", 'font': {'size': 20}},
            delta = {'reference': 50, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 30], 'color': "lightcoral"},
                    {'range': [30, 60], 'color': "lightyellow"},
                    {'range': [60, 80], 'color': "lightgreen"},
                    {'range': [80, 100], 'color': "green"}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90}}))
        
        fig.update_layout(
            height=400,
            font={'color': "darkblue", 'family': "Arial"},
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Model confidence explanation
        st.info(f"🤖 **Model Confidence**: The ensemble model combines XGBoost and Random Forest predictions for higher accuracy. Current confidence level is **{confidence_level}** based on the probability margin from 50%.")

    if st.session_state.show_factors and st.session_state.prediction_results:
        results = st.session_state.prediction_results
        test_input = results['test_input']
        original_prob = results['original_prob']
        negative_shap_factors = results['negative_shap_factors']
        
        st.markdown("---")
        st.header("⚠️ Factors Reducing Success Chances")
        
        if len(negative_shap_factors) > 0:
            for idx, (_, row) in enumerate(negative_shap_factors.iterrows(), 1):
                feature_name = row['feature']
                shap_value = row['shap_value']
                # Only show if the feature is active (value 1) in the input for binary features
                show = True
                if feature_name in results['original_input_prepared'].columns:
                    val = results['original_input_prepared'][feature_name].iloc[0]
                    if feature_name.startswith('is_') and val != 1:
                        show = False
                if show:
                    readable_name = feature_name.replace('_', ' ').title()
                    if 'Time Of Day' in readable_name:
                        readable_name = readable_name.replace('Time Of Day ', 'Time: ')
                    st.write(f"{idx}. **{readable_name}** (Impact: {shap_value:.3f})")
        else:
            st.success("No major negative factors identified!")
        
        # Probability Analysis
        st.subheader("🎯 Probability Analysis")
        task_type_rate = st.session_state.completion_rate_dicts['task_type'].get(test_input['task_type'], 0.0)
        task_difficulty_rate = st.session_state.completion_rate_dicts['task_difficulty'].get(test_input['task_difficulty'], 0.0)
        
        analysis_results = {
            'student_input': test_input,
            'task_type_completion_rate': task_type_rate,
            'task_difficulty_completion_rate': task_difficulty_rate,
            'predicted_probability': float(original_prob),
            'prediction': 'will complete' if results['original_prediction'] == 1 else 'will not complete',
            'negative_factors': [
                {
                    'factor': row['feature'].replace('_', ' ').title(),
                    'impact': row['shap_value']
                }
                for _, row in negative_shap_factors.iterrows()
            ]
        }
        
        explanation = generate_explanation_only(analysis_results)
        st.write(explanation)

    if st.session_state.show_recommendations and st.session_state.prediction_results:
        results = st.session_state.prediction_results
        test_input = results['test_input']
        original_prob = results['original_prob']
        negative_shap_factors = results['negative_shap_factors']
        
        st.markdown("---")
        st.header("💡 Personalized Recommendations")
        
        recommendations = generate_recommendations_for_negative_factors(
            test_input, negative_shap_factors, st.session_state.feature_cols, st.session_state.df
        )
        
        if recommendations:
            st.write(f"**Current probability:** {original_prob:.1%}")
            st.write("**Actionable changes that could improve your success probability:**")
            
            for i, rec in enumerate(recommendations, 1):
                with st.expander(f"📈 Recommendation {i}: {rec['parameter'].replace('_', ' ').title()}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Current:** {rec['current_value']}")
                        st.write(f"**Recommended:** {rec['recommended_value']}")
                    with col2:
                        st.write(f"**Probability increase:** +{rec['probability_increase']:.1%}")
                        st.write(f"**New probability:** {rec['new_probability']:.1%}")
                    impact = rec['original_shap_impact']
                    if impact is not None:
                        st.write(f"**Impact:** This addresses the negative SHAP factor ({impact:.3f})")
            
            # Best recommendation
            best_rec = max(recommendations, key=lambda x: x['probability_increase'])
            st.info(f"🎯 **TOP RECOMMENDATION:** Change your {best_rec['parameter'].replace('_', ' ')} from {best_rec['current_value']} to {best_rec['recommended_value']}. This could increase your success probability by {best_rec['probability_increase']:.1%}")
            
            # Summary
            max_improvement = max(rec['probability_increase'] for rec in recommendations)
            st.success(f"**Potential Max Improvement:** +{max_improvement:.1%} (Achievable Probability: {original_prob + max_improvement:.1%})")
        else:
            st.success("No specific recommendations available - your current setup is already quite optimal!")

# --- STATISTICS PAGE ---
elif page == "Statistics":
    st.header("📊 Advanced Analytics & Insights")
    
    if not selected_tasks or len(selected_tasks) == 0:
        st.info("No data available yet. Add tasks in the Task Tracker page.")
        st.stop()
    
    df_stats = pd.DataFrame(selected_tasks)
    
    # Enhanced summary with better styling
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; margin: 20px 0;">
        <h2 style="color: white; text-align: center; margin: 0;">📈 Dataset Overview</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Key metrics in cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Tasks", len(df_stats), help="Total number of tasks in your dataset")
    
    with col2:
        completion_rate = df_stats['completed_on_time'].mean()
        st.metric("Overall Success Rate", f"{completion_rate:.1%}", 
                 delta=f"{completion_rate - 0.5:.1%}" if completion_rate > 0.5 else f"{completion_rate - 0.5:.1%}")
    
    with col3:
        avg_difficulty = df_stats['task_difficulty'].mean()
        st.metric("Average Difficulty", f"{avg_difficulty:.1f}/3", help="Average task difficulty level")
    
    with col4:
        avg_days_remaining = df_stats['days_remaining'].mean()
        st.metric("Avg Days Remaining", f"{avg_days_remaining:.1f} days", help="Average days until deadline")
    
    # Enhanced completion rate analysis
    st.subheader("🎯 Success Rate Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Completion rate by task type with enhanced styling
        task_type_success = df_stats.groupby('task_type')['completed_on_time'].agg(['mean', 'count']).reset_index()
        task_type_success.columns = ['Task Type', 'Success Rate', 'Count']
        
        fig1 = px.bar(task_type_success, x='Task Type', y='Success Rate',
                     color='Success Rate', color_continuous_scale='RdYlGn',
                     text='Success Rate', title='Success Rate by Task Type',
                     hover_data=['Count'])
        fig1.update_traces(texttemplate='%{text:.1%}', textposition='outside')
        fig1.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Completion rate by difficulty with trend line
        difficulty_success = df_stats.groupby('task_difficulty')['completed_on_time'].agg(['mean', 'count']).reset_index()
        difficulty_success.columns = ['Difficulty', 'Success Rate', 'Count']
        
        fig2 = px.scatter(difficulty_success, x='Difficulty', y='Success Rate', 
                         size='Count', color='Success Rate', color_continuous_scale='RdYlGn',
                         title='Success Rate by Task Difficulty',
                         hover_data=['Count'])
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)
    
    # Time-based analysis
    st.subheader("⏰ Time-Based Patterns")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Time of day analysis
        time_success = df_stats.groupby('time_of_day')['completed_on_time'].agg(['mean', 'count']).reset_index()
        time_success.columns = ['Time of Day', 'Success Rate', 'Count']
        
        fig3 = px.bar(time_success, x='Time of Day', y='Success Rate',
                     color='Success Rate', color_continuous_scale='RdYlGn',
                     text='Success Rate', title='Success Rate by Time of Day',
                     hover_data=['Count'])
        fig3.update_traces(texttemplate='%{text:.1%}', textposition='outside')
        fig3.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        # Day of week analysis
        day_success = df_stats.groupby('day_of_week')['completed_on_time'].agg(['mean', 'count']).reset_index()
        day_success.columns = ['Day of Week', 'Success Rate', 'Count']
        
        # Reorder days
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_success['Day of Week'] = pd.Categorical(day_success['Day of Week'], categories=day_order, ordered=True)
        day_success = day_success.sort_values('Day of Week')
        
        fig4 = px.line(day_success, x='Day of Week', y='Success Rate',
                      markers=True, title='Success Rate Trend by Day of Week',
                      hover_data=['Count'])
        fig4.update_layout(height=400)
        st.plotly_chart(fig4, use_container_width=True)
    
    # Advanced analytics
    st.subheader("🔍 Advanced Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Overdue tasks vs success rate
        overdue_success = df_stats.groupby('overdue_tasks')['completed_on_time'].agg(['mean', 'count']).reset_index()
        overdue_success.columns = ['Overdue Tasks', 'Success Rate', 'Count']
        overdue_success = overdue_success[overdue_success['Count'] >= 3]  # Filter for meaningful data
        
        if len(overdue_success) > 1:
            fig5 = px.scatter(overdue_success, x='Overdue Tasks', y='Success Rate',
                             size='Count', color='Success Rate', color_continuous_scale='RdYlGn',
                             title='Impact of Overdue Tasks on Success Rate',
                             hover_data=['Count'])
            fig5.update_layout(height=400)
            st.plotly_chart(fig5, use_container_width=True)
        else:
            st.info("Insufficient data to analyze overdue tasks impact")
    
    with col2:
        # Days remaining vs success rate
        days_success = df_stats.groupby('days_remaining')['completed_on_time'].agg(['mean', 'count']).reset_index()
        days_success.columns = ['Days Remaining', 'Success Rate', 'Count']
        days_success = days_success[days_success['Count'] >= 2]  # Filter for meaningful data
        
        if len(days_success) > 1:
            fig6 = px.scatter(days_success, x='Days Remaining', y='Success Rate',
                             size='Count', color='Success Rate', color_continuous_scale='RdYlGn',
                             title='Impact of Days Remaining on Success Rate',
                             hover_data=['Count'])
            fig6.update_layout(height=400)
            st.plotly_chart(fig6, use_container_width=True)
        else:
            st.info("Insufficient data to analyze days remaining impact")
    
    # Data quality and distribution
    st.subheader("📊 Data Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Task difficulty distribution
        fig8 = px.histogram(df_stats, x='task_difficulty', color='completed_on_time',
                           nbins=3, title='Task Difficulty Distribution by Success',
                           labels={'task_difficulty': 'Task Difficulty', 'completed_on_time': 'Completed'})
        fig8.update_layout(height=400)
        st.plotly_chart(fig8, use_container_width=True)
    
    with col2:
        # Days remaining distribution
        fig9 = px.histogram(df_stats, x='days_remaining', color='completed_on_time',
                           nbins=10, title='Days Remaining Distribution by Success',
                           labels={'days_remaining': 'Days Remaining', 'completed_on_time': 'Completed'})
        fig9.update_layout(height=400)
        st.plotly_chart(fig9, use_container_width=True)
    
    # Raw data with enhanced styling
    st.subheader("📋 Raw Data Table")
    with st.expander("View detailed data"):
        st.dataframe(df_stats, hide_index=True, use_container_width=True)

# --- MODEL INSIGHTS PAGE ---
elif page == "Model Insights":
    st.header("🤖 Model Performance & Insights")
    
    if not st.session_state.model_trained:
        st.warning("Model not trained yet. Please train the model in the Prediction page first.")
        st.stop()
    
    if not selected_tasks or len(selected_tasks) == 0:
        st.info("No data available. Add tasks in the Task Tracker page and train the model.")
        st.stop()
    
    # Model overview
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; margin: 20px 0;">
        <h2 style="color: white; text-align: center; margin: 0;">🧠 Ensemble Model Overview</h2>
        <p style="color: white; text-align: center; margin: 10px 0 0 0;">Advanced machine learning combining XGBoost and Random Forest</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Model performance comparison
    st.subheader("📊 Model Performance Comparison")
    
    evaluations = st.session_state.model_evaluations
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Ensemble Accuracy", f"{evaluations['ensemble']['accuracy']:.2%}")
        st.metric("Cross-Validation", f"{evaluations['ensemble']['cv_mean']:.2%} ± {evaluations['ensemble']['cv_std']:.2%}")
    
    with col2:
        st.metric("XGBoost Accuracy", f"{evaluations['xgb']['accuracy']:.2%}")
        st.metric("Cross-Validation", f"{evaluations['xgb']['cv_mean']:.2%} ± {evaluations['xgb']['cv_std']:.2%}")
    
    with col3:
        st.metric("Random Forest Accuracy", f"{evaluations['rf']['accuracy']:.2%}")
        st.metric("Cross-Validation", f"{evaluations['rf']['cv_mean']:.2%} ± {evaluations['rf']['cv_std']:.2%}")
    
    # Performance comparison chart
    model_names = ['Ensemble', 'XGBoost', 'Random Forest']
    accuracies = [evaluations['ensemble']['accuracy'], evaluations['xgb']['accuracy'], evaluations['rf']['accuracy']]
    cv_scores = [evaluations['ensemble']['cv_mean'], evaluations['xgb']['cv_mean'], evaluations['rf']['cv_mean']]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Training Accuracy', x=model_names, y=accuracies, marker_color='lightblue'))
    fig.add_trace(go.Bar(name='Cross-Validation', x=model_names, y=cv_scores, marker_color='darkblue'))
    fig.update_layout(title='Model Performance Comparison', barmode='group', height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Model architecture insights
    st.subheader("🏗️ Model Architecture Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Ensemble Model Details**")
        st.info("""
        **Model Composition:**
        - **XGBoost**: Gradient boosting with hyperparameter tuning
        - **Random Forest**: Ensemble of decision trees
        - **Voting**: Soft voting for final prediction
        
        **Feature Engineering:**
        - 10+ engineered features
        - Time-based patterns
        - Interaction effects
        - Categorical encoding
        """)
    
    with col2:
        st.markdown("**Training Process**")
        st.info("""
        **Hyperparameter Tuning:**
        - Grid search with cross-validation
        - Optimized for accuracy
        - 3-fold CV for tuning
        - 5-fold CV for evaluation
        
        **Feature Selection:**
        - Automatic feature importance
        - SHAP analysis for interpretability
        - Cross-validation stability
        """)
    
    # Model interpretability
    st.subheader("🔍 Model Interpretability")
    
    st.markdown("""
    **SHAP (SHapley Additive exPlanations) Analysis:**
    - Explains individual predictions
    - Shows feature contributions
    - Identifies negative factors
    - Enables personalized recommendations
    """)
    
    # Data quality metrics
    st.subheader("📈 Data Quality Metrics")
    
    df_insights = pd.DataFrame(selected_tasks)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Data Points", len(df_insights))
    
    with col2:
        missing_data = df_insights.isnull().sum().sum()
        st.metric("Missing Values", missing_data)
    
    with col3:
        feature_count = len(st.session_state.feature_cols)
        st.metric("Features Used", feature_count)
    
    with col4:
        class_balance = df_insights['completed_on_time'].value_counts(normalize=True)
        balance_ratio = min(class_balance) / max(class_balance)
        st.metric("Class Balance", f"{balance_ratio:.2f}")
    
    # Model recommendations
    st.subheader("💡 Model Recommendations")
    
    if len(selected_tasks) < 100:
        st.warning("**Data Quantity**: Consider collecting more data (aim for 100+ tasks) for better model performance.")
    
    if balance_ratio < 0.3:
        st.warning("**Class Imbalance**: Your dataset has imbalanced classes. Consider collecting more data for the minority class.")
    
    if missing_data > 0:
        st.warning("**Missing Data**: Some data points have missing values. Consider cleaning your dataset.")
    
    st.success("**Model Status**: Your ensemble model is ready for predictions with enhanced accuracy and interpretability!")

    # Calculate and display accuracy, precision, recall, and F1 score for each model
    if selected_tasks and st.session_state.model_trained:
        df_eval = pd.DataFrame(selected_tasks)
        y_true = df_eval['completed_on_time']
        # Re-apply feature engineering and encoding
        df_eval_fe = create_advanced_features(df_eval)
        df_eval_encoded = pd.get_dummies(df_eval_fe, columns=['task_type', 'time_of_day'])
        for col in st.session_state.feature_cols:
            if col not in df_eval_encoded.columns:
                df_eval_encoded[col] = 0
        X_eval = df_eval_encoded[st.session_state.feature_cols]
        models = {
            'Ensemble': st.session_state.model,
            'XGBoost': st.session_state.xgb_model,
            'Random Forest': st.session_state.rf_model
        }
        metrics_table = []
        for name, model in models.items():
            y_pred = model.predict(X_eval)
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            metrics_table.append({
                'Model': name,
                'Accuracy': f"{acc:.2%}",
                'Precision': f"{prec:.2%}",
                'Recall': f"{rec:.2%}",
                'F1 Score': f"{f1:.2%}"
            })
        st.subheader("🔎 Classification Metrics")
        st.table(metrics_table)

# Footer
st.markdown("---")
st.markdown("*Student Success Prediction System - Upload your data, train the model, and get personalized insights!*")

