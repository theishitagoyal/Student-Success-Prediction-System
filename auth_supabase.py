import streamlit as st
from supabase import create_client
import hashlib
import os
from datetime import datetime
from config import SUPABASE_URL, SUPABASE_KEY, SESSION_KEYS, PASSWORD_MIN_LENGTH, USERNAME_MIN_LENGTH

# Initialize Supabase client
try:
    if SUPABASE_URL != 'YOUR_SUPABASE_URL' and SUPABASE_KEY != 'YOUR_SUPABASE_ANON_KEY':
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    else:
        st.warning("⚠️ Please configure your Supabase credentials in config.py")
        supabase = None
except Exception as e:
    st.error(f"Failed to connect to Supabase: {str(e)}")
    supabase = None

def hash_password(password):
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def create_users_table():
    """Create users table in Supabase if it doesn't exist"""
    if not supabase:
        return False
    
    try:
        # This will be handled by Supabase SQL editor
        # You'll need to run this SQL in your Supabase dashboard:
        """
        CREATE TABLE IF NOT EXISTS users (
            id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
            username VARCHAR UNIQUE NOT NULL,
            email VARCHAR UNIQUE NOT NULL,
            password_hash VARCHAR NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            last_login TIMESTAMP WITH TIME ZONE,
            is_active BOOLEAN DEFAULT TRUE
        );
        """
        return True
    except Exception as e:
        st.error(f"Error creating users table: {str(e)}")
        return False

def register_user(username, email, password):
    """Register a new user"""
    if not supabase:
        return False, "Database connection failed"
    
    try:
        # Check if user already exists
        existing_user = supabase.table('users').select('username').eq('username', username).execute()
        if existing_user.data:
            return False, "Username already exists"
        
        existing_email = supabase.table('users').select('email').eq('email', email).execute()
        if existing_email.data:
            return False, "Email already registered"
        
        # Hash password and create user
        password_hash = hash_password(password)
        
        user_data = {
            'username': username,
            'email': email,
            'password_hash': password_hash,
            'created_at': datetime.now().isoformat(),
            'is_active': True
        }
        
        result = supabase.table('users').insert(user_data).execute()
        
        if result.data:
            return True, "Registration successful!"
        else:
            return False, "Registration failed"
            
    except Exception as e:
        return False, f"Registration error: {str(e)}"

def verify_user(username, password):
    """Verify user credentials"""
    if not supabase:
        return False, "Database connection failed"
    
    try:
        password_hash = hash_password(password)
        
        result = supabase.table('users').select('*').eq('username', username).eq('password_hash', password_hash).execute()
        
        if result.data:
            user = result.data[0]
            # Update last login
            supabase.table('users').update({'last_login': datetime.now().isoformat()}).eq('id', user['id']).execute()
            return True, user
        else:
            return False, "Invalid credentials"
            
    except Exception as e:
        return False, f"Login error: {str(e)}"

def get_user_by_id(user_id):
    """Get user by ID"""
    if not supabase:
        return None
    
    try:
        result = supabase.table('users').select('*').eq('id', user_id).execute()
        return result.data[0] if result.data else None
    except Exception as e:
        st.error(f"Error fetching user: {str(e)}")
        return None

def update_user_profile(user_id, **kwargs):
    """Update user profile"""
    if not supabase:
        return False
    
    try:
        result = supabase.table('users').update(kwargs).eq('id', user_id).execute()
        return bool(result.data)
    except Exception as e:
        st.error(f"Error updating profile: {str(e)}")
        return False

def auth_ui():
    """Main authentication UI centered on the main page"""
    # Check if user is already logged in
    if 'user_id' in st.session_state and st.session_state['user_id']:
        user = get_user_by_id(st.session_state['user_id'])
        if user:
            st.sidebar.success(f"Welcome, {user['username']}!")
            if st.sidebar.button("Logout"):
                for key in ['user_id', 'username', 'email']:
                    if key in st.session_state:
                        del st.session_state[key]
                # Clear uploaded data/session-only dataset on logout
                for key in ['uploaded_df', 'data_source']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
            return True

    # Centered login/register UI
    st.markdown(
        "<h1 style='text-align: center; margin-bottom: 2rem;'>🎓 Student Success Predictor</h1>",
        unsafe_allow_html=True
    )
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown('<div class="centered-card">', unsafe_allow_html=True)
        st.markdown('<h2 class="centered-title">🔐 Login or Register</h2>', unsafe_allow_html=True)
        tab1, tab2 = st.tabs(["Login", "Register"])
        with tab1:
            login_username = st.text_input("Username", key="login_username")
            login_password = st.text_input("Password", type="password", key="login_password")
            if st.button("Login", key="login_btn"):
                if login_username and login_password:
                    success, result = verify_user(login_username, login_password)
                    if success:
                        # Clear uploaded data/session-only dataset on login
                        for key in ['uploaded_df', 'data_source']:
                            if key in st.session_state:
                                del st.session_state[key]
                        st.session_state['user_id'] = result['id']
                        st.session_state['username'] = result['username']
                        st.session_state['email'] = result['email']
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error(result)
                else:
                    st.warning("Please enter both username and password")
        with tab2:
            reg_username = st.text_input("Username", key="reg_username")
            reg_email = st.text_input("Email", key="reg_email")
            reg_password = st.text_input("Password", type="password", key="reg_password")
            reg_confirm_password = st.text_input("Confirm Password", type="password", key="reg_confirm_password")
            if st.button("Register", key="register_btn"):
                if reg_username and reg_email and reg_password and reg_confirm_password:
                    if reg_password != reg_confirm_password:
                        st.error("Passwords do not match")
                    elif len(reg_password) < PASSWORD_MIN_LENGTH:
                        st.error(f"Password must be at least {PASSWORD_MIN_LENGTH} characters")
                    elif len(reg_username) < USERNAME_MIN_LENGTH:
                        st.error(f"Username must be at least {USERNAME_MIN_LENGTH} characters")
                    else:
                        success, message = register_user(reg_username, reg_email, reg_password)
                        if success:
                            st.success(message)
                        else:
                            st.error(message)
                else:
                    st.warning("Please fill in all fields")
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("""
        <style>
        .stTabs [data-baseweb="tab-list"] {
            justify-content: center;
        }
        </style>
    """, unsafe_allow_html=True)
    return False

def require_auth():
    """Decorator to require authentication for specific pages/functions"""
    if 'user_id' not in st.session_state or not st.session_state['user_id']:
        st.error("Please login to access this feature")
        st.stop()
    return True

def get_current_user():
    """Get current logged-in user data"""
    if 'user_id' in st.session_state and st.session_state['user_id']:
        return get_user_by_id(st.session_state['user_id'])
    return None 