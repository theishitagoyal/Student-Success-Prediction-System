import streamlit as st
from supabase import create_client
from config import SUPABASE_URL, SUPABASE_KEY

st.title("🔧 Supabase Connection Test")

st.write("**Testing your Supabase connection...**")

# Check if credentials are still placeholder values
if SUPABASE_URL == 'YOUR_SUPABASE_URL' or SUPABASE_KEY == 'YOUR_SUPABASE_ANON_KEY':
    st.error("❌ You still have placeholder values in config.py!")
    st.write("Please update your `config.py` file with your real Supabase credentials.")
    st.stop()

try:
    # Try to create Supabase client
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    st.success("✅ Successfully connected to Supabase!")
    
    # Test if users table exists
    try:
        result = supabase.table('users').select('count').limit(1).execute()
        st.success("✅ Users table exists and is accessible!")
    except Exception as e:
        st.warning("⚠️ Users table might not exist yet.")
        st.write("You need to run the SQL script in your Supabase SQL Editor:")
        st.code("""
CREATE TABLE IF NOT EXISTS users (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    username VARCHAR UNIQUE NOT NULL,
    email VARCHAR UNIQUE NOT NULL,
    password_hash VARCHAR NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_login TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT TRUE
);
        """)
    
    st.write(f"**Project URL:** {SUPABASE_URL}")
    st.write(f"**API Key:** {SUPABASE_KEY[:20]}...{SUPABASE_KEY[-10:]}")
    
except Exception as e:
    st.error(f"❌ Connection failed: {str(e)}")
    st.write("**Common issues:**")
    st.write("1. Check if your Project URL is correct")
    st.write("2. Make sure you're using the 'anon public' key, not 'service_role'")
    st.write("3. Verify your Supabase project is active") 