# Supabase Setup Guide for Student Success Predictor

This guide will help you set up Supabase for the authentication system in your Streamlit app.

## Step 1: Create a Supabase Account

1. Go to [https://supabase.com](https://supabase.com)
2. Click "Start your project" and sign up
3. Create a new organization (if needed)
4. Create a new project

## Step 2: Get Your Supabase Credentials

1. In your Supabase dashboard, go to **Settings** → **API**
2. Copy the following values:
   - **Project URL** (looks like: `https://your-project-id.supabase.co`)
   - **anon public** key (starts with `eyJ...`)

## Step 3: Configure Your App

1. Open `config.py` in your project
2. Replace the placeholder values:

```python
# Replace these with your actual Supabase credentials
SUPABASE_URL = "https://your-project-id.supabase.co"
SUPABASE_KEY = "your-anon-key-here"
```

## Step 4: Create the Users Table

1. In your Supabase dashboard, go to **SQL Editor**
2. Create a new query and paste this SQL:

```sql
-- Create users table
CREATE TABLE IF NOT EXISTS users (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    username VARCHAR UNIQUE NOT NULL,
    email VARCHAR UNIQUE NOT NULL,
    password_hash VARCHAR NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_login TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT TRUE
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);

-- Enable Row Level Security (RLS)
ALTER TABLE users ENABLE ROW LEVEL SECURITY;

-- Create policy to allow all operations (for now)
CREATE POLICY "Allow all operations" ON users FOR ALL USING (true);
```

3. Click **Run** to execute the SQL

## Step 5: Test the Setup

1. Run your Streamlit app:
   ```bash
   streamlit run Student_task_success_predictor_streamlit.py
   ```

2. Try registering a new user
3. Try logging in with the registered user

## Step 6: Environment Variables (Optional but Recommended)

For better security, you can use environment variables:

1. Create a `.env` file in your project root:
   ```
   SUPABASE_URL=https://your-project-id.supabase.co
   SUPABASE_KEY=your-anon-key-here
   ```

2. Install python-dotenv:
   ```bash
   pip install python-dotenv
   ```

3. Update `config.py`:
   ```python
   from dotenv import load_dotenv
   load_dotenv()
   
   SUPABASE_URL = os.getenv('SUPABASE_URL', 'YOUR_SUPABASE_URL')
   SUPABASE_KEY = os.getenv('SUPABASE_KEY', 'YOUR_SUPABASE_ANON_KEY')
   ```

## Troubleshooting

### Common Issues:

1. **"Database connection failed"**
   - Check your Supabase URL and key
   - Make sure your project is active

2. **"Table does not exist"**
   - Run the SQL script in Step 4
   - Check if the table was created in the **Table Editor**

3. **"Invalid credentials"**
   - Make sure you're using the correct username/password
   - Check if the user was created successfully

4. **"Username already exists"**
   - Try a different username
   - Check the users table in Supabase dashboard

### Security Notes:

- The current implementation uses SHA-256 for password hashing
- For production, consider using bcrypt or Argon2
- Enable Row Level Security (RLS) policies for better security
- Use environment variables for sensitive data

## Next Steps

Once authentication is working, you can:

1. Add user roles (admin, teacher, student)
2. Implement password reset functionality
3. Add email verification
4. Create user profiles
5. Add activity logging

## Support

If you encounter issues:
1. Check the Supabase documentation: [https://supabase.com/docs](https://supabase.com/docs)
2. Check the Streamlit documentation: [https://docs.streamlit.io](https://docs.streamlit.io)
3. Review the error messages in your Streamlit app 