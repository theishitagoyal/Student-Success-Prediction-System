# 🎓 Student Success Prediction System with Authentication

A Streamlit-based machine learning application that predicts student task completion success with secure user authentication using Supabase.

## ✨ Features

- **🔐 Secure Authentication**: User registration and login with Supabase
- **🤖 Machine Learning**: Ensemble of XGBoost and Random Forest for task completion prediction, with SHAP explanations for interpretability
- **📊 Data Analysis**: Interactive visualizations and insights
- **🎯 Personalized Recommendations**: AI-powered suggestions to improve success probability
- **📈 Real-time Predictions**: Instant analysis of task completion chances
- **💡 SHAP Explanations**: Understand what factors influence predictions

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install streamlit xgboost scikit-learn pandas shap matplotlib plotly google-generativeai supabase python-dotenv
```

### 2. Set Up Supabase

1. Go to [https://supabase.com](https://supabase.com) and create an account
2. Create a new project
3. Get your credentials from **Settings** → **API**

### 3. Configure Environment Variables

Create a `.env` file in your project root with the following content:

```env
GEMINI_API_KEY=your-gemini-api-key-here
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_KEY=your-anon-key-here
```

> **Note:** Never commit your `.env` file to version control. `.env` is already in `.gitignore`.

### 4. Create Database Table

In your Supabase SQL Editor, run:

```sql
CREATE TABLE IF NOT EXISTS users (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    username VARCHAR UNIQUE NOT NULL,
    email VARCHAR UNIQUE NOT NULL,
    password_hash VARCHAR NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_login TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT TRUE
);

CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Allow all operations" ON users FOR ALL USING (true);
```

### 5. Run the Application

```bash
# Run the version with authentication
streamlit run app_with_auth.py

# Or run the original version without authentication
streamlit run Student_task_success_predictor_streamlit.py
```

## 📁 File Structure

```
aiml/
├── app_with_auth.py                    # Main app with authentication
├── Student_task_success_predictor_streamlit.py  # Original app
├── auth_supabase.py                    # Authentication module
├── config.py                           # Configuration settings (loads from .env and st.secrets)
├── SUPABASE_SETUP.md                   # Detailed Supabase setup guide
├── test_supabase.py                    # Supabase connection and table test
└── README.md                           # This file
```

## 🔧 How to Use

### 1. Authentication
- **Register**: Create a new account with username, email, and password
- **Login**: Use your credentials to access the application
- **Logout**: Click the logout button in the sidebar

### 2. Data Upload
- Upload a CSV file with the following columns:
  - `overdue_tasks`: Number of overdue tasks
  - `task_difficulty`: Difficulty level (1-5)
  - `days_remaining`: Days until deadline
  - `day_of_week`: Day of the week
  - `task_type`: Type of task (e.g., "Assignment", "Project")
  - `time_of_day`: Time of day (e.g., "Morning", "Afternoon")
  - `completed_on_time`: Target variable (1 for completed, 0 for not)

### 3. Model Training
- The app automatically trains an XGBoost model when you upload data
- View model accuracy and data analysis visualizations

### 4. Making Predictions
- Input task parameters (overdue tasks, difficulty, etc.)
- Get instant predictions and analysis
- View factors affecting success probability
- Receive personalized recommendations

## 📊 Sample Data Format

```csv
overdue_tasks,task_difficulty,days_remaining,day_of_week,task_type,time_of_day,completed_on_time
2,3,5,Monday,Assignment,Morning,1
1,4,3,Wednesday,Project,Afternoon,0
3,2,7,Friday,Quiz,Evening,1
```

## 🔒 Security Features

- **Password Hashing**: SHA-256 encryption for stored passwords
- **Session Management**: Secure user sessions
- **Input Validation**: Comprehensive form validation
- **SQL Injection Protection**: Parameterized queries

## 🛠️ Configuration

### Environment Variables and Secrets

- All secrets (Gemini API key, Supabase URL, Supabase anon key) are loaded from a `.env` file using [python-dotenv](https://pypi.org/project/python-dotenv/) and accessed via Streamlit's `st.secrets`.
- Do **not** hardcode secrets in your code.
- Example `.env`:

```env
GEMINI_API_KEY=your-gemini-api-key-here
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_KEY=your-anon-key-here
```

### Session State Keys (for Developers)

The app uses Streamlit's session state to manage user and app state. Main keys include:
- `model_trained`, `model`, `explainer`, `feature_cols`, `completion_rate_dicts`, `df`, `model_gemini`, `show_predictions`, `show_factors`, `show_recommendations`, `prediction_results`, `data_source`, `uploaded_df`, `upload_action`, `user_id`, `username`, `email`

### Customization

- **Password Requirements**: Modify `PASSWORD_MIN_LENGTH` in `config.py`
- **Username Requirements**: Modify `USERNAME_MIN_LENGTH` in `config.py`
- **API Keys**: Set Gemini API key in your `.env` file

## 🐛 Troubleshooting

### Common Issues

1. **"Database connection failed"**
   - Check your Supabase URL and key in your `.env` file
   - Ensure your Supabase project is active

2. **"Table does not exist"**
   - Run the SQL script in your Supabase SQL Editor
   - Check the Table Editor to confirm the table was created

3. **"Invalid credentials"**
   - Verify username and password
   - Check if the user was created successfully

4. **"Username already exists"**
   - Try a different username
   - Check the users table in Supabase dashboard

### Getting Help

1. Check the [Supabase Setup Guide](SUPABASE_SETUP.md)
2. Review error messages in the Streamlit app
3. Check the [Supabase documentation](https://supabase.com/docs)
4. Check the [Streamlit documentation](https://docs.streamlit.io)

## 🚀 Deployment

### Local Development
```bash
streamlit run app_with_auth.py
```

### Cloud Deployment
- **Streamlit Cloud**: Connect your GitHub repository
- **Heroku**: Use the provided requirements.txt
- **AWS/GCP**: Deploy as a containerized application

## 📈 Future Enhancements

- [ ] User roles (admin, teacher, student)
- [ ] Password reset functionality
- [ ] Email verification
- [ ] User profiles and preferences
- [ ] Activity logging and analytics
- [ ] Advanced ML models
- [ ] Real-time collaboration features

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io) for the web framework
- [Supabase](https://supabase.com) for the backend database
- [XGBoost](https://xgboost.readthedocs.io) for machine learning
- [SHAP](https://shap.readthedocs.io) for model explanations
- [Google Gemini](https://ai.google.dev) for AI explanations

---

**Made with ❤️ for educational success prediction**

## 🧪 Testing Supabase Connection

To verify your Supabase credentials and table setup, run:

```bash
streamlit run test_supabase.py
```

This script checks your connection and guides you if the `users` table is missing.

## 🗄️ Database Tables

### users Table
(see below for schema)

### user_tasks Table
The app stores user tasks in a `user_tasks` table. Example schema:

```sql
CREATE TABLE IF NOT EXISTS user_tasks (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    overdue_tasks INTEGER,
    task_difficulty INTEGER,
    days_remaining INTEGER,
    day_of_week VARCHAR,
    task_type VARCHAR,
    time_of_day VARCHAR,
    completed_on_time INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

> For full setup, see [SUPABASE_SETUP.md](SUPABASE_SETUP.md). 