import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from xgboost import plot_importance

# ------------------------
# Load trained model
# ------------------------
with open('xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.set_page_config(page_title="AI Candidate Evaluation System", layout="wide")
st.title("AI Candidate Evaluation System")
st.write("Predict whether a candidate should be Hired or Not Hired based on skills, experience, and education.")
st.markdown("""
### AI Candidate Evaluation System
This tool predicts hiring decisions based on:
- Skills Score
- Experience
- Education Level
Built using Machine Learning (XGBoost)
""")

# ------------------------
# User Input
# ------------------------
st.header("Enter Candidate Details")

skills_score = st.slider("Skills Score", 0, 100, 50)
experience_years = st.number_input("Years of Experience", 0, 50, 1)
education_level = st.selectbox("Education Level", ["High School", "Bachelor", "Master", "PhD"])

# ------------------------
# Default values
# ------------------------
default_values = {
    'age': 25,
    'cgpa': 8.5,
    'internships': 1,
    'projects': 3,
    'programming_languages': 3,
    'certifications': 2,
    'hackathons': 1,
    'research_papers': 0,
    'soft_skills_score': 70,
    'resume_length_words': 800,
    'university_tier_Tier 2': 0,
    'university_tier_Tier 3': 0,
    'company_type_Mid-size': 0,
    'company_type_Startup': 0
}

# ------------------------
# Encoding
# ------------------------
education_level_Masters = 1 if education_level == "Master" else 0
education_level_PhD = 1 if education_level == "PhD" else 0

# ------------------------
# DataFrame
# ------------------------
candidate_df = pd.DataFrame({
    'age': [default_values['age']],
    'cgpa': [default_values['cgpa']],
    'internships': [default_values['internships']],
    'projects': [default_values['projects']],
    'programming_languages': [default_values['programming_languages']],
    'certifications': [default_values['certifications']],
    'experience_years': [experience_years],
    'hackathons': [default_values['hackathons']],
    'research_papers': [default_values['research_papers']],
    'skills_score': [skills_score],
    'soft_skills_score': [default_values['soft_skills_score']],
    'resume_length_words': [default_values['resume_length_words']],
    'education_level_Masters': [education_level_Masters],
    'education_level_PhD': [education_level_PhD],
    'university_tier_Tier 2': [default_values['university_tier_Tier 2']],
    'university_tier_Tier 3': [default_values['university_tier_Tier 3']],
    'company_type_Mid-size': [default_values['company_type_Mid-size']],
    'company_type_Startup': [default_values['company_type_Startup']]
})

# ------------------------
# Prediction
# ------------------------
if st.button("Predict"):
    prob = model.predict_proba(candidate_df)[:, 1][0]
    threshold = 0.5
    prediction = "Hired" if prob > threshold else "Not Hired"

    st.subheader("Prediction Results")
    st.write(f"Prediction: **{prediction}**")
    st.write(f"Confidence: **{prob:.2f}**")

    # ------------------------
    # Feature Importance
    # ------------------------
    st.subheader("Top Feature Importance")
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_importance(model, ax=ax, max_num_features=10, show_values=False)
    ax.set_title("Top 10 Features by Importance")
    st.pyplot(fig)