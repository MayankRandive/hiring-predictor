import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from xgboost import plot_importance

# ------------------------
# Load Model + Scaler + Features
# ------------------------
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
features = pickle.load(open("features.pkl", "rb"))

# ------------------------
# Page Config
# ------------------------
st.set_page_config(
    page_title="AI Candidate Evaluation System",
    page_icon="🤖",
    layout="wide"
)

# ------------------------
# Header
# ------------------------
st.title("🤖 AI Candidate Evaluation System")
st.markdown("""
Smart hiring assistant powered by Machine Learning.

Evaluate candidates based on:
- Skills
- Experience
- Education
""")

# ------------------------
# Sidebar Inputs
# ------------------------
st.sidebar.header("📋 Candidate Details")

skills_score = st.sidebar.slider("Skills Score", 0, 100, 50)
experience_years = st.sidebar.number_input("Years of Experience", 0, 50, 1)
education_level = st.sidebar.selectbox(
    "Education Level",
    ["High School", "Bachelor", "Master", "PhD"]
)

# ------------------------
# Default Values
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
# Layout
# ------------------------
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📊 Candidate Summary")
    st.write(f"**Skills Score:** {skills_score}")
    st.write(f"**Experience:** {experience_years} years")
    st.write(f"**Education:** {education_level}")

with col2:
    st.subheader("🚀 Prediction")

    if st.button("Evaluate Candidate"):
        try:
            # ✅ IMPORTANT PART (THIS IS WHAT YOU ASKED)
            candidate_df = candidate_df[features]          # correct order
            candidate_scaled = scaler.transform(candidate_df)  # scaling
            prob = model.predict_proba(candidate_scaled)[:, 1][0]

            threshold = 0.5
            prediction = "Hired" if prob > threshold else "Not Hired"

            # Result
            if prediction == "Hired":
                st.success(f"✅ {prediction}")
            else:
                st.error(f"❌ {prediction}")

            st.write(f"Confidence Score: {prob:.2f}")
            st.progress(float(prob))

        except Exception as e:
            st.error(f"Error: {e}")

# ------------------------
# Feature Importance
# ------------------------
st.subheader("📌 Model Insights")

fig, ax = plt.subplots(figsize=(10, 5))
plot_importance(model, ax=ax, max_num_features=10, show_values=False)
ax.set_title("Top Features")

st.pyplot(fig)