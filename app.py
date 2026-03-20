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

Evaluate candidates based on multiple factors like:
- Education
- Skills
- Experience
- Projects & Certifications
""")

# ------------------------
# Sidebar Inputs (FULL)
# ------------------------
st.sidebar.header("📋 Candidate Details")

age = st.sidebar.number_input("Age", 18, 60, 25)
cgpa = st.sidebar.slider("CGPA", 0.0, 10.0, 8.0)

internships = st.sidebar.number_input("Internships", 0, 10, 1)
projects = st.sidebar.number_input("Projects", 0, 20, 3)
programming_languages = st.sidebar.number_input("Programming Languages", 0, 10, 3)
certifications = st.sidebar.number_input("Certifications", 0, 10, 2)

experience_years = st.sidebar.number_input("Experience (Years)", 0, 50, 1)

hackathons = st.sidebar.number_input("Hackathons", 0, 10, 1)
research_papers = st.sidebar.number_input("Research Papers", 0, 10, 0)

skills_score = st.sidebar.slider("Technical Skills Score", 0, 100, 50)
soft_skills_score = st.sidebar.slider("Soft Skills Score", 0, 100, 70)

resume_length_words = st.sidebar.number_input("Resume Length (Words)", 100, 2000, 800)

education_level = st.sidebar.selectbox(
    "Education Level",
    ["Bachelor", "Master", "PhD"]
)

university_tier = st.sidebar.selectbox(
    "University Tier",
    ["Tier 1", "Tier 2", "Tier 3"]
)

company_type = st.sidebar.selectbox(
    "Target Company Type",
    ["Product-Based", "Mid-size", "Startup"]
)

# ------------------------
# Encoding
# ------------------------
education_level_Masters = 1 if education_level == "Master" else 0
education_level_PhD = 1 if education_level == "PhD" else 0

university_tier_Tier_2 = 1 if university_tier == "Tier 2" else 0
university_tier_Tier_3 = 1 if university_tier == "Tier 3" else 0

company_type_Mid_size = 1 if company_type == "Mid-size" else 0
company_type_Startup = 1 if company_type == "Startup" else 0

# ------------------------
# DataFrame (FULL INPUT)
# ------------------------
candidate_df = pd.DataFrame({
    'age': [age],
    'cgpa': [cgpa],
    'internships': [internships],
    'projects': [projects],
    'programming_languages': [programming_languages],
    'certifications': [certifications],
    'experience_years': [experience_years],
    'hackathons': [hackathons],
    'research_papers': [research_papers],
    'skills_score': [skills_score],
    'soft_skills_score': [soft_skills_score],
    'resume_length_words': [resume_length_words],
    'education_level_Masters': [education_level_Masters],
    'education_level_PhD': [education_level_PhD],
    'university_tier_Tier 2': [university_tier_Tier_2],
    'university_tier_Tier 3': [university_tier_Tier_3],
    'company_type_Mid-size': [company_type_Mid_size],
    'company_type_Startup': [company_type_Startup]
})

# ------------------------
# Layout
# ------------------------
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📊 Candidate Summary")
    st.write(candidate_df)

with col2:
    st.subheader("🚀 Prediction")

    if st.button("Evaluate Candidate"):
        try:
            # Ensure correct order
            candidate_df = candidate_df[features]

            # Scale
            candidate_scaled = scaler.transform(candidate_df)

            # Predict
            prob = model.predict_proba(candidate_scaled)[:, 1][0]

            prediction = "Hired" if prob > 0.5 else "Not Hired"

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