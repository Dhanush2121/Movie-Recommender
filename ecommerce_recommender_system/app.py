import streamlit as st
from recommendation_engine import hybrid_recommendation
from utils.data_loader import load_data

st.set_page_config(page_title="E-commerce Recommender", layout="wide")
st.title("🛍️ Hybrid Recommender System")

# Load data
movies, ratings = load_data()

# Sidebar
user_id = st.sidebar.selectbox("Select User ID", ratings['userId'].unique())
st.sidebar.write("Rate some products below to get better recommendations.")

# Recommend
if st.button("Get Recommendations"):
    recommendations = hybrid_recommendation(user_id, movies, ratings)
    st.subheader("📦 Recommended Products for You:")
    for title in recommendations:
        st.markdown(f"✅ {title}")
