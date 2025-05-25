import streamlit as st
import pandas as pd
from scraper import scrape_karkidi_jobs
from model_utils import load_model_and_vectorizer
from sklearn.preprocessing import normalize

@st.cache_resource
def get_model_and_vectorizer():
    return load_model_and_vectorizer()

def classify_new_jobs(df, model, vectorizer):
    df = df.copy()
    df["Skills"] = df["Skills"].fillna("").str.lower()
    X = vectorizer.transform(df["Skills"])
    X = normalize(X)
    df["Cluster"] = model.predict(X)
    return df

st.set_page_config(page_title="Job Notifier", layout="wide")
st.title("💼 Job Notifier App (Karkidi.com)")
st.markdown("Search for latest jobs based on your interests.")

with st.sidebar:
    st.header("🎯 Your Preferences")
    keyword = st.text_input("Enter job keyword", "data science")
    pages = st.slider("Pages to scrape", 1, 3, 1)
    user_clusters = st.multiselect("Select preferred clusters", [0, 1, 2, 3, 4], default=[0, 4])
    run_button = st.button("🔍 Fetch Jobs")

if run_button:
    with st.spinner("Scraping new jobs..."):
        jobs_df = scrape_karkidi_jobs(keyword=keyword, pages=pages)

    if jobs_df.empty:
        st.warning("No jobs found.")
    else:
        model, vectorizer = get_model_and_vectorizer()
        classified_df = classify_new_jobs(jobs_df, model, vectorizer)
        matched_df = classified_df[classified_df["Cluster"].isin(user_clusters)]

        if not matched_df.empty:
            st.success(f"🎯 Found {len(matched_df)} matching jobs!")
            st.dataframe(matched_df[["Title", "Company", "Location", "Skills", "Cluster"]])
            csv = matched_df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Matches", data=csv, file_name="matched_jobs.csv", mime="text/csv")
        else:
            st.info("✅ No matching jobs in selected clusters.")
