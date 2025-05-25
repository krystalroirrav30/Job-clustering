from datetime import datetime
import pandas as pd
import time
from sklearn.preprocessing import normalize
import joblib
from scraper import scrape_karkidi_jobs
from model_utils import load_model_and_vectorizer

def preprocess_and_predict(df, model, vectorizer):
    df = df.copy()
    df["Skills"] = df["Skills"].fillna("").str.lower()
    X = vectorizer.transform(df["Skills"])
    X = normalize(X)
    df["Cluster"] = model.predict(X)
    return df

def daily_scrape_and_cluster(pages=3):
    print("Starting scrape and cluster at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    df_jobs = scrape_karkidi_jobs(pages=pages)

    if df_jobs.empty:
        print("No jobs found.")
        return

    model, vectorizer = load_model_and_vectorizer()
    df_clustered = preprocess_and_predict(df_jobs, model, vectorizer)
    filename = f"data/clustered_jobs_{datetime.today().strftime('%Y-%m-%d')}.csv"
    df_clustered.to_csv(filename, index=False)
    print(f"Scrape + cluster complete. Saved to {filename}")

if __name__ == "__main__":
    daily_scrape_and_cluster()
