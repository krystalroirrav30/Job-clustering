from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans

def preprocess_skills(df):
    df = df.copy()
    df["Skills"] = df["Skills"].fillna("").str.lower()
    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\\b\\w+\\b", stop_words="english")
    X = vectorizer.fit_transform(df["Skills"])
    X = normalize(X)
    return X, vectorizer

def cluster_skills(X, n_clusters=8):
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    model.fit(X)
    return model
