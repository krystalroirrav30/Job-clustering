import joblib

def save_model(model, vectorizer, model_path="models/job_cluster_model.pkl", vec_path="models/tfidf_vectorizer.pkl"):
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vec_path)

def load_model_and_vectorizer(model_path="models/job_cluster_model.pkl", vec_path="models/tfidf_vectorizer.pkl"):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)
    return model, vectorizer

def print_top_keywords_per_cluster(model, vectorizer, n_terms=10):
    try:
        feature_names = vectorizer.get_feature_names_out()
    except AttributeError:
        feature_names = vectorizer.get_feature_names()

    for i, centroid in enumerate(model.cluster_centers_):
        print(f"Cluster {i}:")
        top_indices = centroid.argsort()[-n_terms:][::-1]
        top_terms = [feature_names[ind] for ind in top_indices]
        print(", ".join(top_terms))
        print()
