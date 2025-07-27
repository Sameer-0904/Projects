import joblib
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("recommend.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logging.info("🔁 Loading cleaned DataFrame...")
try:
    df = joblib.load('df_cleaned.pkl')
except Exception as e:
    logging.error("❌ Failed to load df_cleaned.pkl: %s", str(e))
    raise e

logging.info("🔠 Recomputing TF-IDF and cosine similarity...")
tfidf = TfidfVectorizer(max_features=5000)
tfidf_matrix = tfidf.fit_transform(df['cleaned_text'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
logging.info("✅ Cosine similarity computed.")

def recommend_movies(movie_name, top_n=5):
    logging.info("🎬 Recommending movies for: '%s'", movie_name)
    idx = df[df['title'].str.lower() == movie_name.lower()].index
    if len(idx) == 0:
        logging.warning("⚠️ Movie not found in dataset.")
        return None
    idx = idx[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n + 1]
    movie_indices = [i[0] for i in sim_scores]
    result_df = df[['title']].iloc[movie_indices].reset_index(drop=True)
    result_df.index = result_df.index + 1
    result_df.index.name = "S.No."
    return result_df
