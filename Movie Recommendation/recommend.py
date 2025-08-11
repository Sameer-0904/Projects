import joblib
import logging
import requests
import io

# ====== CONFIG ======
DF_URL = "https://huggingface.co/datasets/Sameer0904/movie-recommendation/blob/main/df_cleaned.pkl"
COSINE_URL = "https://huggingface.co/datasets/Sameer0904/movie-recommendation/blob/main/cosine_sim.pkl"
# ====================

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("recommend.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

def load_joblib_from_url(url):
    logging.info("📥 Downloading: %s", url)
    response = requests.get(url)
    response.raise_for_status()
    return joblib.load(io.BytesIO(response.content))

logging.info("🔁 Loading data from URLs...")
try:
    df = load_joblib_from_url(DF_URL)
    cosine_sim = load_joblib_from_url(COSINE_URL)
    logging.info("✅ Data loaded successfully from links.")
except Exception as e:
    logging.error("❌ Failed to load required files: %s", str(e))
    raise e

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
    logging.info("✅ Top %d recommendations ready.", top_n)
    result_df = df[['title']].iloc[movie_indices].reset_index(drop=True)
    result_df.index = result_df.index + 1
    result_df.index.name = "S.No."
    return result_df
