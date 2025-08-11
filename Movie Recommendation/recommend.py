import os
import joblib
import logging
import requests

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("recommend.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

BASE_DIR = os.path.dirname(__file__)

def download_file(url, filename):
    """Download file from URL if not already present."""
    if not os.path.exists(filename):
        logging.info(f"⬇️ Downloading {filename} ...")
        r = requests.get(url)
        r.raise_for_status()
        with open(filename, 'wb') as f:
            f.write(r.content)
        logging.info(f"✅ Download complete: {filename}")
    else:
        logging.info(f"📂 Found cached file: {filename}")

logging.info("🔁 Loading data...")

try:
    # Load small local file
    df_path = os.path.join(BASE_DIR, 'df_cleaned.pkl')
    df = joblib.load(df_path)

    # Download large file from Hugging Face if missing
    cosine_path = os.path.join(BASE_DIR, 'cosine_sim.pkl')
    download_file(
        "https://huggingface.co/datasets/Sameer0904/movie-recommendation/blob/main/cosine_sim.pkl",
        cosine_path
    )
    cosine_sim = joblib.load(cosine_path)

    logging.info("✅ Data loaded successfully.")
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

    # Create DataFrame with clean serial numbers starting from 1
    result_df = df[['title']].iloc[movie_indices].reset_index(drop=True)
    result_df.index = result_df.index + 1  # Start from 1 instead of 0
    result_df.index.name = "S.No."

    return result_df

