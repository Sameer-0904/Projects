import joblib
import logging
import os
import requests

# Google Drive file URLs (direct download)
DF_URL = "https://drive.google.com/uc?export=download&id=15qvUpi_ZoHHMX1dEVhXq4jqegjFFvyGm"
COSINE_URL = "https://drive.google.com/uc?export=download&id=14rs3W0tlucexSnDkn24KT3udXVq2B71Q"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("recommend.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

def download_file(url, filename):
    """Download a file from Google Drive if it doesn't exist."""
    if not os.path.exists(filename):
        logging.info(f"⬇️ Downloading {filename} from Google Drive...")
        r = requests.get(url, stream=True)
        if r.status_code == 200:
            with open(filename, 'wb') as f:
                f.write(r.content)
            logging.info(f"✅ {filename} downloaded.")
        else:
            logging.error(f"❌ Failed to download {filename}, status code: {r.status_code}")
            raise Exception(f"Download failed: {filename}")

# Download files if missing
download_file(DF_URL, "df_cleaned.pkl")
download_file(COSINE_URL, "cosine_sim.pkl")

logging.info("🔁 Loading data...")
try:
    df = joblib.load('df_cleaned.pkl')
    cosine_sim = joblib.load('cosine_sim.pkl')
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

    result_df = df[['title']].iloc[movie_indices].reset_index(drop=True)
    result_df.index = result_df.index + 1
    result_df.index.name = "S.No."
    return result_df
