import joblib
import logging
import os
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
    movie_name = movie_name.strip().lower()

    idx_list = df[df['title'].str.lower() == movie_name].index
    if len(idx_list) == 0:
        logging.warning("⚠️ Movie '%s' not found in dataset.", movie_name)
        return None

    idx = idx_list[0]

    if idx >= cosine_sim.shape[0]:
        logging.error("❌ Index %d is out of range for cosine similarity matrix (max %d).",
                      idx, cosine_sim.shape[0] - 1)
        return None

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n + 1]
    movie_indices = [i[0] for i in sim_scores]

    logging.info("✅ Top %d recommendations ready.", top_n)

    result_df = df[['title']].iloc[movie_indices].reset_index(drop=True)
    result_df.index = result_df.index + 1
    result_df.index.name = "S.No."

    return result_df

