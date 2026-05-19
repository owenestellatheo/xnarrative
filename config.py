"""Central config — tune knobs here."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- API keys ---
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
X_USERNAME = os.getenv("X_USERNAME")  # burner account
X_EMAIL = os.getenv("X_EMAIL")
X_PASSWORD = os.getenv("X_PASSWORD")

# --- Models ---
MODEL_HAIKU = "claude-haiku-4-5-20251001"
MODEL_SONNET = "claude-sonnet-4-6"
MODEL_OPUS = "claude-opus-4-7"

# --- Paths ---
ROOT = Path(__file__).parent
CACHE_DIR = ROOT / "cache"
SESSION_FILE = ROOT / "session.json"
CACHE_DIR.mkdir(exist_ok=True)

# --- Scraping ---
MAX_TWEETS_PER_LANG = 300       # be polite to X
SCRAPE_SLEEP_SECONDS = 2.0      # between requests

# --- Analysis ---
TRANSLATION_BATCH = 40
PER_TWEET_ANALYSIS_BATCH = 30
SPIKE_ZSCORE_THRESHOLD = 2.5
ENGAGEMENT_OUTLIER_ZSCORE = 2.5
TEMPORAL_BIN_MINUTES = 30
MIN_CLUSTER_SIZE = 5

# --- Pipeline stages (used by --force-stage) ---
STAGES = [
    "scrape",
    "localize",
    "translate",
    "embed",
    "cluster",
    "per_tweet",
    "outliers",
    "temporal",
    "spike_summary",
    "narrative",
    "knowledge_graph",
]
