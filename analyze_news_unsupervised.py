import pandas as pd
import nltk
import re
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score # Added
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
import argparse
from pathlib import Path
import sys
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt # Added

# Add project root to path to allow importing project modules
sys.path.append(str(Path(__file__).resolve().parent))
from main_files.config_loader import config

logger = config.logger

# Download necessary NLTK data (run this once)
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    # Ensure punkt_tab is also available, as it's sometimes needed by the tokenizer
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')


def preprocess_text(text: str) -> str:
    """
    Cleans and preprocesses a single text string.
    - Lowercase
    - Remove punctuation and numbers
    - Tokenize
    - Remove stopwords
    - Lemmatize
    """
    if not isinstance(text, str):
        return ""

    lemmatizer = WordNetLemmatizer()
    
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
    text = re.sub(r'\d+', '', text)      # Remove numbers
    
    tokens = nltk.word_tokenize(text)
    
    stop_words = set(stopwords.words('english'))
    # custom_stopwords = ['eth', 'ethereum', 'crypto', 'bitcoin', 'btc', 'price'] 
    # stop_words.update(custom_stopwords)

    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return " ".join(tokens)

def get_top_terms_for_sbert_clusters(documents_series: pd.Series, labels: np.ndarray, n_terms=10):
    """
    Extracts the most frequent N terms for each cluster from the original processed text.
    Assumes documents_series contains the 'processed_text'.
    """
    df = pd.DataFrame({'doc': documents_series, 'label': labels})
    top_terms = {}
    for cluster_id in np.unique(labels):
        cluster_texts = " ".join(df[df['label'] == cluster_id]['doc'])
        words = cluster_texts.split()
        most_common_words = [word for word, count in Counter(words).most_common(n_terms)]
        top_terms[cluster_id] = most_common_words
    return top_terms

def plot_optimal_k_metrics(k_range, inertia_scores, silhouette_scores, output_dir="reports/unsupervised_analysis"):
    """Plots Elbow method and Silhouette scores."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(12, 7)) # Increased figure size

    # Plot Elbow Method (Inertia)
    color = 'tab:red'
    ax1.set_xlabel('Number of Clusters (K)')
    ax1.set_ylabel('Inertia (Sum of Squared Distances)', color=color)
    ax1.plot(k_range, inertia_scores, marker='o', color=color, label='Inertia (Elbow Method)')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle=':', alpha=0.7) # Adjusted grid

    # Create a second y-axis for Silhouette Scores
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Silhouette Score', color=color) # S
    ax2.plot(k_range, silhouette_scores, marker='x', color=color, linestyle='--', label='Silhouette Score')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.suptitle('Optimal K Analysis: Elbow Method & Silhouette Score', fontsize=16) # Changed to suptitle
    fig.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
    
    # Add legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    # Place legend outside plot for clarity
    fig.legend(lines + lines2, labels + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=2) 
    
    plot_path = Path(output_dir) / "optimal_k_plot.png"
    plt.savefig(plot_path, bbox_inches='tight') # Added bbox_inches
    logger.info(f"Optimal K analysis plot saved to: {plot_path}")
    plt.close(fig)

def main(input_csv_path: str, text_column: str, n_clusters: int, output_path: str = None, 
         sbert_model_name: str = 'all-MiniLM-L6-v2', find_optimal_k: bool = False, max_k: int = 15): # Signature updated
    
    logger.info(f"Starting unsupervised news analysis for: {input_csv_path}")
    if find_optimal_k:
        logger.info(f"Mode: Finding optimal K (up to K={max_k}). Clustering will be performed for each K.")
    else:
        logger.info(f"Mode: Clustering with K={n_clusters}.")
    logger.info(f"Text column: '{text_column}', SBERT Model: {sbert_model_name}")

    try:
        news_df = pd.read_csv(input_csv_path)
    except FileNotFoundError:
        logger.error(f"Error: Input CSV file not found at {input_csv_path}")
        return
    except Exception as e:
        logger.error(f"Error reading CSV file: {e}")
        return

    if text_column not in news_df.columns:
        logger.error(f"Error: Text column '{text_column}' not found in the CSV.")
        logger.info(f"Available columns: {news_df.columns.tolist()}")
        return

    news_df.dropna(subset=[text_column], inplace=True)
    if news_df.empty:
        logger.warning("DataFrame is empty after dropping NaN values in text column.")
        return
        
    logger.info(f"Loaded {len(news_df)} news articles.")
    
    logger.info("Preprocessing text data...")
    news_df['processed_text'] = news_df[text_column].apply(preprocess_text)
    
    news_df = news_df[news_df['processed_text'].str.strip().astype(bool)]
    if news_df.empty:
        logger.warning("DataFrame is empty after preprocessing and filtering empty strings.")
        return

    logger.info(f"Number of articles after preprocessing: {len(news_df)}")

    # Sentence-BERT Embeddings
    logger.info(f"Generating Sentence-BERT embeddings using model: {sbert_model_name}...")
    try:
        sbert_model = SentenceTransformer(sbert_model_name)
        embeddings = sbert_model.encode(news_df['processed_text'].tolist(), show_progress_bar=True)
    except Exception as e:
        logger.error(f"Error during Sentence-BERT embedding generation: {e}")
        return
    
    if embeddings.shape[0] == 0:
        logger.error("Embeddings matrix is empty.")
        return
    
    actual_n_clusters = min(n_clusters, embeddings.shape[0])
    if actual_n_clusters < n_clusters:
        logger.warning(f"Number of clusters reduced from {n_clusters} to {actual_n_clusters} due to insufficient samples.")
    
    if actual_n_clusters < 2 and embeddings.shape[0] > 0 and not find_optimal_k: # Check only if not finding optimal_k
         logger.error(f"Cannot perform K-Means clustering with {actual_n_clusters} sample(s). Need at least 2 (or 1 if only 1 sample).")
         if embeddings.shape[0] == 1: # Handle the case of a single document
             news_df['cluster'] = 0
             logger.info("Only one document found. Assigned to cluster 0.")
             if output_path:
                try:
                    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                    news_df.to_csv(output_path, index=False, encoding='utf-8-sig')
                    logger.info(f"Clustered news data saved to: {output_path}")
                except Exception as e:
                    logger.error(f"Error saving output CSV: {e}")
             return # End processing for single document
         else: # Not enough samples for meaningful clustering
             return

    if find_optimal_k:
        # K must be < n_samples for silhouette score, and >= 2
        # We also cap it at max_k to avoid excessive computation
        # Ensure there are enough unique samples for the range of K
        upper_k_bound = min(max_k + 1, embeddings.shape[0])
        k_range = range(2, upper_k_bound) 

        if len(k_range) < 1 : # Check if k_range is empty or has only one value
            logger.error(f"Not enough samples ({embeddings.shape[0]}) or too small max_k ({max_k}) to test K range for optimal K analysis. Need at least 2 samples and max_k >= 2.")
            return

        inertia_scores = []
        silhouette_scores = []
        logger.info(f"Calculating Inertia and Silhouette scores for K in {list(k_range)}...")

        for k_val in k_range:
            logger.info(f"Processing K={k_val}...")
            kmeans_model = KMeans(n_clusters=k_val, random_state=42, n_init='auto')
            cluster_labels = kmeans_model.fit_predict(embeddings)
            inertia_scores.append(kmeans_model.inertia_)
            
            # Silhouette score requires at least 2 distinct cluster labels
            if len(np.unique(cluster_labels)) > 1:
                score = silhouette_score(embeddings, cluster_labels)
                silhouette_scores.append(score)
                logger.info(f"K={k_val}, Inertia={kmeans_model.inertia_:.2f}, Silhouette Score={score:.4f}")
            else:
                silhouette_scores.append(np.nan) # Not applicable if only one cluster is formed
                logger.info(f"K={k_val}, Inertia={kmeans_model.inertia_:.2f}, Silhouette Score: N/A (only one cluster found or all points in one cluster)")
        
        if k_range and inertia_scores and silhouette_scores: # Ensure lists are not empty
             plot_optimal_k_metrics(list(k_range), inertia_scores, silhouette_scores)
        else:
            logger.warning("Could not generate optimal K plot due to empty score lists (possibly too few samples or K values).")
        logger.info("Optimal K analysis completed. To apply a specific K, re-run without --find_optimal_k and set --clusters.")

    else: # Perform clustering with the specified n_clusters
        if actual_n_clusters < 2 and embeddings.shape[0] > 1 : # Need at least 2 clusters if more than 1 sample
             logger.error(f"Cannot perform K-Means clustering with {actual_n_clusters} sample(s) for K={n_clusters}. Need at least 2 clusters for multiple samples.")
             return
        if embeddings.shape[0] == 0:
            logger.error("No data to cluster after preprocessing.")
            return
        if embeddings.shape[0] == 1 and actual_n_clusters == 1: # Special case for single sample
            news_df['cluster'] = 0
            logger.info("Only one document processed. Assigned to cluster 0.")
        elif actual_n_clusters >= 2 :
            logger.info(f"Performing K-Means clustering with {actual_n_clusters} clusters on SBERT embeddings...")
            kmeans = KMeans(n_clusters=actual_n_clusters, random_state=42, n_init='auto')
            news_df['cluster'] = kmeans.fit_predict(embeddings)
            
            logger.info("Top terms per cluster (based on word frequency in processed_text):")
            top_terms = get_top_terms_for_sbert_clusters(news_df['processed_text'], news_df['cluster'].values)
            for cluster_id, terms_list in top_terms.items():
                print(f"Cluster {cluster_id}: {', '.join(terms_list)}")
                logger.info(f"Cluster {cluster_id}: {', '.join(terms_list)}")
        else:
            logger.info("Skipping clustering due to insufficient samples or clusters.")


        if output_path and 'cluster' in news_df.columns: # Save only if clustering was done
            try:
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                news_df.to_csv(output_path, index=False, encoding='utf-8-sig')
                logger.info(f"Clustered news data saved to: {output_path}")
            except Exception as e:
                logger.error(f"Error saving output CSV: {e}")
                
    logger.info("Unsupervised analysis script finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unsupervised analysis of news articles using Sentence-BERT and K-Means.")
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/all_collected_news.csv",
        help="Path to the input CSV file containing news data."
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="text",
        help="Name of the column in the CSV containing the news text."
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=5,
        help="Number of clusters for K-Means (used if --find_optimal_k is false)."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/clustered_news_sbert_kmeans.csv",
        help="Path to save the CSV file with cluster assignments."
    )
    parser.add_argument(
        "--sbert_model",
        type=str,
        default="all-MiniLM-L6-v2", # A good general-purpose model
        help="Name of the SentenceTransformer model to use."
    )
    parser.add_argument(
        "--find_optimal_k",
        action="store_true",
        help="If set, run analysis for a range of K values to find optimal K instead of clustering with --clusters."
    )
    parser.add_argument(
        "--max_k",
        type=int,
        default=15,
        help="Maximum K value to test when --find_optimal_k is set."
    )
    args = parser.parse_args()

    main(
        input_csv_path=args.input, 
        text_column=args.text_column, 
        n_clusters=args.clusters, 
        output_path=args.output,
        sbert_model_name=args.sbert_model,
        find_optimal_k=args.find_optimal_k,
        max_k=args.max_k
    )
