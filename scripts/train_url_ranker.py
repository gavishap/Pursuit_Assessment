import os
from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import ndcg_score
from sklearn.preprocessing import MinMaxScaler
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define paths
ROOT_DIR = Path(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = ROOT_DIR / 'data' / 'mslr'
MODEL_DIR = ROOT_DIR / 'models'
MODEL_FILE = MODEL_DIR / 'url_ranker.txt'

def load_fold(fold_num: int):
    """Load data from a specific fold."""
    fold_dir = DATA_DIR / f'Fold{fold_num}'
    
    train_file = fold_dir / 'train.txt'
    vali_file = fold_dir / 'vali.txt'
    test_file = fold_dir / 'test.txt'
    
    # Function to read MSLR format
    def read_mslr(filepath):
        data = []
        with open(filepath, 'r') as f:
            for line in f:
                # Split line into label, qid, and features
                parts = line.strip().split()
                label = int(parts[0])
                qid = int(parts[1].split(':')[1])
                features = [float(x.split(':')[1]) for x in parts[2:]]
                data.append([label, qid] + features)
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        df.columns = ['label', 'qid'] + [f'feature_{i}' for i in range(len(df.columns)-2)]
        return df
    
    return (
        read_mslr(train_file),
        read_mslr(vali_file),
        read_mslr(test_file)
    )

def normalize_relevance(df: pd.DataFrame) -> pd.DataFrame:
    """Keep relevance scores as integers 0-4 for ranking."""
    # Labels should stay as integers for LightGBM ranking
    return df

def analyze_features(df: pd.DataFrame):
    """Analyze feature distributions and correlations with relevance."""
    logger.info("Analyzing features...")
    
    # Get feature columns
    feature_cols = [col for col in df.columns if col.startswith('feature_')]
    
    # Calculate correlation with label
    correlations = []
    for col in feature_cols:
        corr = df[col].corr(df['label'])
        correlations.append((col, corr))
    
    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    
    logger.info("\nTop 20 features by correlation with relevance:")
    for col, corr in correlations[:20]:
        logger.info(f"{col}: {corr:.4f}")
    
    # Analyze label distribution
    logger.info("\nLabel Distribution:")
    label_dist = df['label'].value_counts().sort_index()
    print(label_dist)
    
    # Create plots directory
    plots_dir = ROOT_DIR / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    # Save plots
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Label distribution
    plt.subplot(1, 2, 1)
    sns.histplot(data=df, x='label', bins=5)
    plt.title('Label Distribution')
    
    # Plot 2: Feature correlations
    plt.subplot(1, 2, 2)
    corr_values = [x[1] for x in correlations[:20]]
    corr_names = [x[0] for x in correlations[:20]]
    sns.barplot(x=corr_values, y=corr_names)
    plt.title('Top 20 Feature Correlations')
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'mslr_analysis.png')
    logger.info("\nPlots saved to plots/mslr_analysis.png")
    
    return correlations

def prepare_data(df: pd.DataFrame):
    """Prepare data for LightGBM training with selected features."""
    # Use pre-analyzed top features
    top_features = [
        'feature_7',   # 0.2381 - Covered query term ratio (anchor)
        'feature_97',  # 0.2362 - Boolean model (anchor)
        'feature_122', # 0.2178 - LMIR.JM (anchor)
        'feature_107', # 0.2167 - BM25 (anchor)
        'feature_102', # 0.2110 - Vector space model (anchor)
        'feature_112', # 0.2070 - LMIR.ABS (anchor)
        'feature_114', # 0.2058 - LMIR.ABS (url)
        'feature_124', # 0.1977 - LMIR.JM (url)
        'feature_110', # 0.1963 - BM25 (url)
        'feature_120', # 0.1893 - LMIR.DIR (url)
        'feature_47',  # 0.1873 - Stream length normalized term frequency (anchor)
        'feature_62',  # 0.1848 - Mean of stream length normalized term frequency (anchor)
        'feature_52',  # 0.1827 - Min of stream length normalized term frequency (anchor)
        'feature_2',   # 0.1805 - Covered query term number (anchor)
        'feature_119', # 0.1699 - LMIR.DIR (whole document)
        'feature_63',  # 0.1645 - Mean of stream length normalized term frequency (title)
        'feature_64',  # 0.1640 - Mean of stream length normalized term frequency (url)
        'feature_115', # 0.1637 - LMIR.ABS (whole document)
        'feature_8',   # 0.1598 - Covered query term ratio (title)
        'feature_54'   # 0.1583 - Min of stream length normalized term frequency (url)
    ]
    
    # Extract features and normalize them
    X = df[top_features].values
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    
    y = df['label'].values
    groups = df.groupby('qid').size().values
    
    logger.info(f"Using top {len(top_features)} most correlated features")
    
    return X, y, groups

def train_model(X_train, y_train, groups_train, X_vali, y_vali, groups_vali):
    """Train LightGBM ranker optimized for URL relevance ranking."""
    train_data = lgb.Dataset(
        X_train,
        label=y_train,
        group=groups_train,
        free_raw_data=False
    )
    
    valid_data = lgb.Dataset(
        X_vali,
        label=y_vali,
        group=groups_vali,
        reference=train_data,
        free_raw_data=False
    )
    
    # Parameters optimized for URL ranking
    params = {
        'objective': 'lambdarank',
        'metric': ['ndcg', 'map'],
        'ndcg_eval_at': [1, 3, 5, 10],
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': 6,
        'min_data_in_leaf': 50,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'cat_smooth': 10,
        'label_gain': [0, 1, 3, 7, 15],  # Gain for labels 0,1,2,3,4
        'verbose': -1
    }
    
    logger.info("Training model...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[valid_data],
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(100)
        ]
    )
    
    # Log feature importance
    importance = model.feature_importance(importance_type='gain')
    feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
    importance_df = pd.DataFrame({'feature': feature_names, 'importance': importance})
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    logger.info("\nTop 10 most important features:")
    for _, row in importance_df.head(10).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.2f}")
    
    return model

def evaluate_model(model, X, y, groups):
    """Evaluate model using nDCG and check score distributions."""
    predictions = model.predict(X)
    
    # Calculate nDCG for each query group
    ndcg_scores = []
    start_idx = 0
    
    for group_size in groups:
        end_idx = start_idx + group_size
        group_preds = predictions[start_idx:end_idx]
        group_true = y[start_idx:end_idx]
        
        if group_size > 1:
            group_preds = group_preds.reshape(1, -1)
            group_true = group_true.reshape(1, -1)
            k = min(10, group_size)
            score = ndcg_score(group_true, group_preds, k=k)
            ndcg_scores.append(score)
        
        start_idx = end_idx
    
    # Normalize predictions to 0-1 range for final output
    predictions = (predictions - predictions.min()) / (predictions.max() - predictions.min())
    
    # Analyze prediction distribution
    logger.info(f"Prediction stats (normalized to 0-1):")
    logger.info(f"  Min: {predictions.min():.4f}")
    logger.info(f"  Max: {predictions.max():.4f}")
    logger.info(f"  Mean: {predictions.mean():.4f}")
    logger.info(f"  Std: {predictions.std():.4f}")
    
    return np.mean(ndcg_scores) if ndcg_scores else 0.0

def main():
    """Main function to train and evaluate the model using all folds."""
    try:
        # First check what features are available in the first fold
        logger.info("Checking available features in first fold...")
        train_df, _, _ = load_fold(1)
        features = [col for col in train_df.columns if col.startswith('feature_')]
        logger.info(f"Dataset has {len(features)} features")
        logger.info(f"Feature range: from {min(features)} to {max(features)}")
        
        # Load and combine all folds
        logger.info("\nLoading all folds...")
        all_train_dfs = []
        all_vali_dfs = []
        all_test_dfs = []
        
        for fold in range(1, 6):
            logger.info(f"Processing fold {fold}...")
            train_df, vali_df, test_df = load_fold(fold)
            
            # Keep original integer labels
            train_df = normalize_relevance(train_df)
            vali_df = normalize_relevance(vali_df)
            test_df = normalize_relevance(test_df)
            
            all_train_dfs.append(train_df)
            all_vali_dfs.append(vali_df)
            all_test_dfs.append(test_df)
        
        # Combine all folds
        train_df = pd.concat(all_train_dfs, ignore_index=True)
        vali_df = pd.concat(all_vali_dfs, ignore_index=True)
        test_df = pd.concat(all_test_dfs, ignore_index=True)
        
        logger.info(f"Combined dataset sizes:")
        logger.info(f"  Train: {len(train_df):,} samples")
        logger.info(f"  Validation: {len(vali_df):,} samples")
        logger.info(f"  Test: {len(test_df):,} samples")
        
        # Prepare data using pre-analyzed top features
        logger.info("Preparing data...")
        X_train, y_train, groups_train = prepare_data(train_df)
        X_vali, y_vali, groups_vali = prepare_data(vali_df)
        X_test, y_test, groups_test = prepare_data(test_df)
        
        # Train model
        model = train_model(X_train, y_train, groups_train, X_vali, y_vali, groups_vali)
        
        # Evaluate model
        logger.info("Evaluating model...")
        test_ndcg = evaluate_model(model, X_test, y_test, groups_test)
        logger.info(f"Test nDCG@10: {test_ndcg:.4f}")
        
        # Save model
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model.save_model(str(MODEL_FILE))
        logger.info(f"Model saved to {MODEL_FILE}")
        
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        raise

if __name__ == "__main__":
    main() 
