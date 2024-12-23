import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ranking.ml import MLRanker
from scripts.gather_training_data import load_labeled_data
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_model(y_true, y_pred):
    """Evaluate model performance."""
    # Convert scores to binary classes for evaluation
    y_true_binary = (np.array(y_true) >= 0.5).astype(int)
    y_pred_binary = (np.array(y_pred) >= 0.5).astype(int)
    
    metrics = {
        'accuracy': accuracy_score(y_true_binary, y_pred_binary),
        'precision': precision_score(y_true_binary, y_pred_binary),
        'recall': recall_score(y_true_binary, y_pred_binary),
        'f1': f1_score(y_true_binary, y_pred_binary)
    }
    
    return metrics

def main():
    """Main function to train and evaluate the ML model."""
    # Load labeled data
    labeled_data = load_labeled_data()
    
    if not labeled_data:
        logger.error("No labeled data found")
        return
        
    # Split data into features and labels
    X = labeled_data
    y = [link['manual_relevance_score'] for link in labeled_data]
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )
    
    # Initialize and train model
    model = MLRanker()
    logger.info("Training model...")
    model.train(X_train, y_train)
    
    # Evaluate on test set
    logger.info("Evaluating model...")
    test_scores = model.batch_calculate_scores(X_test)
    metrics = evaluate_model(y_test, test_scores)
    
    logger.info("Model Performance:")
    for metric, value in metrics.items():
        logger.info(f"{metric.capitalize()}: {value:.3f}")
    
    # Save model
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    model.save_model(model_dir)
    logger.info(f"Model saved to {model_dir}")
    
    # Example predictions
    logger.info("\nExample Predictions:")
    for i in range(min(5, len(X_test))):
        true_score = y_test[i]
        pred_score = test_scores[i]
        url = X_test[i]['url']
        logger.info(f"URL: {url}")
        logger.info(f"True Score: {true_score:.2f}")
        logger.info(f"Predicted Score: {pred_score:.2f}\n")

if __name__ == "__main__":
    main() 
