# ============================================================
# NETWORK INTELLIGENCE MODEL INFERENCE
# Real-time fraud scoring for individual transactions
# ============================================================

import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from pathlib import Path
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

class NetworkFraudScorer:
    """Real-time fraud scoring using network intelligence"""
    
    def __init__(self, model_dir="artifacts/models"):
        self.model_dir = Path(model_dir)
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.network_cache = {}  # Cache network features
        
        logger.remove()
        logger.add(lambda msg: print(msg, end=""))
        
        self.load_model()
    
    def load_model(self):
        """Load trained model and artifacts"""
        logger.info("Loading fraud detection model...")
        
        # Load LightGBM model
        model_path = self.model_dir / "lgbm_model.txt"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        self.model = lgb.Booster(model_file=str(model_path))
        
        # Load scaler
        scaler_path = self.model_dir / "scaler.pkl"
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")
        self.scaler = joblib.load(scaler_path)
        
        # Load feature columns
        feature_path = self.model_dir / "feature_columns.txt"
        if not feature_path.exists():
            raise FileNotFoundError(f"Feature columns not found: {feature_path}")
        
        with open(feature_path, 'r') as f:
            self.feature_columns = [line.strip() for line in f.readlines()]
        
        logger.info(f"Model loaded with {len(self.feature_columns)} features")
    
    def score_transaction(self, cust_id, dest_id, amount, 
                         cust_network_features=None, dest_network_features=None):
        """
        Score a single transaction
        
        Parameters:
        - cust_id: Customer ID
        - dest_id: Destination ID  
        - amount: Transaction amount
        - cust_network_features: Dict of customer network features (optional)
        - dest_network_features: Dict of destination network features (optional)
        
        Returns:
        - Dict with fraud probability, prediction, and risk bucket
        """
        
        # Create transaction DataFrame
        transaction = pd.DataFrame({
            'cust_id': [cust_id],
            'dest_id': [dest_id],
            'amount': [amount],
            'amount_log': [np.log1p(amount)]
        })
        
        # Add network features (use cache or provided features)
        for feature in self.feature_columns:
            if feature == 'amount_log':
                continue  # Already added
                
            if feature.startswith('cust_'):
                feature_name = feature[5:]  # Remove 'cust_' prefix
                if cust_network_features and feature_name in cust_network_features:
                    transaction[feature] = cust_network_features[feature_name]
                elif cust_id in self.network_cache:
                    transaction[feature] = self.network_cache[cust_id].get(feature_name, 0)
                else:
                    transaction[feature] = 0  # Default value
            
            elif feature.startswith('dest_'):
                feature_name = feature[5:]  # Remove 'dest_' prefix
                if dest_network_features and feature_name in dest_network_features:
                    transaction[feature] = dest_network_features[feature_name]
                elif dest_id in self.network_cache:
                    transaction[feature] = self.network_cache[dest_id].get(feature_name, 0)
                else:
                    transaction[feature] = 0  # Default value
        
        # Ensure all features are present
        for feature in self.feature_columns:
            if feature not in transaction.columns:
                transaction[feature] = 0
        
        # Scale features
        X = transaction[self.feature_columns].values
        X_scaled = self.scaler.transform(X)
        
        # Make prediction
        fraud_probability = self.model.predict(X_scaled)[0]
        fraud_prediction = 1 if fraud_probability > 0.5 else 0
        
        # Risk bucket
        if fraud_probability >= 0.8:
            risk_bucket = "HIGH"
        elif fraud_probability >= 0.5:
            risk_bucket = "MEDIUM"
        else:
            risk_bucket = "LOW"
        
        return {
            'cust_id': cust_id,
            'dest_id': dest_id,
            'amount': amount,
            'fraud_probability': round(fraud_probability, 4),
            'fraud_prediction': fraud_prediction,
            'risk_bucket': risk_bucket
        }
    
    def score_batch(self, transactions_df):
        """Score a batch of transactions"""
        results = []
        
        for _, row in transactions_df.iterrows():
            result = self.score_transaction(
                cust_id=row['cust_id'],
                dest_id=row['dest_id'], 
                amount=row['amount']
            )
            results.append(result)
        
        return pd.DataFrame(results)
    
    def update_network_cache(self, node_id, network_features):
        """Update cached network features for a node"""
        self.network_cache[node_id] = network_features
    
    def clear_cache(self):
        """Clear network feature cache"""
        self.network_cache.clear()


# ============================================================
# DEMO USAGE
# ============================================================

def demo_inference():
    """Demonstrate how to use the scorer"""
    logger.info("Network Intelligence Fraud Scorer Demo")
    logger.info("=" * 50)
    
    # Initialize scorer
    scorer = NetworkFraudScorer()
    
    # Example transactions
    test_transactions = [
        {'cust_id': 'C123456789', 'dest_id': 'M987654321', 'amount': 50.0},
        {'cust_id': 'C111111111', 'dest_id': 'M222222222', 'amount': 10000.0},
        {'cust_id': 'C333333333', 'dest_id': 'M444444444', 'amount': 25.75},
        {'cust_id': 'C555555555', 'dest_id': 'M666666666', 'amount': 500000.0},
    ]
    
    # Score individual transactions
    logger.info("Scoring individual transactions:")
    for i, txn in enumerate(test_transactions, 1):
        result = scorer.score_transaction(**txn)
        logger.info(f"Transaction {i}: ${txn['amount']:,.2f} -> {result['risk_bucket']} risk ({result['fraud_probability']:.3f})")
    
    # Score as batch
    logger.info("\nBatch scoring:")
    batch_df = pd.DataFrame(test_transactions)
    batch_results = scorer.score_batch(batch_df)
    logger.info(batch_results[['cust_id', 'amount', 'fraud_probability', 'risk_bucket']])
    
    # Example with network features
    logger.info("\nScoring with custom network features:")
    custom_result = scorer.score_transaction(
        cust_id='HIGH_RISK_CUSTOMER',
        dest_id='SUSPICIOUS_MERCHANT', 
        amount=1000.0,
        cust_network_features={
            'degree_centrality': 0.95,  # Highly connected
            'betweenness_centrality': 0.8,
            'pagerank': 0.01
        },
        dest_network_features={
            'degree_centrality': 0.02,  # Isolated
            'triangle_count': 0,
            'community_size': 1
        }
    )
    logger.info(f"Custom features result: {custom_result}")

if __name__ == "__main__":
    demo_inference()