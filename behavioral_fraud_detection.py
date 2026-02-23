# ============================================================
# BEHAVIORAL FRAUD DETECTION - USAGE SCRIPT
# Load and use behavioral fraud detection model
# ============================================================
import os
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import json

class BehavioralFraudPredictor:
    """Production-ready behavioral fraud prediction system"""
    
    def __init__(self, models_path="artifacts/behavioral_"):
        self.models_path = Path(models_path)
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.load_model()
    
    def load_model(self):
        """Load behavioral fraud detection model"""
        try:
            # Try to find behavioral model files
            model_file = None
            scaler_file = None
            features_file = None
            
            for root, dirs, files in os.walk(self.models_path):
                for file in files:
                    if 'behavioral' in file.lower() and file.endswith(('.pkl', '.joblib')):
                        if 'model' in file:
                            model_file = Path(root) / file
                        elif 'scaler' in file:
                            scaler_file = Path(root) / file
                        elif 'features' in file:
                            features_file = Path(root) / file
            
            if model_file and model_file.exists():
                self.model = joblib.load(model_file)
                print(f"✅ Behavioral model loaded from {model_file}")
            
            if scaler_file and scaler_file.exists():
                self.scaler = joblib.load(scaler_file)
                print(f"✅ Scaler loaded from {scaler_file}")
            
            if features_file and features_file.exists():
                self.feature_columns = joblib.load(features_file)
                print(f"✅ Feature columns loaded from {features_file}")
            
            if not self.model:
                print("⚠️ No behavioral model found, using fallback prediction")
                
        except Exception as e:
            print(f"❌ Error loading behavioral model: {e}")
    
    def predict_behavioral_risk(self, transaction_data):
        """Predict behavioral fraud risk for a transaction"""
        try:
            if self.model is None:
                return self._fallback_prediction(transaction_data)
            
            # Extract behavioral features
            features = self._extract_behavioral_features(transaction_data)
            
            # Scale features if scaler available
            if self.scaler:
                features_scaled = self.scaler.transform([features])
            else:
                features_scaled = [features]
            
            # Get prediction
            if hasattr(self.model, 'predict_proba'):
                risk_score = self.model.predict_proba(features_scaled)[0][1]
            else:
                risk_score = self.model.predict(features_scaled)[0]
            
            return {
                'behavioral_risk_score': float(risk_score),
                'risk_level': self._categorize_risk(risk_score),
                'key_factors': self._identify_risk_factors(transaction_data),
                'confidence': min(0.95, max(0.6, abs(risk_score - 0.5) * 2)),
                'model_used': 'behavioral_fraud_detection'
            }
            
        except Exception as e:
            print(f"⚠️ Error in behavioral prediction: {e}")
            return self._fallback_prediction(transaction_data)
    
    def _extract_behavioral_features(self, data):
        """Extract behavioral features from transaction data"""
        features = []
        
        # Device behavior features
        features.append(1.0 / (data.get('device_seen_count', 1) + 1))  # device_trust_score
        features.append(float(not data.get('is_device_shared', False)))  # device_exclusivity
        features.append(min(1.0, data.get('velocity_score', 5) / 10))  # velocity_risk
        features.append(min(1.0, data.get('spending_deviation_score', 0.5)))  # spending_anomaly
        
        # Temporal behavior features
        features.append(float(data.get('is_night_txn', False)))  # night_risk
        features.append(float(data.get('is_weekend', False)))  # weekend_activity
        
        # Transaction behavior features
        amount = data.get('amount', data.get('amount_ngn', 1000))
        features.append(np.log1p(amount))  # amount_log
        features.append(float(amount % 1000 == 0))  # round_amount
        
        # Pad or trim to expected feature count
        expected_count = len(self.feature_columns) if self.feature_columns else 10
        while len(features) < expected_count:
            features.append(0.5)  # Safe default
        
        return features[:expected_count]
    
    def _categorize_risk(self, score):
        """Categorize risk score into levels"""
        if score > 0.8:
            return 'CRITICAL'
        elif score > 0.6:
            return 'HIGH'
        elif score > 0.4:
            return 'MEDIUM'
        elif score > 0.2:
            return 'LOW'
        else:
            return 'MINIMAL'
    
    def _identify_risk_factors(self, data):
        """Identify key behavioral risk factors"""
        factors = []
        
        if data.get('device_seen_count', 10) <= 3:
            factors.append('new_device')
        if data.get('is_device_shared', False):
            factors.append('shared_device')
        if data.get('velocity_score', 5) > 8:
            factors.append('high_velocity')
        if data.get('spending_deviation_score', 0.5) > 0.7:
            factors.append('unusual_spending')
        if data.get('is_night_txn', False):
            factors.append('night_transaction')
        
        return factors[:3] if factors else ['normal_behavior']
    
    def _fallback_prediction(self, data):
        """Fallback prediction when model unavailable"""
        # Simple rule-based scoring
        risk_score = 0.0
        
        if data.get('velocity_score', 5) > 8:
            risk_score += 0.3
        if data.get('spending_deviation_score', 0.5) > 0.7:
            risk_score += 0.3
        if data.get('is_night_txn', False):
            risk_score += 0.2
        if data.get('device_seen_count', 10) <= 3:
            risk_score += 0.2
        
        return {
            'behavioral_risk_score': min(1.0, risk_score),
            'risk_level': self._categorize_risk(risk_score),
            'key_factors': ['rule_based_prediction'],
            'confidence': 0.7,
            'model_used': 'fallback_rules'
        }

# Usage example
def demo_behavioral_prediction():
    """Demo behavioral fraud prediction"""
    predictor = BehavioralFraudPredictor()
    
    # Example transaction
    transaction = {
        'device_seen_count': 2,
        'is_device_shared': True,
        'velocity_score': 9.2,
        'spending_deviation_score': 0.85,
        'is_night_txn': True,
        'is_weekend': False,
        'amount_ngn': 50000
    }
    
    result = predictor.predict_behavioral_risk(transaction)
    
    print("🧠 BEHAVIORAL FRAUD PREDICTION RESULT:")
    print(f"   Risk Score: {result['behavioral_risk_score']:.3f}")
    print(f"   Risk Level: {result['risk_level']}")
    print(f"   Key Factors: {', '.join(result['key_factors'])}")
    print(f"   Confidence: {result['confidence']:.3f}")
    
    return result

if __name__ == "__main__":
    demo_behavioral_prediction()