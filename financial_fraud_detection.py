# ============================================================
# FINANCIAL CREDIT RISK - USAGE SCRIPT
# Load and use financial credit risk model
# ============================================================
import os
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

class FinancialRiskPredictor:
    """Production-ready financial credit risk prediction system"""
    
    def __init__(self, models_path="artifacts/financial_"):
        self.models_path = Path(models_path)
        self.model = None
        self.scaler = None
        self.encoders = {}
        self.feature_columns = None
        self.load_model()
    
    def load_model(self):
        """Load financial risk model"""
        try:
            # Find financial model files
            for root, dirs, files in os.walk(self.models_path):
                for file in files:
                    if 'financial' in file.lower() and file.endswith(('.pkl', '.joblib')):
                        file_path = Path(root) / file
                        
                        if 'model' in file:
                            self.model = joblib.load(file_path)
                            print(f"✅ Financial model loaded from {file_path}")
                        elif 'scaler' in file:
                            self.scaler = joblib.load(file_path)
                            print(f"✅ Scaler loaded from {file_path}")
                        elif 'encoders' in file:
                            self.encoders = joblib.load(file_path)
                            print(f"✅ Encoders loaded from {file_path}")
                        elif 'features' in file:
                            self.feature_columns = joblib.load(file_path)
                            print(f"✅ Feature columns loaded from {file_path}")
            
            if not self.model:
                print("⚠️ No financial model found, using fallback prediction")
                
        except Exception as e:
            print(f"❌ Error loading financial model: {e}")
    
    def predict_financial_risk(self, transaction_data):
        """Predict financial fraud risk for a transaction"""
        try:
            if self.model is None:
                return self._fallback_prediction(transaction_data)
            
            # Extract financial features
            features = self._extract_financial_features(transaction_data)
            
            # Scale features
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
                'financial_risk_score': float(risk_score),
                'risk_level': self._categorize_risk(risk_score),
                'risk_factors': self._analyze_financial_risks(transaction_data),
                'amount_analysis': self._analyze_amount_patterns(transaction_data),
                'confidence': min(0.95, max(0.6, abs(risk_score - 0.5) * 2)),
                'model_used': 'financial_credit_risk'
            }
            
        except Exception as e:
            print(f"⚠️ Error in financial prediction: {e}")
            return self._fallback_prediction(transaction_data)
    
    def _extract_financial_features(self, data):
        """Extract financial features from transaction data"""
        features = []
        
        # Basic transaction features
        amount = data.get('amount', 1000)
        features.append(amount)  # amount
        features.append(np.log1p(amount))  # amount_log
        features.append(np.sqrt(amount))  # amount_sqrt
        
        # Balance features
        old_balance = data.get('oldbalanceOrg', 10000)
        new_balance = data.get('newbalanceOrig', old_balance - amount)
        features.append(old_balance)  # oldbalanceOrg
        features.append(new_balance)  # newbalanceOrig
        
        # Balance ratios and changes
        features.append(amount / (old_balance + 1))  # balance_ratio
        features.append(new_balance - old_balance)  # balance_change
        
        # Transaction type (encode if available)
        txn_type = data.get('type', 'PAYMENT')
        type_mapping = {'PAYMENT': 0, 'TRANSFER': 1, 'CASH_OUT': 2, 'DEBIT': 3, 'CASH_IN': 4}
        features.append(type_mapping.get(txn_type, 0))  # type_encoded
        
        # Risk indicators
        features.append(float(amount > 100000))  # is_large_amount
        features.append(float(amount % 1000 == 0))  # is_round_amount
        features.append(float(old_balance == 0))  # zero_balance_orig
        
        # Customer behavior (simulated if not available)
        features.append(data.get('customer_txn_count', 5))  # customer_txn_count
        features.append(data.get('customer_avg_amount', amount))  # customer_avg_amount
        
        # Temporal features
        step = data.get('step', 100)
        features.append(step % 24)  # transaction_hour
        features.append(float((step % 24) >= 9 and (step % 24) <= 17))  # is_business_hours
        
        # Pad to expected feature count
        expected_count = len(self.feature_columns) if self.feature_columns else 15
        while len(features) < expected_count:
            features.append(0.0)
        
        return features[:expected_count]
    
    def _categorize_risk(self, score):
        """Categorize financial risk score"""
        if score > 0.85:
            return 'CRITICAL'
        elif score > 0.7:
            return 'HIGH'
        elif score > 0.5:
            return 'MEDIUM'
        elif score > 0.3:
            return 'LOW'
        else:
            return 'MINIMAL'
    
    def _analyze_financial_risks(self, data):
        """Analyze specific financial risk factors"""
        risks = []
        
        amount = data.get('amount', 1000)
        old_balance = data.get('oldbalanceOrg', 10000)
        
        if amount > 100000:
            risks.append('large_transaction')
        if amount > old_balance * 0.9:
            risks.append('account_drainage')
        if old_balance == 0:
            risks.append('zero_balance_account')
        if data.get('type') in ['TRANSFER', 'CASH_OUT']:
            risks.append('high_risk_type')
        if amount % 1000 == 0:
            risks.append('round_amount')
        
        return risks[:3] if risks else ['normal_transaction']
    
    def _analyze_amount_patterns(self, data):
        """Analyze transaction amount patterns"""
        amount = data.get('amount', 1000)
        
        return {
            'amount': amount,
            'amount_category': 'large' if amount > 50000 else 'medium' if amount > 5000 else 'small',
            'is_round': amount % 1000 == 0,
            'is_suspicious': amount > 100000 or (amount < 100 and amount % 10 == 0)
        }
    
    def _fallback_prediction(self, data):
        """Fallback prediction for financial risk"""
        risk_score = 0.0
        
        amount = data.get('amount', 1000)
        old_balance = data.get('oldbalanceOrg', 10000)
        
        if amount > 100000:
            risk_score += 0.4
        if amount > old_balance * 0.8:
            risk_score += 0.3
        if old_balance == 0:
            risk_score += 0.2
        if data.get('type') in ['TRANSFER', 'CASH_OUT']:
            risk_score += 0.1
        
        return {
            'financial_risk_score': min(1.0, risk_score),
            'risk_level': self._categorize_risk(risk_score),
            'risk_factors': ['rule_based_assessment'],
            'amount_analysis': self._analyze_amount_patterns(data),
            'confidence': 0.7,
            'model_used': 'fallback_rules'
        }

# Usage example
def demo_financial_prediction():
    """Demo financial risk prediction"""
    predictor = FinancialRiskPredictor()
    
    # Example transaction
    transaction = {
        'amount': 150000,
        'type': 'TRANSFER',
        'oldbalanceOrg': 200000,
        'newbalanceOrig': 50000,
        'oldbalanceDest': 50000,
        'newbalanceDest': 200000,
        'step': 150,
        'customer_txn_count': 3,
        'customer_avg_amount': 25000
    }
    
    result = predictor.predict_financial_risk(transaction)
    
    print("💰 FINANCIAL RISK PREDICTION RESULT:")
    print(f"   Risk Score: {result['financial_risk_score']:.3f}")
    print(f"   Risk Level: {result['risk_level']}")
    print(f"   Risk Factors: {', '.join(result['risk_factors'])}")
    print(f"   Amount Category: {result['amount_analysis']['amount_category']}")
    print(f"   Confidence: {result['confidence']:.3f}")
    
    return result

if __name__ == "__main__":
    demo_financial_prediction()