# ============================================================
# FINANCIAL FRAUD DETECTION SYSTEM  
# Credit risk and financial transaction fraud detection
# ============================================================

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from typing import Dict, List, Optional, Any
from collections import defaultdict, Counter
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import IsolationForest
import joblib

class FinancialFraudDetector:
    """Financial fraud detection for credit risk and transaction fraud"""
    
    def __init__(self):
        self.transaction_analyzer = TransactionAnalyzer()
        self.credit_risk_analyzer = CreditRiskAnalyzer()
        self.account_analyzer = AccountAnalyzer()
        self.model = None
        self.scaler = StandardScaler()
        
    def analyze_financial_transaction(self, transaction_data: Dict) -> Dict[str, Any]:
        """Analyze financial transaction for fraud indicators"""
        
        # Extract different types of financial features
        transaction_features = self.transaction_analyzer.extract_transaction_features(transaction_data)
        credit_features = self.credit_risk_analyzer.extract_credit_features(transaction_data)
        account_features = self.account_analyzer.extract_account_features(transaction_data)
        
        # Combine all financial features
        financial_features = {
            **transaction_features,
            **credit_features,
            **account_features
        }
        
        # Calculate financial fraud score
        fraud_score = self._calculate_financial_fraud_score(financial_features)
        
        result = {
            'transaction_id': transaction_data.get('transaction_id'),
            'user_id': transaction_data.get('user_id'),
            'financial_features': financial_features,
            'fraud_score': fraud_score,
            'risk_indicators': self._identify_financial_risks(financial_features),
            'credit_risk_level': self._assess_credit_risk(credit_features),
            'recommended_action': self._get_financial_action(fraud_score)
        }
        
        return result
    
    def _calculate_financial_fraud_score(self, features: Dict) -> float:
        """Calculate financial fraud score based on features"""
        
        score = 0.0
        
        # Amount-based risk
        amount = features.get('transaction_amount', 0)
        if amount > features.get('account_avg_transaction', 1000):
            score += 0.3
        
        # Velocity risk
        if features.get('transactions_last_hour', 0) > 5:
            score += 0.4
        
        # Credit risk
        if features.get('credit_utilization', 0) > 0.9:
            score += 0.3
        
        # Account age risk
        if features.get('account_age_days', 365) < 30:
            score += 0.2
        
        # Time-based risk
        current_hour = datetime.now().hour
        if current_hour >= 22 or current_hour <= 6:
            score += 0.1
        
        return min(score, 1.0)
    
    def _identify_financial_risks(self, features: Dict) -> List[Dict]:
        """Identify specific financial risk indicators"""
        
        risks = []
        
        # Large transaction risk
        if features.get('amount_vs_avg_ratio', 1) > 5:
            risks.append({
                'type': 'LARGE_TRANSACTION',
                'severity': 'HIGH',
                'description': 'Transaction amount significantly above average'
            })
        
        # Velocity risk
        if features.get('transactions_last_24h', 0) > 20:
            risks.append({
                'type': 'HIGH_VELOCITY',
                'severity': 'HIGH',
                'description': 'Unusually high transaction frequency'
            })
        
        # Credit limit risk
        if features.get('approaching_credit_limit', False):
            risks.append({
                'type': 'CREDIT_LIMIT_RISK',
                'severity': 'MEDIUM',
                'description': 'Transaction approaches credit limit'
            })
        
        # New merchant risk
        if features.get('new_merchant', False):
            risks.append({
                'type': 'NEW_MERCHANT',
                'severity': 'MEDIUM', 
                'description': 'First transaction with this merchant'
            })
        
        # Geo-location risk
        if features.get('geo_velocity_impossible', False):
            risks.append({
                'type': 'IMPOSSIBLE_TRAVEL',
                'severity': 'CRITICAL',
                'description': 'Geographic movement impossible given timing'
            })
        
        return risks
    
    def _assess_credit_risk(self, credit_features: Dict) -> str:
        """Assess credit risk level"""
        
        risk_score = 0
        
        if credit_features.get('credit_utilization', 0) > 0.8:
            risk_score += 3
        elif credit_features.get('credit_utilization', 0) > 0.5:
            risk_score += 1
        
        if credit_features.get('missed_payments_last_6m', 0) > 2:
            risk_score += 3
        elif credit_features.get('missed_payments_last_6m', 0) > 0:
            risk_score += 1
        
        if credit_features.get('debt_to_income_ratio', 0) > 0.4:
            risk_score += 2
        
        if risk_score >= 5:
            return "HIGH"
        elif risk_score >= 3:
            return "MEDIUM"
        elif risk_score >= 1:
            return "LOW"
        else:
            return "MINIMAL"
    
    def _get_financial_action(self, fraud_score: float) -> str:
        """Get recommended action for financial fraud"""
        
        if fraud_score >= 0.8:
            return "BLOCK_TRANSACTION"
        elif fraud_score >= 0.6:
            return "REQUIRE_MANUAL_APPROVAL"
        elif fraud_score >= 0.4:
            return "ADDITIONAL_VERIFICATION"
        elif fraud_score >= 0.2:
            return "ENHANCED_MONITORING"
        else:
            return "APPROVE"

class TransactionAnalyzer:
    """Analyze transaction patterns and characteristics"""
    
    def extract_transaction_features(self, transaction_data: Dict) -> Dict:
        """Extract transaction-level features"""
        
        features = {}
        
        # Basic transaction info
        features['transaction_amount'] = float(transaction_data.get('amount', 0))
        features['currency'] = transaction_data.get('currency', 'USD')
        features['transaction_type'] = transaction_data.get('transaction_type', 'PURCHASE')
        features['merchant_id'] = transaction_data.get('merchant_id', '')
        features['merchant_category'] = transaction_data.get('merchant_category', '')
        
        # Time-based features
        timestamp = transaction_data.get('timestamp', datetime.now())
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        
        features['hour_of_day'] = timestamp.hour
        features['day_of_week'] = timestamp.weekday()
        features['is_weekend'] = timestamp.weekday() >= 5
        features['is_night'] = timestamp.hour >= 22 or timestamp.hour <= 6
        
        # Amount analysis
        features['is_round_amount'] = features['transaction_amount'] % 1 == 0
        features['amount_log'] = np.log1p(features['transaction_amount'])
        
        # Historical comparison (requires user history)
        user_history = transaction_data.get('user_transaction_history', [])
        if user_history:
            historical_amounts = [tx.get('amount', 0) for tx in user_history]
            features['account_avg_transaction'] = np.mean(historical_amounts)
            features['account_max_transaction'] = max(historical_amounts)
            features['account_min_transaction'] = min(historical_amounts)
            features['amount_vs_avg_ratio'] = features['transaction_amount'] / max(features['account_avg_transaction'], 1)
            features['amount_zscore'] = (features['transaction_amount'] - features['account_avg_transaction']) / max(np.std(historical_amounts), 1)
        else:
            features['account_avg_transaction'] = features['transaction_amount']
            features['account_max_transaction'] = features['transaction_amount']
            features['account_min_transaction'] = features['transaction_amount']
            features['amount_vs_avg_ratio'] = 1.0
            features['amount_zscore'] = 0.0
        
        # Velocity features
        recent_transactions = [tx for tx in user_history 
                             if (timestamp - datetime.fromisoformat(tx.get('timestamp', timestamp.isoformat()))).total_seconds() <= 3600]
        features['transactions_last_hour'] = len(recent_transactions)
        
        recent_24h = [tx for tx in user_history 
                     if (timestamp - datetime.fromisoformat(tx.get('timestamp', timestamp.isoformat()))).total_seconds() <= 86400]
        features['transactions_last_24h'] = len(recent_24h)
        
        # Merchant analysis
        merchant_history = [tx for tx in user_history if tx.get('merchant_id') == features['merchant_id']]
        features['new_merchant'] = len(merchant_history) == 0
        features['merchant_transaction_count'] = len(merchant_history)
        
        # Geographic features
        location = transaction_data.get('location', {})
        features['country'] = location.get('country', '')
        features['city'] = location.get('city', '')
        features['lat'] = location.get('latitude', 0)
        features['lon'] = location.get('longitude', 0)
        
        # Check for impossible travel
        if user_history:
            last_location = user_history[-1].get('location', {})
            if last_location:
                features['geo_velocity_impossible'] = self._check_impossible_travel(
                    location, last_location, 
                    timestamp, datetime.fromisoformat(user_history[-1].get('timestamp', timestamp.isoformat()))
                )
            else:
                features['geo_velocity_impossible'] = False
        else:
            features['geo_velocity_impossible'] = False
        
        return features
    
    def _check_impossible_travel(self, current_loc: Dict, prev_loc: Dict, 
                               current_time: datetime, prev_time: datetime) -> bool:
        """Check if travel between locations is physically impossible"""
        
        # Calculate distance between locations (simplified)
        lat1, lon1 = prev_loc.get('latitude', 0), prev_loc.get('longitude', 0)
        lat2, lon2 = current_loc.get('latitude', 0), current_loc.get('longitude', 0)
        
        # Haversine distance (approximate)
        R = 6371  # Earth's radius in km
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
        distance_km = 2 * R * np.arcsin(np.sqrt(a))
        
        # Calculate time difference in hours
        time_diff_hours = (current_time - prev_time).total_seconds() / 3600
        
        # Check if travel speed > 1000 km/h (impossible for normal travel)
        if time_diff_hours > 0:
            speed_kmh = distance_km / time_diff_hours
            return speed_kmh > 1000
        
        return False

class CreditRiskAnalyzer:
    """Analyze credit risk factors"""
    
    def extract_credit_features(self, transaction_data: Dict) -> Dict:
        """Extract credit risk features"""
        
        features = {}
        
        # Credit account information
        credit_info = transaction_data.get('credit_info', {})
        
        features['credit_limit'] = credit_info.get('credit_limit', 0)
        features['current_balance'] = credit_info.get('current_balance', 0)
        features['available_credit'] = features['credit_limit'] - features['current_balance']
        
        # Credit utilization
        if features['credit_limit'] > 0:
            features['credit_utilization'] = features['current_balance'] / features['credit_limit']
        else:
            features['credit_utilization'] = 0
        
        # Check if transaction approaches credit limit
        transaction_amount = float(transaction_data.get('amount', 0))
        features['approaching_credit_limit'] = (features['current_balance'] + transaction_amount) > (features['credit_limit'] * 0.9)
        
        # Payment history
        payment_history = credit_info.get('payment_history', [])
        features['missed_payments_last_6m'] = sum(1 for payment in payment_history[-6:] 
                                                 if payment.get('status') == 'MISSED')
        features['late_payments_last_6m'] = sum(1 for payment in payment_history[-6:] 
                                               if payment.get('status') == 'LATE')
        
        # Credit age
        account_opened = credit_info.get('account_opened_date')
        if account_opened:
            if isinstance(account_opened, str):
                account_opened = datetime.fromisoformat(account_opened)
            features['credit_account_age_months'] = (datetime.now() - account_opened).days / 30
        else:
            features['credit_account_age_months'] = 0
        
        # Income and debt information
        financial_info = transaction_data.get('financial_info', {})
        features['monthly_income'] = financial_info.get('monthly_income', 0)
        features['total_debt'] = financial_info.get('total_debt', 0)
        
        if features['monthly_income'] > 0:
            features['debt_to_income_ratio'] = features['total_debt'] / features['monthly_income']
        else:
            features['debt_to_income_ratio'] = 0
        
        # Credit score (if available)
        features['credit_score'] = credit_info.get('credit_score', 0)
        
        return features

class AccountAnalyzer:
    """Analyze account characteristics and behavior"""
    
    def extract_account_features(self, transaction_data: Dict) -> Dict:
        """Extract account-level features"""
        
        features = {}
        
        # Account basic info
        account_info = transaction_data.get('account_info', {})
        
        features['account_type'] = account_info.get('account_type', 'UNKNOWN')
        features['account_status'] = account_info.get('status', 'ACTIVE')
        
        # Account age
        account_created = account_info.get('created_date')
        if account_created:
            if isinstance(account_created, str):
                account_created = datetime.fromisoformat(account_created)
            features['account_age_days'] = (datetime.now() - account_created).days
        else:
            features['account_age_days'] = 0
        
        # Verification status
        features['email_verified'] = account_info.get('email_verified', False)
        features['phone_verified'] = account_info.get('phone_verified', False)
        features['identity_verified'] = account_info.get('identity_verified', False)
        features['verification_score'] = sum([
            features['email_verified'],
            features['phone_verified'],
            features['identity_verified']
        ]) / 3.0
        
        # Account activity
        activity_info = account_info.get('activity', {})
        features['total_transactions'] = activity_info.get('total_transactions', 0)
        features['total_amount_transacted'] = activity_info.get('total_amount', 0)
        features['days_since_last_login'] = activity_info.get('days_since_last_login', 0)
        features['login_frequency_per_week'] = activity_info.get('logins_per_week', 0)
        
        # Risk flags
        flags = account_info.get('risk_flags', [])
        features['has_chargeback_history'] = 'CHARGEBACK' in flags
        features['has_fraud_history'] = 'FRAUD' in flags
        features['has_suspicious_activity'] = 'SUSPICIOUS' in flags
        features['total_risk_flags'] = len(flags)
        
        # Linked accounts and devices
        features['linked_accounts_count'] = len(account_info.get('linked_accounts', []))
        features['registered_devices_count'] = len(account_info.get('registered_devices', []))
        
        # Calculate account trustworthiness score
        trust_score = 0.0
        
        # Age factor
        if features['account_age_days'] > 365:
            trust_score += 0.3
        elif features['account_age_days'] > 90:
            trust_score += 0.2
        
        # Verification factor
        trust_score += features['verification_score'] * 0.3
        
        # Activity factor
        if features['total_transactions'] > 50:
            trust_score += 0.2
        elif features['total_transactions'] > 10:
            trust_score += 0.1
        
        # Risk flags factor
        trust_score -= min(features['total_risk_flags'] * 0.1, 0.3)
        
        features['account_trustworthiness_score'] = max(0.0, min(1.0, trust_score))
        
        return features

# ============================================================
# FINANCIAL FRAUD MODEL
# ============================================================

class FinancialFraudModel:
    """Machine learning model for financial fraud detection"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.label_encoders = {}
    
    def train(self, financial_features: pd.DataFrame, fraud_labels: pd.Series):
        """Train financial fraud detection model"""
        
        # Handle categorical features
        categorical_features = ['currency', 'transaction_type', 'merchant_category', 
                               'country', 'account_type', 'account_status']
        
        X = financial_features.copy()
        
        for col in categorical_features:
            if col in X.columns:
                self.label_encoders[col] = LabelEncoder()
                X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
        
        # Handle boolean features
        bool_columns = X.select_dtypes(include=['bool']).columns
        X[bool_columns] = X[bool_columns].astype(int)
        
        # Prepare feature columns
        self.feature_columns = [col for col in X.columns 
                               if col not in ['transaction_id', 'user_id']]
        
        X_processed = X[self.feature_columns].fillna(0)
        X_scaled = self.scaler.fit_transform(X_processed)
        
        # Train model
        self.model = lgb.LGBMClassifier(
            objective='binary',
            n_estimators=300,
            max_depth=10,
            learning_rate=0.05,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42
        )
        
        self.model.fit(X_scaled, fraud_labels)
        
        return self
    
    def predict_fraud(self, financial_features: pd.DataFrame) -> List[Dict]:
        """Predict financial fraud probability"""
        
        if self.model is None:
            raise ValueError("Model not trained")
        
        X = financial_features.copy()
        
        # Apply same preprocessing
        categorical_features = ['currency', 'transaction_type', 'merchant_category',
                               'country', 'account_type', 'account_status']
        
        for col in categorical_features:
            if col in X.columns and col in self.label_encoders:
                X[col] = self.label_encoders[col].transform(X[col].astype(str))
        
        bool_columns = X.select_dtypes(include=['bool']).columns
        X[bool_columns] = X[bool_columns].astype(int)
        
        X_processed = X[self.feature_columns].fillna(0)
        X_scaled = self.scaler.transform(X_processed)
        
        fraud_probabilities = self.model.predict_proba(X_scaled)[:, 1]
        
        results = []
        for i, row in financial_features.iterrows():
            prob = fraud_probabilities[i]
            results.append({
                'transaction_id': row.get('transaction_id'),
                'user_id': row.get('user_id'),
                'financial_fraud_probability': prob,
                'risk_level': self._assess_risk_level(prob),
                'recommended_action': self._get_recommended_action(prob)
            })
        
        return results
    
    def _assess_risk_level(self, prob: float) -> str:
        """Assess financial fraud risk level"""
        if prob >= 0.8:
            return "CRITICAL"
        elif prob >= 0.6:
            return "HIGH"
        elif prob >= 0.4:
            return "MEDIUM"
        elif prob >= 0.2:
            return "LOW"
        else:
            return "MINIMAL"
    
    def _get_recommended_action(self, prob: float) -> str:
        """Get recommended action for financial fraud"""
        if prob >= 0.8:
            return "BLOCK_TRANSACTION"
        elif prob >= 0.6:
            return "MANUAL_REVIEW"
        elif prob >= 0.4:
            return "ADDITIONAL_AUTH"
        elif prob >= 0.2:
            return "MONITOR"
        else:
            return "APPROVE"

# ============================================================
# USAGE EXAMPLE
# ============================================================

def demo_financial_fraud_detection():
    """Demonstrate financial fraud detection"""
    
    print("💳 Financial Fraud Detection System")
    print("="*50)
    
    # Initialize detector
    detector = FinancialFraudDetector()
    
    # Example transaction data
    transaction_data = {
        'transaction_id': 'TXN_12345',
        'user_id': 'USER_67890',
        'amount': 2500.00,
        'currency': 'USD',
        'transaction_type': 'PURCHASE',
        'merchant_id': 'MERCHANT_999',
        'merchant_category': 'ELECTRONICS',
        'timestamp': '2026-02-20T23:45:00',
        'location': {
            'country': 'US',
            'city': 'New York',
            'latitude': 40.7128,
            'longitude': -74.0060
        },
        'credit_info': {
            'credit_limit': 5000.00,
            'current_balance': 1200.00,
            'credit_score': 720,
            'account_opened_date': '2020-01-15'
        },
        'account_info': {
            'account_type': 'PREMIUM',
            'status': 'ACTIVE',
            'created_date': '2019-05-10',
            'email_verified': True,
            'phone_verified': True,
            'identity_verified': False,
            'risk_flags': []
        },
        'user_transaction_history': [
            {'amount': 150.00, 'timestamp': '2026-02-20T10:00:00', 'merchant_id': 'MERCHANT_123'},
            {'amount': 75.50, 'timestamp': '2026-02-19T15:30:00', 'merchant_id': 'MERCHANT_456'}
        ]
    }
    
    # Analyze transaction
    result = detector.analyze_financial_transaction(transaction_data)
    
    print(f"💰 Transaction: {result['transaction_id']}")
    print(f"👤 User: {result['user_id']}")
    print(f"🎯 Fraud Score: {result['fraud_score']:.3f}")
    print(f"⚠️  Risk Indicators: {len(result['risk_indicators'])}")
    print(f"💳 Credit Risk: {result['credit_risk_level']}")
    print(f"🎬 Recommended Action: {result['recommended_action']}")
    
    print(f"\n📊 Key Financial Features:")
    features = result['financial_features']
    print(f"   💵 Amount vs Avg Ratio: {features.get('amount_vs_avg_ratio', 0):.2f}")
    print(f"   📈 Credit Utilization: {features.get('credit_utilization', 0):.2%}")
    print(f"   🏦 Account Age: {features.get('account_age_days', 0)} days")
    print(f"   🔄 Transactions (24h): {features.get('transactions_last_24h', 0)}")
    
    if result['risk_indicators']:
        print(f"\n⚠️  Risk Indicators:")
        for risk in result['risk_indicators']:
            print(f"   • {risk['type']}: {risk['description']} ({risk['severity']})")
    
    print(f"\n💳 Financial Fraud Detection Ready!")

if __name__ == "__main__":
    demo_financial_fraud_detection()