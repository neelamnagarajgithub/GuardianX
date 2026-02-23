# ============================================================
# NETWORK SECURITY INTELLIGENCE - USAGE SCRIPT
# Load and use network security model
# ============================================================

import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from pathlib import Path

class NetworkSecurityPredictor:
    """Production-ready network security threat prediction system"""
    
    def __init__(self, models_path="artifacts"):
        self.models_path = Path(models_path)
        self.model = None
        self.scaler = None
        self.encoders = {}
        self.feature_columns = None
        self.load_model()
    
    def load_model(self):
        """Load network security model"""
        try:
            # Try to find your existing lgbm_model.txt first
            lgbm_path = self.models_path / "models" / "lgbm_model.txt"
            if lgbm_path.exists():
                self.model = lgb.Booster(model_file=str(lgbm_path))
                print(f"✅ Advanced network model loaded from {lgbm_path}")
                return
            
            # Find security model files
            for root, dirs, files in os.walk(self.models_path):
                for file in files:
                    if ('security' in file.lower() or 'network' in file.lower()) and file.endswith(('.pkl', '.joblib', '.txt')):
                        file_path = Path(root) / file
                        
                        if file.endswith('.txt') and 'lgb' in file.lower():
                            self.model = lgb.Booster(model_file=str(file_path))
                            print(f"✅ LightGBM security model loaded from {file_path}")
                        elif 'model' in file:
                            self.model = joblib.load(file_path)
                            print(f"✅ Security model loaded from {file_path}")
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
                print("⚠️ No security model found, using fallback prediction")
                
        except Exception as e:
            print(f"❌ Error loading security model: {e}")
    
    def predict_security_threat(self, transaction_data):
        """Predict network security threat for a transaction"""
        try:
            if self.model is None:
                return self._fallback_prediction(transaction_data)
            
            # Extract security features
            features = self._extract_security_features(transaction_data)
            
            # Scale features if scaler available
            if self.scaler:
                features_scaled = self.scaler.transform([features])
            else:
                features_scaled = [features]
            
            # Get prediction
            if hasattr(self.model, 'predict_proba'):
                threat_score = self.model.predict_proba(features_scaled)[0][1]
            elif hasattr(self.model, 'predict'):
                threat_score = self.model.predict(features_scaled)[0]
                if isinstance(threat_score, np.ndarray):
                    threat_score = threat_score[0]
                # Normalize LightGBM output to 0-1 range if needed
                if threat_score > 1 or threat_score < 0:
                    threat_score = 1 / (1 + np.exp(-threat_score))  # Sigmoid
            else:
                threat_score = 0.5
            
            return {
                'security_threat_score': float(threat_score),
                'threat_level': self._categorize_threat(threat_score),
                'attack_patterns': self._detect_attack_patterns(transaction_data),
                'bot_indicators': self._detect_bot_behavior(transaction_data),
                'network_anomalies': self._detect_network_anomalies(transaction_data),
                'confidence': min(0.95, max(0.6, abs(threat_score - 0.5) * 2)),
                'model_used': 'network_security_intelligence'
            }
            
        except Exception as e:
            print(f"⚠️ Error in security prediction: {e}")
            return self._fallback_prediction(transaction_data)
    
    def _extract_security_features(self, data):
        """Extract network security features from transaction data"""
        features = []
        
        # Basic transaction features
        amount = data.get('amount', 1000)
        step = data.get('step', 100)
        
        features.append(amount)  # amount
        features.append(np.log1p(amount))  # amount_log
        features.append(step)  # step
        
        # Bot detection features
        features.append(float(amount < 1))  # micro_transaction_attack
        features.append(float(amount % 100 == 0))  # round_amount_pattern
        features.append(float(amount == 0.01))  # penny_transaction
        
        # Temporal attack features
        hour = step % 24
        features.append(hour)  # transaction_hour
        features.append(float(hour < 6 or hour > 22))  # night_attack
        features.append(float((step // 24) % 7 >= 5))  # weekend_attack
        
        # Account security features
        old_balance = data.get('oldbalanceOrg', 10000)
        new_balance = data.get('newbalanceOrig', old_balance - amount)
        features.append(old_balance)  # oldbalanceOrg
        features.append(new_balance)  # newbalanceOrig
        features.append(float(old_balance == 0))  # zero_balance_creation
        features.append(float(old_balance == 0 and amount > 10000))  # instant_large_transaction
        
        # Balance manipulation detection
        expected_balance = old_balance - amount
        features.append(float(abs(expected_balance - new_balance) > 0.01))  # balance_inconsistency
        features.append(float(old_balance > 0 and new_balance == 0))  # emptied_account
        
        # Transaction type security
        txn_type = data.get('type', 'PAYMENT')
        type_mapping = {'PAYMENT': 0, 'TRANSFER': 1, 'CASH_OUT': 2, 'DEBIT': 3, 'CASH_IN': 4}
        encoded_type = type_mapping.get(txn_type, 0)
        features.append(encoded_type)  # type_encoded
        features.append(float(txn_type in ['TRANSFER', 'CASH_OUT']))  # high_risk_type
        
        # Network behavior indicators
        features.append(data.get('user_frequency', 5))  # transaction_frequency
        features.append(data.get('destination_popularity', 0.1))  # destination_risk
        
        # Velocity and burst patterns
        features.append(data.get('velocity_score', 5) / 10)  # normalized_velocity
        features.append(float(data.get('burst_detected', False)))  # burst_pattern
        
        # Pad to expected feature count (use your model's expected feature count)
        expected_count = len(self.feature_columns) if self.feature_columns else 20
        while len(features) < expected_count:
            features.append(0.0)
        
        return features[:expected_count]
    
    def _categorize_threat(self, score):
        """Categorize security threat score"""
        if score > 0.9:
            return 'CRITICAL'
        elif score > 0.75:
            return 'HIGH'
        elif score > 0.6:
            return 'MEDIUM'
        elif score > 0.4:
            return 'LOW'
        else:
            return 'MINIMAL'
    
    def _detect_attack_patterns(self, data):
        """Detect specific attack patterns"""
        patterns = []
        
        amount = data.get('amount', 1000)
        
        if amount < 1:
            patterns.append('card_testing')
        if amount % 100 == 0 and amount > 1000:
            patterns.append('automation_signature')
        if data.get('type') == 'TRANSFER' and amount > 100000:
            patterns.append('large_transfer_attack')
        if data.get('oldbalanceOrg', 0) == 0:
            patterns.append('account_creation_fraud')
        
        return patterns[:3] if patterns else ['no_attack_detected']
    
    def _detect_bot_behavior(self, data):
        """Detect bot behavior indicators"""
        indicators = []
        
        amount = data.get('amount', 1000)
        
        if amount == round(amount) and amount % 100 == 0:
            indicators.append('exact_amounts')
        if data.get('user_frequency', 5) > 20:
            indicators.append('high_frequency')
        if data.get('burst_detected', False):
            indicators.append('burst_activity')
        
        return indicators[:3] if indicators else ['human_behavior']
    
    def _detect_network_anomalies(self, data):
        """Detect network-level anomalies"""
        anomalies = []
        
        if data.get('destination_popularity', 0.1) > 0.8:
            anomalies.append('popular_destination')
        if data.get('velocity_score', 5) > 8:
            anomalies.append('velocity_anomaly')
        if data.get('geo_anomaly', False):
            anomalies.append('geographic_anomaly')
        
        return anomalies[:3] if anomalies else ['normal_network_behavior']
    
    def _fallback_prediction(self, data):
        """Fallback prediction for security threats"""
        threat_score = 0.0
        
        amount = data.get('amount', 1000)
        
        # Rule-based threat scoring
        if amount < 1:
            threat_score += 0.3  # Micro transactions
        if amount % 100 == 0:
            threat_score += 0.2  # Round amounts
        if data.get('oldbalanceOrg', 1000) == 0:
            threat_score += 0.3  # Zero balance accounts
        if data.get('type') in ['TRANSFER', 'CASH_OUT']:
            threat_score += 0.2  # High-risk types
        
        return {
            'security_threat_score': min(1.0, threat_score),
            'threat_level': self._categorize_threat(threat_score),
            'attack_patterns': ['rule_based_detection'],
            'bot_indicators': ['basic_pattern_matching'],
            'network_anomalies': ['heuristic_analysis'],
            'confidence': 0.7,
            'model_used': 'fallback_rules'
        }

# Usage example
def demo_security_prediction():
    """Demo network security threat prediction"""
    predictor = NetworkSecurityPredictor()
    
    # Example suspicious transaction
    transaction = {
        'amount': 0.50,  # Micro transaction
        'type': 'TRANSFER',
        'oldbalanceOrg': 0,  # Zero balance
        'newbalanceOrig': 0,
        'step': 150,  # Night time
        'user_frequency': 25,  # High frequency
        'velocity_score': 9.2,  # High velocity
        'burst_detected': True,
        'destination_popularity': 0.9
    }
    
    result = predictor.predict_security_threat(transaction)
    
    print("🔒 NETWORK SECURITY THREAT PREDICTION RESULT:")
    print(f"   Threat Score: {result['security_threat_score']:.3f}")
    # ============================================================
# NETWORK SECURITY INTELLIGENCE - USAGE SCRIPT
# Load and use network security model
# ============================================================

import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from pathlib import Path

class NetworkSecurityPredictor:
    """Production-ready network security threat prediction system"""
    
    def __init__(self, models_path="artifacts"):
        self.models_path = Path(models_path)
        self.model = None
        self.scaler = None
        self.encoders = {}
        self.feature_columns = None
        self.load_model()
    
    def load_model(self):
        """Load network security model"""
        try:
            # Try to find your existing lgbm_model.txt first
            lgbm_path = self.models_path / "models" / "lgbm_model.txt"
            if lgbm_path.exists():
                self.model = lgb.Booster(model_file=str(lgbm_path))
                print(f"✅ Advanced network model loaded from {lgbm_path}")
                return
            
            # Find security model files
            for root, dirs, files in os.walk(self.models_path):
                for file in files:
                    if ('security' in file.lower() or 'network' in file.lower()) and file.endswith(('.pkl', '.joblib', '.txt')):
                        file_path = Path(root) / file
                        
                        if file.endswith('.txt') and 'lgb' in file.lower():
                            self.model = lgb.Booster(model_file=str(file_path))
                            print(f"✅ LightGBM security model loaded from {file_path}")
                        elif 'model' in file:
                            self.model = joblib.load(file_path)
                            print(f"✅ Security model loaded from {file_path}")
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
                print("⚠️ No security model found, using fallback prediction")
                
        except Exception as e:
            print(f"❌ Error loading security model: {e}")
    
    def predict_security_threat(self, transaction_data):
        """Predict network security threat for a transaction"""
        try:
            if self.model is None:
                return self._fallback_prediction(transaction_data)
            
            # Extract security features
            features = self._extract_security_features(transaction_data)
            
            # Scale features if scaler available
            if self.scaler:
                features_scaled = self.scaler.transform([features])
            else:
                features_scaled = [features]
            
            # Get prediction
            if hasattr(self.model, 'predict_proba'):
                threat_score = self.model.predict_proba(features_scaled)[0][1]
            elif hasattr(self.model, 'predict'):
                threat_score = self.model.predict(features_scaled)[0]
                if isinstance(threat_score, np.ndarray):
                    threat_score = threat_score[0]
                # Normalize LightGBM output to 0-1 range if needed
                if threat_score > 1 or threat_score < 0:
                    threat_score = 1 / (1 + np.exp(-threat_score))  # Sigmoid
            else:
                threat_score = 0.5
            
            return {
                'security_threat_score': float(threat_score),
                'threat_level': self._categorize_threat(threat_score),
                'attack_patterns': self._detect_attack_patterns(transaction_data),
                'bot_indicators': self._detect_bot_behavior(transaction_data),
                'network_anomalies': self._detect_network_anomalies(transaction_data),
                'confidence': min(0.95, max(0.6, abs(threat_score - 0.5) * 2)),
                'model_used': 'network_security_intelligence'
            }
            
        except Exception as e:
            print(f"⚠️ Error in security prediction: {e}")
            return self._fallback_prediction(transaction_data)
    
    def _extract_security_features(self, data):
        """Extract network security features from transaction data"""
        features = []
        
        # Basic transaction features
        amount = data.get('amount', 1000)
        step = data.get('step', 100)
        
        features.append(amount)  # amount
        features.append(np.log1p(amount))  # amount_log
        features.append(step)  # step
        
        # Bot detection features
        features.append(float(amount < 1))  # micro_transaction_attack
        features.append(float(amount % 100 == 0))  # round_amount_pattern
        features.append(float(amount == 0.01))  # penny_transaction
        
        # Temporal attack features
        hour = step % 24
        features.append(hour)  # transaction_hour
        features.append(float(hour < 6 or hour > 22))  # night_attack
        features.append(float((step // 24) % 7 >= 5))  # weekend_attack
        
        # Account security features
        old_balance = data.get('oldbalanceOrg', 10000)
        new_balance = data.get('newbalanceOrig', old_balance - amount)
        features.append(old_balance)  # oldbalanceOrg
        features.append(new_balance)  # newbalanceOrig
        features.append(float(old_balance == 0))  # zero_balance_creation
        features.append(float(old_balance == 0 and amount > 10000))  # instant_large_transaction
        
        # Balance manipulation detection
        expected_balance = old_balance - amount
        features.append(float(abs(expected_balance - new_balance) > 0.01))  # balance_inconsistency
        features.append(float(old_balance > 0 and new_balance == 0))  # emptied_account
        
        # Transaction type security
        txn_type = data.get('type', 'PAYMENT')
        type_mapping = {'PAYMENT': 0, 'TRANSFER': 1, 'CASH_OUT': 2, 'DEBIT': 3, 'CASH_IN': 4}
        encoded_type = type_mapping.get(txn_type, 0)
        features.append(encoded_type)  # type_encoded
        features.append(float(txn_type in ['TRANSFER', 'CASH_OUT']))  # high_risk_type
        
        # Network behavior indicators
        features.append(data.get('user_frequency', 5))  # transaction_frequency
        features.append(data.get('destination_popularity', 0.1))  # destination_risk
        
        # Velocity and burst patterns
        features.append(data.get('velocity_score', 5) / 10)  # normalized_velocity
        features.append(float(data.get('burst_detected', False)))  # burst_pattern
        
        # Pad to expected feature count (use your model's expected feature count)
        expected_count = len(self.feature_columns) if self.feature_columns else 20
        while len(features) < expected_count:
            features.append(0.0)
        
        return features[:expected_count]
    
    def _categorize_threat(self, score):
        """Categorize security threat score"""
        if score > 0.9:
            return 'CRITICAL'
        elif score > 0.75:
            return 'HIGH'
        elif score > 0.6:
            return 'MEDIUM'
        elif score > 0.4:
            return 'LOW'
        else:
            return 'MINIMAL'
    
    def _detect_attack_patterns(self, data):
        """Detect specific attack patterns"""
        patterns = []
        
        amount = data.get('amount', 1000)
        
        if amount < 1:
            patterns.append('card_testing')
        if amount % 100 == 0 and amount > 1000:
            patterns.append('automation_signature')
        if data.get('type') == 'TRANSFER' and amount > 100000:
            patterns.append('large_transfer_attack')
        if data.get('oldbalanceOrg', 0) == 0:
            patterns.append('account_creation_fraud')
        
        return patterns[:3] if patterns else ['no_attack_detected']
    
    def _detect_bot_behavior(self, data):
        """Detect bot behavior indicators"""
        indicators = []
        
        amount = data.get('amount', 1000)
        
        if amount == round(amount) and amount % 100 == 0:
            indicators.append('exact_amounts')
        if data.get('user_frequency', 5) > 20:
            indicators.append('high_frequency')
        if data.get('burst_detected', False):
            indicators.append('burst_activity')
        
        return indicators[:3] if indicators else ['human_behavior']
    
    def _detect_network_anomalies(self, data):
        """Detect network-level anomalies"""
        anomalies = []
        
        if data.get('destination_popularity', 0.1) > 0.8:
            anomalies.append('popular_destination')
        if data.get('velocity_score', 5) > 8:
            anomalies.append('velocity_anomaly')
        if data.get('geo_anomaly', False):
            anomalies.append('geographic_anomaly')
        
        return anomalies[:3] if anomalies else ['normal_network_behavior']
    
    def _fallback_prediction(self, data):
        """Fallback prediction for security threats"""
        threat_score = 0.0
        
        amount = data.get('amount', 1000)
        
        # Rule-based threat scoring
        if amount < 1:
            threat_score += 0.3  # Micro transactions
        if amount % 100 == 0:
            threat_score += 0.2  # Round amounts
        if data.get('oldbalanceOrg', 1000) == 0:
            threat_score += 0.3  # Zero balance accounts
        if data.get('type') in ['TRANSFER', 'CASH_OUT']:
            threat_score += 0.2  # High-risk types
        
        return {
            'security_threat_score': min(1.0, threat_score),
            'threat_level': self._categorize_threat(threat_score),
            'attack_patterns': ['rule_based_detection'],
            'bot_indicators': ['basic_pattern_matching'],
            'network_anomalies': ['heuristic_analysis'],
            'confidence': 0.7,
            'model_used': 'fallback_rules'
        }

# Usage example
def demo_security_prediction():
    """Demo network security threat prediction"""
    predictor = NetworkSecurityPredictor()
    
    # Example suspicious transaction
    transaction = {
        'amount': 0.50,  # Micro transaction
        'type': 'TRANSFER',
        'oldbalanceOrg': 0,  # Zero balance
        'newbalanceOrig': 0,
        'step': 150,  # Night time
        'user_frequency': 25,  # High frequency
        'velocity_score': 9.2,  # High velocity
        'burst_detected': True,
        'destination_popularity': 0.9
    }
    
    result = predictor.predict_security_threat(transaction)
    
    print("🔒 NETWORK SECURITY THREAT PREDICTION RESULT:")
    print(f"   Threat Score: {result['security_threat_score']:.3f}")
    