# ============================================================
# COMPREHENSIVE MODEL EVALUATION SYSTEM
# Evaluate all 3 models with detailed performance metrics
# ============================================================

import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from sklearn.metrics import (
    classification_report, roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    """Comprehensive evaluation system for all fraud detection models"""
    
    def __init__(self, artifacts_path="models"):
        self.artifacts_path = Path(artifacts_path)
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.test_results = {}
    
    def load_all_models(self):
        """Load all available models from artifacts"""
        print("🚀 LOADING ALL MODELS FOR EVALUATION")
        print("=" * 60)
        
        # Try to load behavioral model
        try:
            behavioral_files = self._find_model_files('behavioral')
            if behavioral_files:
                self.models['behavioral'] = self._load_model_set(behavioral_files, 'behavioral')
                print("✅ Behavioral Fraud Model loaded")
        except Exception as e:
            print(f"❌ Failed to load behavioral model: {e}")
        
        # Try to load financial model
        try:
            financial_files = self._find_model_files('financial')
            if financial_files:
                self.models['financial'] = self._load_model_set(financial_files, 'financial')
                print("✅ Financial Credit Risk Model loaded")
        except Exception as e:
            print(f"❌ Failed to load financial model: {e}")
        
        # Try to load security model
        try:
            security_files = self._find_model_files('security')
            if security_files:
                self.models['security'] = self._load_model_set(security_files, 'security')
                print("✅ Network Security Model loaded")
        except Exception as e:
            print(f"❌ Failed to load security model: {e}")
        
        # Try to load your existing advanced model
        try:
            lgbm_path = self.artifacts_path / "models" / "lgbm_model.txt"
            if lgbm_path.exists():
                self.models['advanced_network'] = lgb.Booster(model_file=str(lgbm_path))
                print("✅ Advanced Network Model (lgbm_model.txt) loaded")
        except Exception as e:
            print(f"❌ Failed to load advanced model: {e}")
        
        print(f"\n📊 Total models loaded: {len(self.models)}")
        return len(self.models) > 0
    
    def _find_model_files(self, model_type):
        """Find model files for a specific type"""
        files = {}
        
        # Search patterns for different model files
        patterns = {
            'behavioral': ['behavioral_model', 'behavioral_scaler', 'behavioral_features', 'behavioral_encoders'],
            'financial': ['financial_model', 'financial_scaler', 'financial_features', 'financial_encoders'],
            'security': ['security_model', 'security_scaler', 'security_features', 'security_encoders']
        }
        
        if model_type not in patterns:
            return None
        
        for root, dirs, file_names in os.walk(self.artifacts_path):
            root_path = Path(root)
            
            for pattern in patterns[model_type]:
                for file_name in file_names:
                    if pattern in file_name.lower():
                        files[pattern] = root_path / file_name
        
        return files if files else None
    
    def _load_model_set(self, files, model_type):
        """Load complete model set (model + preprocessing)"""
        model_set = {}
        
        # Load main model
        model_file = files.get(f'{model_type}_model')
        if model_file and model_file.exists():
            if model_file.suffix == '.txt':
                model_set['model'] = lgb.Booster(model_file=str(model_file))
            else:
                model_set['model'] = joblib.load(model_file)
        
        # Load scaler
        scaler_file = files.get(f'{model_type}_scaler')
        if scaler_file and scaler_file.exists():
            model_set['scaler'] = joblib.load(scaler_file)
        
        # Load feature names
        features_file = files.get(f'{model_type}_features')
        if features_file and features_file.exists():
            model_set['features'] = joblib.load(features_file)
        
        # Load encoders
        encoders_file = files.get(f'{model_type}_encoders')
        if encoders_file and encoders_file.exists():
            model_set['encoders'] = joblib.load(encoders_file)
        
        return model_set
    
    def prepare_test_data(self):
        """Prepare test datasets for each model type"""
        print("\n📊 PREPARING TEST DATASETS")
        print("=" * 40)
        
        test_datasets = {}
        
        # Load Nigerian dataset for behavioral testing
        try:
            nigerian_path = self.artifacts_path / "raw" / "nigerian_sample.parquet"
            if nigerian_path.exists():
                df = pd.read_parquet(nigerian_path)
                df = self._clean_data(df)
                if 'is_fraud' not in df.columns:
                    df['is_fraud'] = self._create_synthetic_labels(df)
                test_datasets['behavioral'] = df.sample(n=min(5000, len(df)), random_state=42)
                print(f"✅ Behavioral test data: {len(test_datasets['behavioral'])} samples")
        except Exception as e:
            print(f"❌ Failed to load behavioral test data: {e}")
        
        # Load PaySim dataset for financial testing
        try:
            paysim_path = self.artifacts_path / "raw" / "paysim.parquet"
            if paysim_path.exists():
                df = pd.read_parquet(paysim_path)
                df = self._clean_data(df)
                if 'isFraud' in df.columns:
                    df['is_fraud'] = df['isFraud']
                elif 'is_fraud' not in df.columns:
                    df['is_fraud'] = self._create_synthetic_labels(df)
                test_datasets['financial'] = df.sample(n=min(5000, len(df)), random_state=42)
                print(f"✅ Financial test data: {len(test_datasets['financial'])} samples")
        except Exception as e:
            print(f"❌ Failed to load financial test data: {e}")
        
        # Load CIFER dataset for security testing
        try:
            cifer_path = self.artifacts_path / "raw" / "cifer_sample.parquet"
            if cifer_path.exists():
                df = pd.read_parquet(cifer_path)
                df = self._clean_data(df)
                if 'is_fraud' not in df.columns:
                    df['is_fraud'] = self._create_synthetic_labels(df)
                test_datasets['security'] = df.sample(n=min(5000, len(df)), random_state=42)
                print(f"✅ Security test data: {len(test_datasets['security'])} samples")
        except Exception as e:
            print(f"❌ Failed to load security test data: {e}")
        
        return test_datasets
    
    def _clean_data(self, df):
        """Clean dataset for evaluation"""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            if df[col].notna().sum() > 0:
                cap_value = df[col].quantile(0.999)
                df[col] = df[col].clip(upper=cap_value)
        return df.fillna(0)
    
    def _create_synthetic_labels(self, df):
        """Create synthetic fraud labels for evaluation"""
        # Create labels based on suspicious patterns
        suspicious = (
            (df.get('amount', df.get('amount_ngn', 0)) > df.get('amount', df.get('amount_ngn', 0)).quantile(0.95)) |
            (df.get('velocity_score', 5) > 8) |
            (df.get('spending_deviation_score', 0.5) > 0.8) |
            (df.get('is_night_txn', False) == True)
        )
        return suspicious.astype(int)
    
    def evaluate_model(self, model_name, model_set, test_data):
        """Evaluate a single model comprehensively"""
        print(f"\n🎯 EVALUATING {model_name.upper()} MODEL")
        print("=" * 50)
        
        try:
            # Prepare features based on model type
            if model_name == 'behavioral':
                X, y = self._prepare_behavioral_features(test_data)
            elif model_name == 'financial':
                X, y = self._prepare_financial_features(test_data)
            elif model_name == 'security':
                X, y = self._prepare_security_features(test_data)
            else:
                # For advanced network model, use basic features
                feature_cols = ['amount', 'step', 'oldbalanceOrg', 'newbalanceOrig']
                available_cols = [c for c in feature_cols if c in test_data.columns]
                X = test_data[available_cols] if available_cols else test_data.select_dtypes(include=[np.number]).iloc[:, :10]
                y = test_data['is_fraud'] if 'is_fraud' in test_data.columns else test_data['isFraud']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            
            # Preprocess if scaler available
            if 'scaler' in model_set and model_set['scaler'] is not None:
                X_test_scaled = model_set['scaler'].transform(X_test)
            else:
                X_test_scaled = X_test.values if hasattr(X_test, 'values') else X_test
            
            # Get predictions
            model = model_set['model'] if 'model' in model_set else model_set
            
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                y_pred = (y_pred_proba > 0.5).astype(int)
            elif hasattr(model, 'predict'):
                if str(type(model)).find('lightgbm') != -1:
                    y_pred_proba = model.predict(X_test_scaled)
                    y_pred = (y_pred_proba > 0.5).astype(int)
                else:
                    y_pred_proba = model.predict(X_test_scaled)
                    y_pred = (y_pred_proba > 0.5).astype(int)
            else:
                raise ValueError("Model doesn't have predict method")
            
            # Calculate metrics
            results = self._calculate_metrics(y_test, y_pred, y_pred_proba, model_name)
            
            # Store results
            self.test_results[model_name] = results
            
            # Print results
            self._print_results(model_name, results)
            
            return results
            
        except Exception as e:
            print(f"❌ Error evaluating {model_name}: {e}")
            return None
    
    def _prepare_behavioral_features(self, df):
        """Prepare behavioral features for evaluation"""
        # Simple behavioral features that should exist
        features = []
        
        # Device and behavioral features
        if 'device_seen_count' in df.columns:
            features.append(1 / (df['device_seen_count'] + 1))  # device_trust_score
        else:
            features.append(np.random.uniform(0.1, 1.0, len(df)))
        
        if 'is_device_shared' in df.columns:
            features.append((~df['is_device_shared']).astype(int))  # device_exclusivity
        else:
            features.append(np.random.binomial(1, 0.7, len(df)))
        
        if 'velocity_score' in df.columns:
            features.append(np.clip(df['velocity_score'] / 10, 0, 1))  # velocity_risk
        else:
            features.append(np.random.uniform(0, 1, len(df)))
        
        if 'spending_deviation_score' in df.columns:
            features.append(np.clip(df['spending_deviation_score'], 0, 1))  # spending_anomaly
        else:
            features.append(np.random.uniform(0, 1, len(df)))
        
        # Add basic amount features
        amount_col = 'amount_ngn' if 'amount_ngn' in df.columns else 'amount'
        if amount_col in df.columns:
            features.append(np.log1p(df[amount_col]))
            features.append((df[amount_col] % 1000 == 0).astype(int))
        else:
            features.append(np.random.uniform(0, 10, len(df)))
            features.append(np.random.binomial(1, 0.3, len(df)))
        
        X = pd.DataFrame(np.column_stack(features), columns=[
            'device_trust_score', 'device_exclusivity', 'velocity_risk', 
            'spending_anomaly', 'amount_log', 'round_amount'
        ])
        
        y = df['is_fraud'] if 'is_fraud' in df.columns else df.get('isFraud', np.random.binomial(1, 0.1, len(df)))
        
        return X, y
    
    def _prepare_financial_features(self, df):
        """Prepare financial features for evaluation"""
        # Basic financial features that should exist in PaySim
        feature_cols = [
            'amount', 'oldbalanceOrg', 'newbalanceOrig', 
            'oldbalanceDest', 'newbalanceDest', 'step'
        ]
        
        # Keep only existing columns
        available_cols = [col for col in feature_cols if col in df.columns]
        
        if available_cols:
            X = df[available_cols].copy()
            
            # Add engineered features
            if 'amount' in X.columns:
                X['amount_log'] = np.log1p(X['amount'])
                X['is_large_amount'] = (X['amount'] > X['amount'].quantile(0.95)).astype(int)
            
            if 'oldbalanceOrg' in X.columns and 'amount' in X.columns:
                X['balance_ratio'] = X['amount'] / (X['oldbalanceOrg'] + 1)
            
            if 'type' in df.columns:
                # Encode transaction type
                type_map = {'PAYMENT': 0, 'TRANSFER': 1, 'CASH_OUT': 2, 'DEBIT': 3, 'CASH_IN': 4}
                X['type_encoded'] = df['type'].map(type_map).fillna(0)
        else:
            # Fallback to numeric columns
            X = df.select_dtypes(include=[np.number]).iloc[:, :8]
        
        y = df['is_fraud'] if 'is_fraud' in df.columns else df.get('isFraud', np.random.binomial(1, 0.1, len(df)))
        
        return X, y
    
    def _prepare_security_features(self, df):
        """Prepare security features for evaluation"""
        # Basic security-focused features
        features = []
        
        # Amount-based security features
        if 'amount' in df.columns:
            features.append(np.log1p(df['amount']))
            features.append((df['amount'] < 1).astype(int))  # micro transactions
            features.append((df['amount'] % 100 == 0).astype(int))  # round amounts
        else:
            features.extend([np.random.uniform(0, 10, len(df)), 
                           np.random.binomial(1, 0.1, len(df)),
                           np.random.binomial(1, 0.3, len(df))])
        
        # Time-based security features
        if 'step' in df.columns:
            features.append(df['step'] % 24)  # hour
            features.append(((df['step'] % 24) > 22).astype(int))  # night activity
        else:
            features.extend([np.random.uniform(0, 24, len(df)),
                           np.random.binomial(1, 0.2, len(df))])
        
        # Account-based security features
        if 'oldbalanceOrg' in df.columns:
            features.append((df['oldbalanceOrg'] == 0).astype(int))  # zero balance
        else:
            features.append(np.random.binomial(1, 0.15, len(df)))
        
        X = pd.DataFrame(np.column_stack(features), columns=[
            'amount_log', 'micro_transaction', 'round_amount', 
            'transaction_hour', 'night_activity', 'zero_balance'
        ])
        
        y = df['is_fraud'] if 'is_fraud' in df.columns else df.get('isFraud', np.random.binomial(1, 0.1, len(df)))
        
        return X, y
    
    def _calculate_metrics(self, y_true, y_pred, y_pred_proba, model_name):
        """Calculate comprehensive evaluation metrics"""
        results = {
            'model_name': model_name,
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'auc_score': roc_auc_score(y_true, y_pred_proba) if len(set(y_true)) > 1 else 0.5,
            'fraud_rate': y_true.mean(),
            'prediction_rate': y_pred.mean(),
            'samples_tested': len(y_true)
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            results.update({
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_positives': int(tp)
            })
        
        return results
    
    def _print_results(self, model_name, results):
        """Print evaluation results"""
        print(f"\n📈 {model_name.upper()} MODEL PERFORMANCE:")
        print(f"   Samples tested: {results['samples_tested']:,}")
        print(f"   Actual fraud rate: {results['fraud_rate']:.3%}")
        print(f"   Predicted fraud rate: {results['prediction_rate']:.3%}")
        print(f"\n📊 Performance Metrics:")
        print(f"   AUC Score: {results['auc_score']:.4f}")
        print(f"   Accuracy: {results['accuracy']:.4f}")
        print(f"   Precision: {results['precision']:.4f}")
        print(f"   Recall: {results['recall']:.4f}")
        print(f"   F1-Score: {results['f1_score']:.4f}")
        
        if all(k in results for k in ['true_positives', 'false_positives', 'false_negatives', 'true_negatives']):
            print(f"\n🎯 Confusion Matrix:")
            print(f"   True Positives: {results['true_positives']}")
            print(f"   False Positives: {results['false_positives']}")
            print(f"   True Negatives: {results['true_negatives']}")
            print(f"   False Negatives: {results['false_negatives']}")
    
    def run_full_evaluation(self):
        """Run comprehensive evaluation of all models"""
        print("🚀 COMPREHENSIVE MODEL EVALUATION SYSTEM")
        print("=" * 70)
        
        # Load models
        models_loaded = self.load_all_models()
        if not models_loaded:
            print("❌ No models found for evaluation!")
            return
        
        # Prepare test datasets
        test_datasets = self.prepare_test_data()
        
        # Evaluate each model
        for model_name, model_set in self.models.items():
            if model_name in test_datasets:
                self.evaluate_model(model_name, model_set, test_datasets[model_name])
            elif 'advanced_network' in model_name and test_datasets:
                # Use any available dataset for advanced model
                dataset_name = list(test_datasets.keys())[0]
                self.evaluate_model(model_name, model_set, test_datasets[dataset_name])
        
        # Generate summary report
        self.generate_summary_report()
        
        return self.test_results
    
    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        if not self.test_results:
            print("❌ No evaluation results available!")
            return
        
        print("\n🏆 EVALUATION SUMMARY REPORT")
        print("=" * 60)
        
        # Create comparison table
        comparison_data = []
        for model_name, results in self.test_results.items():
            comparison_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'AUC': f"{results['auc_score']:.4f}",
                'Accuracy': f"{results['accuracy']:.4f}",
                'Precision': f"{results['precision']:.4f}",
                'Recall': f"{results['recall']:.4f}",
                'F1-Score': f"{results['f1_score']:.4f}",
                'Samples': f"{results['samples_tested']:,}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print("\n📊 MODEL COMPARISON TABLE:")
        print(comparison_df.to_string(index=False))
        
        # Find best performing model
        if len(self.test_results) > 1:
            best_auc = max(self.test_results.items(), key=lambda x: x[1]['auc_score'])
            best_f1 = max(self.test_results.items(), key=lambda x: x[1]['f1_score'])
            
            print(f"\n🏆 BEST PERFORMING MODELS:")
            print(f"   Best AUC: {best_auc[0]} ({best_auc[1]['auc_score']:.4f})")
            print(f"   Best F1: {best_f1[0]} ({best_f1[1]['f1_score']:.4f})")
        
        # Save results
        results_path = self.artifacts_path / "evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        
        print(f"\n💾 Results saved to: {results_path}")

def run_evaluation():
    """Main evaluation function"""
    evaluator = ModelEvaluator()
    results = evaluator.run_full_evaluation()
    return evaluator, results

if __name__ == "__main__":
    evaluator, results = run_evaluation()