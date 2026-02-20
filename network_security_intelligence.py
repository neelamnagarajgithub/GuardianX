"""
NSI_PROD_V2 - Production-ready Network Security Intelligence (single-file)

Features:
- Dataset adapter (maps fraud datasets -> synthetic network behavior for pipeline testing)
- Production-grade feature extractor
- LightGBM model with class-weight and isotonic calibration
- Artifact management (models/features saved under artifacts/)
- Simple CLI: train / demo / infer

Dependencies:
- pandas, numpy, scikit-learn, lightgbm, joblib, loguru
- (optional) datasets or local parquet files in RAW dir produced by your download_and_save script

Run examples:
$ python nsi_prod_v2.py train
$ python nsi_prod_v2.py demo
$ python nsi_prod_v2.py infer --input sample_features.parquet
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass
import argparse
import numpy as np
import pandas as pd
from loguru import logger
import joblib
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.calibration import CalibratedClassifierCV

# -----------------------------
# PRODUCTION CONFIG
# -----------------------------
ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", "/kaggle/working/artifacts"))
RAW = ARTIFACT_DIR / "raw"
MODEL_DIR = ARTIFACT_DIR / "models"
FEATURE_DIR = ARTIFACT_DIR / "features"

RANDOM_STATE = int(os.getenv("RANDOM_STATE", 42))

RAW.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
FEATURE_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Network Security Intelligence (production-grade)
# -----------------------------

class NetworkSecurityIntelligence:
    """Core feature extraction + heuristics for network security intelligence.

    Note: In production, you should replace heuristics with specialized services
    (GeoIP, VPN detection, threat feeds). This class focuses on deterministic,
    vectorizable features that work in an ML pipeline.
    """

    def __init__(self):
        self.suspicious_ips = set()
        self.vpn_providers = set()
        self.bot_patterns = {}
        self._load_threat_intelligence()

    def _load_threat_intelligence(self):
        # Minimal example; replace with periodic feeds or DBs in prod
        self.suspicious_ips = {"1.2.3.4", "5.6.7.8"}
        self.vpn_providers = {"nordvpn", "expressvpn", "surfshark", "cyberghost"}
        self.bot_patterns = {
            'scrapy': r'scrapy/[\d.]+',
            'selenium': r'selenium|webdriver',
            'headless': r'headless|phantom',
            'automation': r'automation|bot|crawler|spider'
        }

    # -----------------------------
    # Public API used by dataset builder / training pipeline
    # -----------------------------

    def extract_ip_features_from_group(self, ip: str, group: Dict) -> Dict:
        """Mapping from aggregated group dict -> feature dict.
        The incoming `group` is expected to be pre-aggregated counters/lists.
        """
        return self._extract_ip_features(ip, group)

    def _extract_ip_features(self, ip: str, data: Dict) -> Dict:
        """Production-grade IP feature extraction"""

        total_requests = int(data.get('nginx_requests', 0)) + int(data.get('api_calls', 0))
        status_list = data.get('status_codes', []) or []

        error_count = sum(1 for s in status_list if int(s) >= 400)
        auth_fail_count = sum(1 for s in status_list if int(s) in (401, 403))

        request_times = data.get('request_times', []) or []
        avg_req_time = float(np.mean(request_times)) if len(request_times) else 0.0
        std_req_time = float(np.std(request_times)) if len(request_times) else 0.0

        features = {
            "ip_address": ip,

            # volume
            "total_requests": total_requests,
            "nginx_request_count": int(data.get('nginx_requests', 0)),
            "api_call_count": int(data.get('api_calls', 0)),
            "redis_operation_count": int(data.get('redis_ops', 0)),

            # diversity
            "unique_endpoints": int(len(data.get('endpoints_accessed', set()))),
            "unique_user_agents": int(len(data.get('user_agents', set()))),
            "unique_sessions": int(len(data.get('unique_sessions', set()))),

            # error behaviour
            "error_rate": float(error_count / max(len(status_list), 1)),
            "auth_failure_rate": float(auth_fail_count / max(len(status_list), 1)),

            # latency behaviour
            "avg_request_time": avg_req_time,
            "request_time_std": std_req_time,

            # intelligence
            "is_private_ip": int(self._is_private_ip(ip)),
            "is_suspicious_ip": int(ip in self.suspicious_ips),
            "geo_risk_score": float(self._get_geo_risk_score(ip)),

            # automation signals
            "has_bot_user_agent": int(self._has_bot_user_agent(data.get('user_agents', set()))),
            "user_agent_entropy": float(self._calculate_user_agent_entropy(data.get('user_agents', set()))),

            # infra masking
            "vpn_probability": float(self._detect_vpn_usage(ip, data)),
            "proxy_probability": float(self._detect_proxy_usage(ip, data)),

            # velocity
            "requests_per_minute": float(self._calculate_request_rate(request_times)),
            "burst_behavior_score": float(self._calculate_burst_score(request_times)),

            # session abuse
            "session_switching_rate": float(self._calculate_session_switching_rate(data.get('unique_sessions', set()))),
            "session_hijacking_score": float(self._calculate_session_hijacking_score(data)),
        }

        return features

    # -----------------------------
    # Heuristic helpers
    # -----------------------------

    def _is_private_ip(self, ip: str) -> bool:
        try:
            import ipaddress
            return ipaddress.ip_address(ip).is_private
        except Exception:
            return False

    def _get_geo_risk_score(self, ip: str) -> float:
        # Placeholder: in prod use GeoIP DB
        # Countries with higher risk would return higher values
        return 0.5

    def _has_bot_user_agent(self, user_agents: set) -> bool:
        for ua in user_agents:
            ua_l = ua.lower()
            for p in self.bot_patterns.values():
                try:
                    if __import__('re').search(p, ua_l):
                        return True
                except Exception:
                    continue
        return False

    def _calculate_user_agent_entropy(self, user_agents: set) -> float:
        if not user_agents:
            return 0.0
        counts = pd.Series(list(user_agents)).value_counts().values
        probs = counts / counts.sum()
        entropy = -np.sum(probs * np.log2(probs + 1e-12))
        return float(entropy)

    def _detect_vpn_usage(self, ip: str, data: Dict) -> float:
        score = 0.0
        for ua in data.get('user_agents', set()):
            for vpn in self.vpn_providers:
                if vpn in ua.lower():
                    score += 0.3
        return min(score, 1.0)

    def _detect_proxy_usage(self, ip: str, data: Dict) -> float:
        score = 0.0
        if len(data.get('user_agents', set())) > 5:
            score += 0.4
        if len(data.get('endpoints_accessed', set())) > 20:
            score += 0.3
        return min(score, 1.0)

    def _calculate_request_rate(self, request_times: List) -> float:
        if len(request_times) < 2:
            return 0.0
        time_span = (max(request_times) - min(request_times)).total_seconds() if hasattr(request_times[0], 'total_seconds') else max(request_times) - min(request_times)
        if time_span == 0:
            return float(len(request_times))
        return float(len(request_times) / (time_span / 60.0))

    def _calculate_burst_score(self, request_times: List) -> float:
        if len(request_times) < 10:
            return 0.0
        sorted_times = sorted(request_times)
        intervals = [(sorted_times[i+1] - sorted_times[i]).total_seconds() for i in range(len(sorted_times)-1)]
        short_intervals = [i for i in intervals if i < 1.0]
        return float(len(short_intervals) / max(len(intervals), 1))

    def _calculate_session_switching_rate(self, sessions: set) -> float:
        return float(len(sessions) / 10.0)

    def _calculate_session_hijacking_score(self, data: Dict) -> float:
        score = 0.0
        if len(data.get('unique_sessions', set())) > 3 and len(data.get('user_agents', set())) > 2:
            score += 0.5
        if int(data.get('api_calls', 0)) > 100 and len(data.get('unique_sessions', set())) > 5:
            score += 0.3
        return min(score, 1.0)

# -----------------------------
# Dataset Builder - adapts fraud datasets into training frame
# -----------------------------

class NSIDatasetBuilder:
    """Converts available raw datasets into a training DataFrame suitable for NSI pipelines.

    This class is intentionally conservative: it looks for parquet files under RAW dir.
    For testing and pipeline validation we create deterministic synthetic network fields.
    """

    def __init__(self, raw_dir: Path):
        self.raw_dir = Path(raw_dir)

    def load_training_frame(self) -> pd.DataFrame:
        frames = []

        # PaySim
        paysim_path = self.raw_dir / "paysim.parquet"
        if paysim_path.exists():
            try:
                df = pd.read_parquet(paysim_path)
                frames.append(self._convert_paysim(df))
            except Exception as e:
                logger.warning(f"Failed to load PaySim: {e}")

        # Nigerian
        nig_path = self.raw_dir / "nigerian.parquet"
        if nig_path.exists():
            try:
                df = pd.read_parquet(nig_path)
                frames.append(self._convert_nigerian(df))
            except Exception as e:
                logger.warning(f"Failed to load Nigerian dataset: {e}")

        # Try cifer sample
        cifer_path = self.raw_dir / "cifer_sample.parquet"
        if cifer_path.exists():
            try:
                df = pd.read_parquet(cifer_path)
                frames.append(self._convert_generic_cifer(df))
            except Exception as e:
                logger.warning(f"Failed to load CIFER sample: {e}")

        if not frames:
            raise FileNotFoundError("No supported datasets found in RAW dir. Place paysim.parquet or nigerian.parquet under RAW.")

        combined = pd.concat(frames, ignore_index=True)
        logger.info(f"Loaded combined training frame with {len(combined)} rows")
        return combined

    # -------------------------
    # converters
    # -------------------------

    def _convert_paysim(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # pay attention to typical PaySim columns: 'step','type','amount','oldbalanceOrg','newbalanceOrig','isFraud'
        if 'isFraud' not in df.columns:
            df['isFraud'] = 0
        # deterministic IP mapping for reproducibility
        df['ip_address'] = (df['step'].fillna(0).astype(int) % 255).astype(str) + "." + (df['amount'].fillna(0).astype(int) % 255).astype(str) + ".1.1"
        df['label'] = df['isFraud'].astype(int)
        return df[['ip_address', 'amount', 'oldbalanceOrg']].assign(label=df['label'])

    def _convert_nigerian(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # common columns might vary; try to find a fraud label
        if 'is_fraud' in df.columns:
            df['label'] = df['is_fraud'].astype(int)
        elif 'fraud_flag' in df.columns:
            df['label'] = df['fraud_flag'].astype(int)
        else:
            df['label'] = 0
        df['ip_address'] = (df.index % 255).astype(str) + ".10.10.10"
        return df[['ip_address']].assign(label=df['label'])

    def _convert_generic_cifer(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # generic mapping
        df['label'] = 0
        df['ip_address'] = (df.index % 255).astype(str) + ".20.20.20"
        return df[['ip_address']].assign(label=df['label'])

# -----------------------------
# Model class
# -----------------------------

class NetworkThreatDetectionModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.calibrator = None

    def train(self, features: pd.DataFrame, labels: pd.Series):
        X = features.drop(columns=['ip_address'])
        y = labels.astype(int)

        self.feature_columns = X.columns.tolist()

        # handle extreme imbalance via class weights
        classes = np.unique(y)
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
        class_weight = dict(zip(classes, weights))

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

        X_train_s = self.scaler.fit_transform(X_train)
        X_val_s = self.scaler.transform(X_val)

        base_model = lgb.LGBMClassifier(
            objective='binary',
            n_estimators=500,
            learning_rate=0.05,
            max_depth=-1,
            num_leaves=64,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight=class_weight,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

        base_model.fit(X_train_s, y_train, eval_set=[(X_val_s, y_val)], eval_metric='auc', verbose=False)

        # calibrate probability estimates
        self.calibrator = CalibratedClassifierCV(base_model, method='isotonic', cv=3)
        self.calibrator.fit(X_train_s, y_train)

        self.model = base_model
        return self

    def predict_threat(self, network_features: pd.DataFrame) -> List[Dict]:
        if self.model is None or self.calibrator is None or self.feature_columns is None:
            raise ValueError("Model not trained or incomplete")

        X = network_features[self.feature_columns]
        X_scaled = self.scaler.transform(X)

        probs = self.calibrator.predict_proba(X_scaled)[:, 1]
        results = []
        for idx, prob in enumerate(probs):
            ip = network_features.iloc[idx]['ip_address']
            results.append({
                'ip_address': ip,
                'threat_probability': float(prob),
                'is_threat': bool(prob > 0.5),
                'risk_level': self._assess_risk_level(float(prob)),
                'recommended_action': self._get_recommended_action(float(prob))
            })
        return results

    def _assess_risk_level(self, prob: float) -> str:
        if prob >= 0.8:
            return 'CRITICAL'
        elif prob >= 0.6:
            return 'HIGH'
        elif prob >= 0.4:
            return 'MEDIUM'
        elif prob >= 0.2:
            return 'LOW'
        else:
            return 'MINIMAL'

    def _get_recommended_action(self, prob: float) -> str:
        if prob >= 0.8:
            return 'BLOCK_IMMEDIATELY'
        elif prob >= 0.6:
            return 'RATE_LIMIT'
        elif prob >= 0.4:
            return 'ENHANCED_MONITORING'
        elif prob >= 0.2:
            return 'LOG_AND_MONITOR'
        else:
            return 'ALLOW'

    def save_model(self, model_path: Path):
        payload = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'calibrator': self.calibrator
        }
        joblib.dump(payload, str(model_path))
        logger.info(f"Saved model to {model_path}")

    def load_model(self, model_path: Path):
        payload = joblib.load(str(model_path))
        self.model = payload['model']
        self.scaler = payload['scaler']
        self.feature_columns = payload['feature_columns']
        self.calibrator = payload.get('calibrator', None)
        logger.info(f"Loaded model from {model_path}")

# -----------------------------
# Utility: build synthetic network features for training
# -----------------------------

def build_synthetic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Given a dataset containing ip_address and label, generate deterministic synthetic features.

    This function produces reproducible features suitable for pipeline testing. When real network
    logs become available, replace this step with true aggregation of logs.
    """

    rng = np.random.RandomState(RANDOM_STATE)

    base = df.copy()
    n = len(base)

    # deterministic but varied features
    base['total_requests'] = (rng.poisson(20, n) + (base.index % 5).astype(int)).astype(int)
    base['nginx_request_count'] = (base['total_requests'] * 0.6).astype(int)
    base['api_call_count'] = (base['total_requests'] * 0.4).astype(int)
    base['redis_ops'] = (rng.poisson(2, n)).astype(int)

    base['unique_endpoints'] = (np.clip(rng.poisson(4, n), 1, 100)).astype(int)
    base['unique_user_agents'] = (np.clip(rng.poisson(2, n), 1, 20)).astype(int)
    base['unique_sessions'] = (np.clip(rng.poisson(3, n), 1, 50)).astype(int)

    base['error_rate'] = rng.beta(2, 10, n)
    base['auth_failure_rate'] = rng.beta(1, 20, n)

    base['avg_request_time'] = rng.exponential(scale=0.1, size=n)
    base['request_time_std'] = rng.exponential(scale=0.02, size=n)

    base['is_private_ip'] = base['ip_address'].apply(lambda x: 1 if str(x).startswith('10.') or str(x).startswith('192.168') else 0)
    base['is_suspicious_ip'] = base['ip_address'].apply(lambda x: 1 if str(x).startswith('1.2') else 0)

    base['has_bot_user_agent'] = (rng.binomial(1, 0.05, n)).astype(int)
    base['user_agent_entropy'] = rng.uniform(0, 3, n)

    base['vpn_probability'] = rng.uniform(0, 0.2, n)
    base['proxy_probability'] = rng.uniform(0, 0.2, n)

    base['requests_per_minute'] = base['total_requests'] / rng.uniform(10, 60, n)
    base['burst_behavior_score'] = rng.uniform(0, 1, n)

    base['session_switching_rate'] = base['unique_sessions'] / 10.0
    base['session_hijacking_score'] = rng.uniform(0, 1, n) * (base['session_switching_rate'] > 0.3)

    # keep canonical ordering of features
    feature_cols = [
        'ip_address', 'total_requests', 'nginx_request_count', 'api_call_count', 'redis_ops',
        'unique_endpoints', 'unique_user_agents', 'unique_sessions', 'error_rate', 'auth_failure_rate',
        'avg_request_time', 'request_time_std', 'is_private_ip', 'is_suspicious_ip', 'has_bot_user_agent',
        'user_agent_entropy', 'vpn_probability', 'proxy_probability', 'requests_per_minute', 'burst_behavior_score',
        'session_switching_rate', 'session_hijacking_score'
    ]

    # ensure all columns exist
    for c in feature_cols:
        if c not in base.columns:
            base[c] = 0

    return base[['ip_address'] + [c for c in feature_cols if c != 'ip_address']]

# -----------------------------
# Training runner
# -----------------------------

def train_nsi_from_datasets(raw_dir: Path = RAW, model_dir: Path = MODEL_DIR):
    logger.info("Starting NSI training from datasets...")

    builder = NSIDatasetBuilder(raw_dir)
    df = builder.load_training_frame()

    # ensure label column exists
    if 'label' not in df.columns:
        df['label'] = 0

    # build synthetic network features (replace with real logs later)
    feature_df = build_synthetic_features(df[['ip_address']])

    # merge label
    feature_df = feature_df.merge(df[['ip_address', 'label']].drop_duplicates('ip_address'), on='ip_address', how='left')
    feature_df['label'] = feature_df['label'].fillna(0).astype(int)

    # save features snapshot
    feature_snapshot = FEATURE_DIR / 'training_features.parquet'
    feature_df.to_parquet(feature_snapshot, index=False)
    logger.info(f"Saved training features snapshot to {feature_snapshot}")

    # prepare model input
    features = feature_df.drop(columns=['label'])
    labels = feature_df['label']

    # train model
    model = NetworkThreatDetectionModel()
    model.train(features, labels)

    # save model
    model_path = model_dir / 'network_threat_model.joblib'
    model.save_model(model_path)

    logger.info("Training complete")
    return model_path

# -----------------------------
# Demo / inference utilities
# -----------------------------

def demo_run(model_path: Path = MODEL_DIR / 'network_threat_model.joblib'):
    logger.info("Running demo of NSI inference...")

    # load model
    model = NetworkThreatDetectionModel()
    model.load_model(model_path)

    # load sample features
    sample_path = FEATURE_DIR / 'training_features.parquet'
    if not sample_path.exists():
        logger.error(f"No feature snapshot found at {sample_path}. Run training first.")
        return

    feat = pd.read_parquet(sample_path).head(20)

    results = model.predict_threat(feat)
    for r in results[:10]:
        logger.info(json.dumps(r))

    logger.info("Demo run complete")

# -----------------------------
# CLI
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description='NSI Production runner')
    sub = parser.add_subparsers(dest='cmd')

    sub.add_parser('train', help='Train NSI from RAW datasets')
    sub.add_parser('demo', help='Run demo inference on saved feature snapshot')

    infer = sub.add_parser('infer', help='Run inference on provided features parquet')
    infer.add_argument('--input', help='Path to features parquet', required=False)

    args = parser.parse_args()

    if args.cmd == 'train':
        train_nsi_from_datasets()
    elif args.cmd == 'demo':
        demo_run()
    elif args.cmd == 'infer':
        if not args.input:
            logger.error('Please provide --input path to features parquet')
            return
        model = NetworkThreatDetectionModel()
        model_path = MODEL_DIR / 'network_threat_model.joblib'
        if not model_path.exists():
            logger.error('Trained model not found. Run `train` first.')
            return
        model.load_model(model_path)
        feat = pd.read_parquet(args.input)
        results = model.predict_threat(feat)
        print(json.dumps(results, indent=2))
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
