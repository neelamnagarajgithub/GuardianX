# ============================================================
# BEHAVIORAL FRAUD DETECTION SYSTEM
# Analyze user behavior patterns for fraud detection
# ============================================================

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from typing import Dict, List, Optional, Any
from collections import defaultdict, Counter
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
from scipy import stats

class BehavioralFraudDetector:
    """Behavioral fraud detection through user activity analysis"""
    
    def __init__(self):
        self.user_profiles = {}
        self.session_analyzer = SessionAnalyzer()
        self.navigation_analyzer = NavigationAnalyzer()
        self.interaction_analyzer = InteractionAnalyzer()
        self.model = None
        self.scaler = StandardScaler()
        
    def analyze_user_session(self, session_data: Dict) -> Dict[str, Any]:
        """Analyze a single user session for behavioral anomalies"""
        
        user_id = session_data.get('user_id')
        session_id = session_data.get('session_id')
        
        # Extract behavioral features
        session_features = self.session_analyzer.extract_session_features(session_data)
        navigation_features = self.navigation_analyzer.extract_navigation_features(session_data)
        interaction_features = self.interaction_analyzer.extract_interaction_features(session_data)
        
        # Combine all features
        behavioral_features = {
            **session_features,
            **navigation_features, 
            **interaction_features
        }
        
        # Compare against user's historical behavior
        anomaly_score = self._calculate_behavior_anomaly_score(user_id, behavioral_features)
        
        result = {
            'user_id': user_id,
            'session_id': session_id,
            'behavioral_features': behavioral_features,
            'anomaly_score': anomaly_score,
            'risk_indicators': self._identify_risk_indicators(behavioral_features),
            'fraud_probability': self._calculate_fraud_probability(behavioral_features, anomaly_score),
            'recommended_action': self._get_recommended_action(anomaly_score)
        }
        
        # Update user profile
        self._update_user_profile(user_id, behavioral_features)
        
        return result
    
    def _calculate_behavior_anomaly_score(self, user_id: str, features: Dict) -> float:
        """Calculate how anomalous current behavior is compared to user's history"""
        
        if user_id not in self.user_profiles:
            return 0.5  # New user, moderate suspicion
        
        profile = self.user_profiles[user_id]
        anomaly_score = 0.0
        feature_count = 0
        
        for feature_name, current_value in features.items():
            if feature_name in profile['feature_history']:
                historical_values = profile['feature_history'][feature_name]
                
                if len(historical_values) >= 5:  # Need minimum history
                    mean_val = np.mean(historical_values)
                    std_val = np.std(historical_values)
                    
                    if std_val > 0:
                        z_score = abs(current_value - mean_val) / std_val
                        # Convert z-score to probability (higher z-score = more anomalous)
                        feature_anomaly = min(z_score / 3.0, 1.0)  # Cap at 1.0
                        anomaly_score += feature_anomaly
                        feature_count += 1
        
        return anomaly_score / max(feature_count, 1)
    
    def _identify_risk_indicators(self, features: Dict) -> List[Dict]:
        """Identify specific behavioral risk indicators"""
        
        risk_indicators = []
        
        # Session duration anomalies
        if features.get('session_duration_minutes', 0) < 0.5:
            risk_indicators.append({
                'type': 'SHORT_SESSION',
                'severity': 'MEDIUM',
                'description': 'Extremely short session duration'
            })
        
        if features.get('session_duration_minutes', 0) > 480:  # 8 hours
            risk_indicators.append({
                'type': 'LONG_SESSION', 
                'severity': 'HIGH',
                'description': 'Unusually long session duration'
            })
        
        # Click pattern anomalies
        if features.get('clicks_per_minute', 0) > 10:
            risk_indicators.append({
                'type': 'HIGH_CLICK_RATE',
                'severity': 'HIGH', 
                'description': 'Automated clicking behavior detected'
            })
        
        # Navigation anomalies
        if features.get('pages_per_minute', 0) > 5:
            risk_indicators.append({
                'type': 'RAPID_NAVIGATION',
                'severity': 'HIGH',
                'description': 'Suspiciously fast page navigation'
            })
        
        # Time-based anomalies
        current_hour = datetime.now().hour
        if current_hour >= 2 and current_hour <= 5:
            risk_indicators.append({
                'type': 'OFF_HOURS_ACTIVITY',
                'severity': 'MEDIUM',
                'description': 'Activity during unusual hours'
            })
        
        return risk_indicators
    
    def _calculate_fraud_probability(self, features: Dict, anomaly_score: float) -> float:
        """Calculate fraud probability based on behavioral analysis"""
        
        base_score = anomaly_score * 0.6
        
        # Add specific risk multipliers
        risk_multiplier = 1.0
        
        # Bot-like behavior
        if features.get('mouse_movement_entropy', 0) < 0.3:
            risk_multiplier += 0.3
        
        # Unusual timing patterns
        if features.get('keystroke_timing_variance', 0) < 0.1:
            risk_multiplier += 0.2
        
        # Rapid operations
        if features.get('actions_per_minute', 0) > 20:
            risk_multiplier += 0.4
        
        fraud_prob = min(base_score * risk_multiplier, 1.0)
        
        return fraud_prob
    
    def _get_recommended_action(self, anomaly_score: float) -> str:
        """Get recommended action based on anomaly score"""
        
        if anomaly_score >= 0.8:
            return "BLOCK_USER"
        elif anomaly_score >= 0.6:
            return "REQUIRE_ADDITIONAL_AUTH"
        elif anomaly_score >= 0.4:
            return "ENHANCED_MONITORING"
        elif anomaly_score >= 0.2:
            return "NORMAL_MONITORING"
        else:
            return "ALLOW"
    
    def _update_user_profile(self, user_id: str, features: Dict):
        """Update user's behavioral profile"""
        
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {
                'first_seen': datetime.now(),
                'session_count': 0,
                'feature_history': defaultdict(list)
            }
        
        profile = self.user_profiles[user_id]
        profile['session_count'] += 1
        profile['last_seen'] = datetime.now()
        
        # Update feature history (keep last 100 values)
        for feature_name, value in features.items():
            profile['feature_history'][feature_name].append(value)
            if len(profile['feature_history'][feature_name]) > 100:
                profile['feature_history'][feature_name].pop(0)

class SessionAnalyzer:
    """Analyze user session patterns"""
    
    def extract_session_features(self, session_data: Dict) -> Dict:
        """Extract session-level behavioral features"""
        
        features = {}
        
        # Session timing
        start_time = session_data.get('start_time')
        end_time = session_data.get('end_time', datetime.now())
        
        if start_time:
            duration = (end_time - start_time).total_seconds() / 60  # minutes
            features['session_duration_minutes'] = duration
        else:
            features['session_duration_minutes'] = 0
        
        # Activity metrics
        events = session_data.get('events', [])
        features['total_events'] = len(events)
        
        if features['session_duration_minutes'] > 0:
            features['events_per_minute'] = features['total_events'] / features['session_duration_minutes']
        else:
            features['events_per_minute'] = features['total_events']
        
        # Event type distribution
        event_types = [event.get('type') for event in events if event.get('type')]
        event_counter = Counter(event_types)
        
        features['click_count'] = event_counter.get('click', 0)
        features['keypress_count'] = event_counter.get('keypress', 0)
        features['scroll_count'] = event_counter.get('scroll', 0)
        features['focus_count'] = event_counter.get('focus', 0)
        
        if features['session_duration_minutes'] > 0:
            features['clicks_per_minute'] = features['click_count'] / features['session_duration_minutes']
            features['keypresses_per_minute'] = features['keypress_count'] / features['session_duration_minutes']
        else:
            features['clicks_per_minute'] = features['click_count']
            features['keypresses_per_minute'] = features['keypress_count']
        
        # Pause patterns
        event_timestamps = [event.get('timestamp') for event in events if event.get('timestamp')]
        if len(event_timestamps) > 1:
            intervals = [(event_timestamps[i+1] - event_timestamps[i]).total_seconds() 
                        for i in range(len(event_timestamps)-1)]
            features['avg_pause_seconds'] = np.mean(intervals)
            features['pause_variance'] = np.var(intervals)
        else:
            features['avg_pause_seconds'] = 0
            features['pause_variance'] = 0
        
        return features

class NavigationAnalyzer:
    """Analyze user navigation patterns"""
    
    def extract_navigation_features(self, session_data: Dict) -> Dict:
        """Extract navigation behavioral features"""
        
        features = {}
        
        # Page visits
        page_views = session_data.get('page_views', [])
        features['unique_pages_visited'] = len(set(page['url'] for page in page_views if page.get('url')))
        features['total_page_views'] = len(page_views)
        
        # Calculate pages per minute
        session_duration = session_data.get('session_duration_minutes', 1)
        features['pages_per_minute'] = features['total_page_views'] / max(session_duration, 1)
        
        # Page timing analysis
        page_durations = []
        for page in page_views:
            if page.get('time_on_page'):
                page_durations.append(page['time_on_page'])
        
        if page_durations:
            features['avg_time_per_page_seconds'] = np.mean(page_durations)
            features['time_per_page_variance'] = np.var(page_durations)
            features['min_time_per_page'] = min(page_durations)
            features['max_time_per_page'] = max(page_durations)
        else:
            features['avg_time_per_page_seconds'] = 0
            features['time_per_page_variance'] = 0
            features['min_time_per_page'] = 0
            features['max_time_per_page'] = 0
        
        # Navigation flow analysis
        page_sequence = [page.get('url') for page in page_views if page.get('url')]
        
        # Back/forward button usage
        features['navigation_back_count'] = sum(1 for page in page_views 
                                               if page.get('navigation_type') == 'back')
        features['navigation_forward_count'] = sum(1 for page in page_views 
                                                  if page.get('navigation_type') == 'forward')
        
        # Depth of navigation
        features['max_navigation_depth'] = max([page.get('depth', 0) for page in page_views] or [0])
        
        # URL pattern analysis
        features['external_link_clicks'] = sum(1 for page in page_views 
                                              if page.get('is_external_link', False))
        
        return features

class InteractionAnalyzer:
    """Analyze user interaction patterns"""
    
    def extract_interaction_features(self, session_data: Dict) -> Dict:
        """Extract interaction behavioral features"""
        
        features = {}
        
        # Mouse movement analysis
        mouse_events = [event for event in session_data.get('events', []) 
                       if event.get('type') == 'mousemove']
        
        if mouse_events:
            # Calculate mouse movement entropy
            x_coords = [event.get('x', 0) for event in mouse_events]
            y_coords = [event.get('y', 0) for event in mouse_events]
            
            # Movement variance
            features['mouse_movement_variance_x'] = np.var(x_coords) if x_coords else 0
            features['mouse_movement_variance_y'] = np.var(y_coords) if y_coords else 0
            
            # Calculate movement entropy (measure of randomness)
            features['mouse_movement_entropy'] = self._calculate_movement_entropy(x_coords, y_coords)
            
            # Movement speed analysis
            speeds = []
            for i in range(1, len(mouse_events)):
                prev_event = mouse_events[i-1]
                curr_event = mouse_events[i]
                
                time_diff = (curr_event.get('timestamp', datetime.now()) - 
                           prev_event.get('timestamp', datetime.now())).total_seconds()
                
                if time_diff > 0:
                    distance = ((curr_event.get('x', 0) - prev_event.get('x', 0))**2 + 
                              (curr_event.get('y', 0) - prev_event.get('y', 0))**2)**0.5
                    speed = distance / time_diff
                    speeds.append(speed)
            
            features['avg_mouse_speed'] = np.mean(speeds) if speeds else 0
            features['mouse_speed_variance'] = np.var(speeds) if speeds else 0
        else:
            features['mouse_movement_variance_x'] = 0
            features['mouse_movement_variance_y'] = 0
            features['mouse_movement_entropy'] = 0
            features['avg_mouse_speed'] = 0
            features['mouse_speed_variance'] = 0
        
        # Keystroke timing analysis
        keypress_events = [event for event in session_data.get('events', []) 
                          if event.get('type') == 'keypress']
        
        if len(keypress_events) > 1:
            keystroke_intervals = []
            for i in range(1, len(keypress_events)):
                time_diff = (keypress_events[i].get('timestamp', datetime.now()) - 
                           keypress_events[i-1].get('timestamp', datetime.now())).total_seconds()
                keystroke_intervals.append(time_diff)
            
            features['avg_keystroke_interval'] = np.mean(keystroke_intervals)
            features['keystroke_timing_variance'] = np.var(keystroke_intervals)
            features['min_keystroke_interval'] = min(keystroke_intervals)
            features['max_keystroke_interval'] = max(keystroke_intervals)
        else:
            features['avg_keystroke_interval'] = 0
            features['keystroke_timing_variance'] = 0
            features['min_keystroke_interval'] = 0
            features['max_keystroke_interval'] = 0
        
        # Scroll behavior analysis
        scroll_events = [event for event in session_data.get('events', []) 
                        if event.get('type') == 'scroll']
        
        features['scroll_event_count'] = len(scroll_events)
        
        if scroll_events:
            scroll_distances = [abs(event.get('delta_y', 0)) for event in scroll_events]
            features['avg_scroll_distance'] = np.mean(scroll_distances)
            features['scroll_distance_variance'] = np.var(scroll_distances)
        else:
            features['avg_scroll_distance'] = 0
            features['scroll_distance_variance'] = 0
        
        # Form interaction analysis
        form_events = [event for event in session_data.get('events', []) 
                      if event.get('type') in ['focus', 'blur', 'input']]
        
        features['form_interaction_count'] = len(form_events)
        
        # Calculate actions per minute
        session_duration = session_data.get('session_duration_minutes', 1)
        total_actions = len(session_data.get('events', []))
        features['actions_per_minute'] = total_actions / max(session_duration, 1)
        
        return features
    
    def _calculate_movement_entropy(self, x_coords: List, y_coords: List) -> float:
        """Calculate entropy of mouse movement patterns"""
        
        if len(x_coords) < 10:  # Need minimum data
            return 0.0
        
        # Discretize coordinates into grid
        grid_size = 50
        x_bins = np.linspace(min(x_coords), max(x_coords), grid_size)
        y_bins = np.linspace(min(y_coords), max(y_coords), grid_size)
        
        # Count movements in each grid cell
        movements = Counter()
        for x, y in zip(x_coords, y_coords):
            x_bin = np.digitize(x, x_bins)
            y_bin = np.digitize(y, y_bins)
            movements[(x_bin, y_bin)] += 1
        
        # Calculate entropy
        total_movements = sum(movements.values())
        probabilities = [count/total_movements for count in movements.values()]
        entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
        
        # Normalize by maximum possible entropy
        max_entropy = np.log2(min(grid_size * grid_size, len(x_coords)))
        
        return entropy / max_entropy if max_entropy > 0 else 0.0

# ============================================================
# BEHAVIORAL FRAUD MODEL
# ============================================================

class BehavioralFraudModel:
    """Machine learning model for behavioral fraud detection"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
    
    def train(self, behavioral_features: pd.DataFrame, fraud_labels: pd.Series):
        """Train behavioral fraud detection model"""
        
        # Prepare features
        self.feature_columns = [col for col in behavioral_features.columns 
                               if col not in ['user_id', 'session_id']]
        
        X = behavioral_features[self.feature_columns].fillna(0)
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model = lgb.LGBMClassifier(
            objective='binary',
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            random_state=42
        )
        
        self.model.fit(X_scaled, fraud_labels)
        
        return self
    
    def predict_fraud(self, behavioral_features: pd.DataFrame) -> List[Dict]:
        """Predict behavioral fraud probability"""
        
        if self.model is None:
            raise ValueError("Model not trained")
        
        X = behavioral_features[self.feature_columns].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        fraud_probabilities = self.model.predict_proba(X_scaled)[:, 1]
        
        results = []
        for i, row in behavioral_features.iterrows():
            prob = fraud_probabilities[i]
            results.append({
                'user_id': row.get('user_id'),
                'session_id': row.get('session_id'),
                'behavioral_fraud_probability': prob,
                'risk_level': self._assess_risk_level(prob),
                'recommended_action': self._get_recommended_action(prob)
            })
        
        return results
    
    def _assess_risk_level(self, prob: float) -> str:
        """Assess behavioral fraud risk level"""
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
        """Get recommended action for behavioral fraud"""
        if prob >= 0.8:
            return "BLOCK_SESSION"
        elif prob >= 0.6:
            return "REQUIRE_CAPTCHA"
        elif prob >= 0.4:
            return "ADDITIONAL_VERIFICATION"
        elif prob >= 0.2:
            return "MONITOR_CLOSELY"
        else:
            return "NORMAL_PROCESSING"

# ============================================================
# USAGE EXAMPLE
# ============================================================

def demo_behavioral_fraud_detection():
    """Demonstrate behavioral fraud detection"""
    
    print("🧠 Behavioral Fraud Detection System")
    print("="*50)
    
    # Initialize detector
    detector = BehavioralFraudDetector()
    
    # Example session data
    session_data = {
        'user_id': 'user_12345',
        'session_id': 'sess_98765',
        'start_time': datetime.now() - timedelta(minutes=45),
        'end_time': datetime.now(),
        'events': [
            {'type': 'click', 'timestamp': datetime.now() - timedelta(minutes=44), 'x': 100, 'y': 200},
            {'type': 'mousemove', 'timestamp': datetime.now() - timedelta(minutes=43), 'x': 150, 'y': 250},
            {'type': 'keypress', 'timestamp': datetime.now() - timedelta(minutes=42), 'key': 'a'},
            {'type': 'scroll', 'timestamp': datetime.now() - timedelta(minutes=41), 'delta_y': 100}
        ],
        'page_views': [
            {'url': '/dashboard', 'time_on_page': 120, 'depth': 1},
            {'url': '/profile', 'time_on_page': 300, 'depth': 2}
        ]
    }
    
    # Analyze session
    result = detector.analyze_user_session(session_data)
    
    print(f"👤 User: {result['user_id']}")
    print(f"🎯 Anomaly Score: {result['anomaly_score']:.3f}")
    print(f"⚠️  Risk Indicators: {len(result['risk_indicators'])}")
    print(f"🎲 Fraud Probability: {result['fraud_probability']:.3f}")
    print(f"🎬 Recommended Action: {result['recommended_action']}")
    
    print(f"\n📊 Key Behavioral Features:")
    features = result['behavioral_features']
    print(f"   ⏱️  Session Duration: {features.get('session_duration_minutes', 0):.1f} minutes")
    print(f"   🖱️  Clicks per Minute: {features.get('clicks_per_minute', 0):.1f}")
    print(f"   📄 Pages per Minute: {features.get('pages_per_minute', 0):.1f}")
    print(f"   🌊 Mouse Movement Entropy: {features.get('mouse_movement_entropy', 0):.3f}")
    
    print(f"\n🛡️  Behavioral Fraud Detection Ready!")

if __name__ == "__main__":
    demo_behavioral_fraud_detection()