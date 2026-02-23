# ============================================================
# TRAVEL AGENCY FRAUD DETECTION SYSTEM - PRODUCTION GRADE
# Real-time monitoring with travel-specific fraud patterns
# ============================================================

import numpy as np
import pandas as pd
import json
import ipaddress
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class DynamicCreditManager:
    """Agentic workflow to dynamically manage B2B credit exposure based on evolving trust"""
    
    def __init__(self):
        # Store credit states for agencies
        self.credit_states = {}
        
    def initialize_agency(self, agency_id, initial_limit=10000.0):
        """Set up initial credit profile for a new agency"""
        if agency_id not in self.credit_states:
            self.credit_states[agency_id] = {
                'current_limit': initial_limit,
                'utilized_credit': 0.0,
                'status': 'ACTIVE',  # ACTIVE, PAUSED, RESTRICTED
                'last_adjustment_date': datetime.now(),
                'trust_score': 0.5   # 0.0 (Bad) to 1.0 (Excellent)
            }
            
    def evaluate_credit_exposure(self, agency_id, agency_data, transaction_risk_score, transaction_amount):
        """Dynamically adjust credit limits based on evolving behavior and risk"""
        self.initialize_agency(agency_id)
        state = self.credit_states[agency_id]
        
        # 1. Update Trust Score (Evolving Trust)
        # High transaction risk lowers trust, low risk builds trust
        risk_penalty = (transaction_risk_score - 0.3) * 0.5  # Centered around 0.3 risk
        state['trust_score'] = max(0.0, min(1.0, state['trust_score'] - risk_penalty))
        
        # 2. Evaluate Chargeback Risk
        chargeback_rate = agency_data.get('chargeback_rate', 0.0) if agency_data else 0.0
        
        # 3. Agentic Decision Logic
        action = "MAINTAIN"
        reasoning = []
        adjustment_multiplier = 1.0
        
        # CRITICAL RISK: Pause Credit
        if transaction_risk_score >= 0.6 or chargeback_rate > 0.05:
            action = "PAUSE"
            state['status'] = 'PAUSED'
            adjustment_multiplier = 0.0
            reasoning.append("Critical risk or high chargebacks detected. Post-paid settlement paused.")
            
        # HIGH RISK: Contract Credit
        elif transaction_risk_score >= 0.4 or chargeback_rate > 0.02:
            action = "CONTRACT"
            state['status'] = 'RESTRICTED'
            adjustment_multiplier = 0.5  # Cut limit in half
            reasoning.append("Elevated risk signals. Contracting credit exposure to minimize potential loss.")
            
        # LOW RISK & HIGH TRUST: Expand Credit
        elif transaction_risk_score < 0.2 and state['trust_score'] > 0.7 and chargeback_rate == 0.0:
            # Only expand if they are actually utilizing their credit
            if state['utilized_credit'] + transaction_amount > state['current_limit'] * 0.8:
                action = "EXPAND"
                state['status'] = 'ACTIVE'
                adjustment_multiplier = 1.25  # Increase limit by 25%
                reasoning.append("Consistent low-risk behavior and high trust score. Expanding credit limit to enable scale.")
            else:
                reasoning.append("Good standing, but current credit limit is sufficient for volume.")
        else:
            reasoning.append("Behavior within normal parameters. Maintaining current credit exposure.")
            
        # Apply adjustments
        old_limit = state['current_limit']
        if action in ["EXPAND", "CONTRACT"]:
            state['current_limit'] = old_limit * adjustment_multiplier
            state['last_adjustment_date'] = datetime.now()
            
        # Update utilization if not paused
        if action != "PAUSE":
            state['utilized_credit'] += transaction_amount
            
        return {
            'agency_id': agency_id,
            'action': action,
            'old_limit': old_limit,
            'new_limit': state['current_limit'],
            'utilized_credit': state['utilized_credit'],
            'trust_score': state['trust_score'],
            'reasoning': reasoning[0]
        }

class RealNSIAdapter:
    """
    Production adapter — wraps the trained NetworkThreatDetectionModel
    so centralintelligence.py gets a real ML score instead of static rules.
    """

    def __init__(self):
        self.model = NetworkThreatDetectionModel()
        self.feature_extractor = NetworkSecurityIntelligence()
        self.is_loaded = False
        self._load_model()

    def _load_model(self):
        """Load the trained NSI model from artifacts"""
        model_path = Path("artifacts/models/network_threat_model.joblib")

        if model_path.exists():
            self.model.load_model(model_path)
            self.is_loaded = True
            print("✅ Real NSI model loaded from artifacts/models/network_threat_model.joblib")
        else:
            print("⚠️  NSI model not found. Run: python network_security_intelligence.py train")
            self.is_loaded = False

    def analyze_network_security(self, transaction_data: dict) -> dict:
        """
        Main method called by centralintelligence.py
        Replaces the old static rule-based RealNetworkSecurityIntelligence class.
        
        Input:  transaction_data dict (same format as before)
        Output: same schema as before so NO other code needs to change
        """

        ip_address = transaction_data.get('ip_address', '0.0.0.0')

        # --- Build the feature row the NSI model expects ---
        raw_row = pd.DataFrame([{
            'ip_address':           ip_address,
            'total_requests':       transaction_data.get('total_requests', 20),
            'nginx_request_count':  transaction_data.get('nginx_request_count', 12),
            'api_call_count':       transaction_data.get('api_call_count', 8),
            'redis_ops':            transaction_data.get('redis_ops', 2),
            'unique_endpoints':     transaction_data.get('unique_endpoints', 4),
            'unique_user_agents':   transaction_data.get('unique_user_agents', 1),
            'unique_sessions':      transaction_data.get('unique_sessions', 1),
            'error_rate':           transaction_data.get('error_rate', 0.05),
            'auth_failure_rate':    transaction_data.get('auth_failure_rate', 0.0),
            'avg_request_time':     transaction_data.get('avg_request_time', 0.1),
            'request_time_std':     transaction_data.get('request_time_std', 0.02),

            # IP intelligence (derived from ip_address)
            'is_private_ip':        1 if self._is_private(ip_address) else 0,
            'is_suspicious_ip':     1 if self._is_suspicious(ip_address) else 0,

            # Automation signals (from transaction context)
            'has_bot_user_agent':   1 if transaction_data.get('burst_detected', False) else 0,
            'user_agent_entropy':   transaction_data.get('user_agent_entropy', 1.5),

            # Masking signals
            'vpn_probability':      self._estimate_vpn_probability(ip_address, transaction_data),
            'proxy_probability':    transaction_data.get('proxy_probability', 0.0),

            # Velocity
            'requests_per_minute':  transaction_data.get('velocity_score', 1.0),
            'burst_behavior_score': 1.0 if transaction_data.get('burst_detected', False) else 0.0,

            # Session abuse
            'session_switching_rate':  transaction_data.get('session_switching_rate', 0.1),
            'session_hijacking_score': transaction_data.get('session_hijacking_score', 0.0),
        }])

        # --- Run real ML model if loaded, else graceful fallback ---
        if self.is_loaded:
            try:
                results = self.model.predict_threat(raw_row)
                result  = results[0]

                threat_prob   = result['threat_probability']
                risk_level    = result['risk_level']
                action        = result['recommended_action']

                # Map to attack vectors for explainability
                attack_vectors = self._explain_threats(transaction_data, threat_prob)

                return {
                    'network_threat_score': threat_prob,
                    'threat_level':         risk_level,
                    'attack_vectors':       attack_vectors,
                    'recommended_action':   action,
                    'model_type':           'REAL_ML_NSI_LGBM',
                    'ip_address':           ip_address
                }

            except Exception as e:
                print(f"⚠️  NSI inference error: {e}. Using fallback.")
                return self._fallback_analysis(transaction_data)
        else:
            return self._fallback_analysis(transaction_data)

    def _estimate_vpn_probability(self, ip: str, data: dict) -> float:
        """Estimate VPN probability from available signals"""
        score = 0.0
        # Tor exit nodes and known VPN ranges
        vpn_prefixes = ('185.220.', '199.87.', '104.244.', '45.142.',
                        '198.96.',  '23.129.', '171.25.',  '51.15.')
        if ip.startswith(vpn_prefixes):
            score += 0.8
        if data.get('geo_velocity_anomaly', False):
            score += 0.3
        return min(score, 1.0)

    def _is_private(self, ip: str) -> bool:
        try:
            import ipaddress
            return ipaddress.ip_address(ip).is_private
        except Exception:
            return False

    def _is_suspicious(self, ip: str) -> bool:
        suspicious_prefixes = ('1.2.3.', '5.6.7.', '185.220.', '199.87.')
        return ip.startswith(suspicious_prefixes)

    def _explain_threats(self, data: dict, threat_prob: float) -> list:
        """Generate human-readable threat explanations from signals"""
        vectors = []

        ip = data.get('ip_address', '')
        vpn_prob = self._estimate_vpn_probability(ip, data)

        if vpn_prob > 0.5:
            vectors.append('vpn_or_tor_node_detected')
        if data.get('geo_velocity_anomaly', False):
            vectors.append('impossible_travel_detected')
        if data.get('burst_detected', False) and data.get('velocity_score', 0) > 7:
            vectors.append('botnet_automation_detected')
        if data.get('is_device_shared', False):
            vectors.append('device_farm_suspected')
        if data.get('session_hijacking_score', 0) > 0.5:
            vectors.append('session_hijacking_suspected')
        if threat_prob > 0.6 and not vectors:
            vectors.append('ml_anomaly_pattern_detected')
        if not vectors:
            vectors.append('clean_network_traffic')

        return vectors

    def _fallback_analysis(self, transaction_data: dict) -> dict:
        """Graceful fallback if model is not loaded"""
        return {
            'network_threat_score': 0.1,
            'threat_level':         'LOW',
            'attack_vectors':       ['model_unavailable_fallback'],
            'recommended_action':   'LOG_AND_MONITOR',
            'model_type':           'FALLBACK_RULES',
            'ip_address':           transaction_data.get('ip_address', '0.0.0.0')
        }


# ============================================================
# NOW UPDATE TravelAgencyFraudDetectionSystem.__init__
# Replace:  self.network_security = RealNetworkSecurityIntelligence()
# With:     self.network_security = RealNSIAdapter()
# ============================================================

class TravelAgencyFraudDetectionSystem:
    """Production-grade fraud detection for travel agency financial platform"""
    
    def __init__(self, models_dir="artifacts"):
        self.models_dir = Path(models_dir)
        
        # Load existing fraud models
        self.load_fraud_models()
        
        # Initialize Dynamic Credit Manager & Network Security
        self.credit_manager = DynamicCreditManager()
        self.network_security = RealNSIAdapter()        # ← REAL ML MODEL NOW
        
        # Travel-specific configuration
        self.travel_config = {
            'high_value_threshold': 50000,      # High-value travel bookings
            'velocity_window_hours': 6,         # Travel booking velocity window
            'agency_risk_weights': {
                'new_agency': 0.4,              # New agencies are high risk
                'established_agency': 0.1,      # Established agencies lower risk
                'premium_agency': 0.05          # Premium agencies lowest risk
            },
            'booking_patterns': {
                'last_minute_multiplier': 2.0,  # Last-minute bookings higher risk
                'group_booking_threshold': 10,  # Group booking size
                'international_multiplier': 1.5 # International bookings
            }
        }
        
        # Real-time monitoring state
        self.agency_profiles = {}
        self.transaction_history = []
        self.risk_alerts = []
        
        print("🛫 TRAVEL AGENCY FRAUD DETECTION SYSTEM INITIALIZED")
        print("✈️ Real-time monitoring: ACTIVE")
        print("🔒 Travel-specific fraud patterns: LOADED")
    
    def load_fraud_models(self):
        """Load existing fraud detection models"""
        try:
            from local_inference import DeploymentFraudDetector
            self.network_intelligence = DeploymentFraudDetector()
            self.network_intelligence.load_model()
            
            from behavioral_fraud_detection import BehavioralFraudPredictor
            self.behavioral_predictor = BehavioralFraudPredictor()
            
            from financial_fraud_detection import FinancialRiskPredictor
            self.financial_predictor = FinancialRiskPredictor()
            
            print("✅ Base fraud models loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading base models: {e}")
    
    def analyze_travel_transaction(self, transaction_data, agency_data=None):
        """Comprehensive travel transaction analysis"""
        
        print(f"\n🛫 TRAVEL AGENCY FRAUD ANALYSIS")
        print(f"Agency: {transaction_data.get('agency_name', 'Unknown')}")
        print(f"Transaction: ${transaction_data.get('amount', 0):,.2f}")
        print(f"Booking Type: {transaction_data.get('booking_type', 'Unknown')}")
        print(f"=" * 60)
        
        # 1. Update agency profile
        agency_id = transaction_data.get('agency_id', 'unknown')
        self.update_agency_profile(agency_id, transaction_data, agency_data)
        
        # 2. Base fraud detection
        base_fraud_analysis = self.run_base_fraud_detection(transaction_data)
        
        # 3. Travel-specific fraud analysis
        travel_fraud_analysis = self.run_travel_fraud_analysis(transaction_data, agency_data)
        
        # 4. Real-time behavioral analysis
        realtime_analysis = self.run_realtime_monitoring(transaction_data)
        
        # 4.5 App/Network Security Analysis (VPN, Bots, IP)
        security_analysis = self.network_security.analyze_network_security(transaction_data)
        print(f"\n🛡️ Application & Network Security:")
        print(f"   Threat Score: {security_analysis['network_threat_score']:.3f}")
        print(f"   Detected Vectors: {', '.join(security_analysis['attack_vectors'])}")
        
        # 5. Calculate final risk assessment
        final_assessment = self.calculate_travel_fraud_risk(
            base_fraud_analysis, 
            travel_fraud_analysis, 
            realtime_analysis,
            security_analysis,
            transaction_data
        )
        
        # 6. Agentic Credit Exposure Management
        credit_decision = self.credit_manager.evaluate_credit_exposure(
            agency_id, 
            agency_data, 
            final_assessment['overall_risk_score'],
            transaction_data.get('amount', 0)
        )
        final_assessment['credit_decision'] = credit_decision
        
        print(f"\n💳 DYNAMIC CREDIT EXPOSURE MANAGEMENT:")
        print(f"   Action: {credit_decision['action']}")
        print(f"   Trust Score: {credit_decision['trust_score']:.2f}/1.00")
        print(f"   Credit Limit: ${credit_decision['old_limit']:,.2f} -> ${credit_decision['new_limit']:,.2f}")
        print(f"   Reasoning: {credit_decision['reasoning']}")
        
        # 7. Log transaction for monitoring
        self.log_transaction(transaction_data, final_assessment)
        
        return final_assessment
    
    def run_base_fraud_detection(self, transaction_data):
        """Run base fraud detection models with travel calibration"""
        
        results = {}
        
        # Network Intelligence (with travel calibration)
        if hasattr(self, 'network_intelligence') and self.network_intelligence.is_loaded:
            raw_result = self.network_intelligence.predict_fraud(transaction_data)
            raw_prob = raw_result.get('fraud_probability', 0.5)
            
            # Travel-specific calibration (more aggressive)
            travel_calibrated = min(0.95, raw_prob * 8.0 + 0.2)
            
            results['network_intelligence'] = {
                'fraud_probability': travel_calibrated,
                'risk_level': self.categorize_risk(travel_calibrated),
                'raw_probability': raw_prob
            }
            
            print(f"🧠 Network Intelligence:")
            print(f"   Travel-Calibrated Score: {travel_calibrated:.4f}")
            print(f"   Risk Level: {results['network_intelligence']['risk_level']}")
        
        # Behavioral Analysis (with travel patterns)
        if hasattr(self, 'behavioral_predictor'):
            raw_result = self.behavioral_predictor.predict_behavioral_risk(transaction_data)
            raw_score = raw_result.get('behavioral_risk_score', 0.5)
            
            # Travel behavioral calibration
            travel_behavioral = min(0.95, raw_score * 10.0 + 0.15)
            
            results['behavioral'] = {
                'behavioral_risk_score': travel_behavioral,
                'risk_level': self.categorize_risk(travel_behavioral),
                'key_factors': raw_result.get('key_factors', ['unknown'])
            }
            
            print(f"\n🎭 Behavioral Analysis:")
            print(f"   Travel-Calibrated Score: {travel_behavioral:.4f}")
            print(f"   Key Factors: {', '.join(results['behavioral']['key_factors'])}")
        
        # Financial Risk (with travel amounts)
        if hasattr(self, 'financial_predictor'):
            raw_result = self.financial_predictor.predict_financial_risk(transaction_data)
            raw_score = raw_result.get('financial_risk_score', 0.5)
            
            # Travel financial calibration
            travel_financial = min(0.95, raw_score * 7.0 + 0.1)
            
            results['financial'] = {
                'financial_risk_score': travel_financial,
                'risk_level': self.categorize_risk(travel_financial),
                'risk_factors': raw_result.get('risk_factors', ['unknown'])
            }
            
            print(f"\n💰 Financial Risk:")
            print(f"   Travel-Calibrated Score: {travel_financial:.4f}")
            print(f"   Risk Factors: {', '.join(results['financial']['risk_factors'])}")
        
        return results
    
    def run_travel_fraud_analysis(self, transaction_data, agency_data):
        """Travel-specific fraud pattern analysis"""
        
        travel_risk_score = 0.0
        travel_risk_factors = []
        
        amount = transaction_data.get('amount', 0)
        booking_type = transaction_data.get('booking_type', 'unknown')
        agency_id = transaction_data.get('agency_id', 'unknown')
        
        print(f"\n✈️ Travel-Specific Fraud Analysis:")
        
        # 1. AGENCY RISK ASSESSMENT
        agency_risk, agency_factors = self.assess_agency_risk(agency_id, agency_data)
        travel_risk_score += agency_risk * 0.3
        travel_risk_factors.extend(agency_factors)
        print(f"   Agency Risk: {agency_risk:.3f} - {', '.join(agency_factors)}")
        
        # 2. BOOKING PATTERN ANALYSIS
        booking_risk, booking_factors = self.analyze_booking_patterns(transaction_data)
        travel_risk_score += booking_risk * 0.25
        travel_risk_factors.extend(booking_factors)
        print(f"   Booking Patterns: {booking_risk:.3f} - {', '.join(booking_factors)}")
        
        # 3. TRAVEL FRAUD SIGNATURES
        signature_risk, signature_factors = self.detect_travel_fraud_signatures(transaction_data)
        travel_risk_score += signature_risk * 0.25
        travel_risk_factors.extend(signature_factors)
        print(f"   Fraud Signatures: {signature_risk:.3f} - {', '.join(signature_factors)}")
        
        # 4. PAYMENT METHOD ANALYSIS
        payment_risk, payment_factors = self.analyze_payment_methods(transaction_data)
        travel_risk_score += payment_risk * 0.2
        travel_risk_factors.extend(payment_factors)
        print(f"   Payment Methods: {payment_risk:.3f} - {', '.join(payment_factors)}")
        
        return {
            'travel_risk_score': min(1.0, travel_risk_score),
            'travel_risk_level': self.categorize_risk(travel_risk_score),
            'travel_risk_factors': travel_risk_factors[:5],  # Top 5 factors
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def assess_agency_risk(self, agency_id, agency_data):
        """Assess travel agency risk profile"""
        risk_score = 0.0
        risk_factors = []
        
        if not agency_data:
            risk_score += 0.5
            risk_factors.append('no_agency_data')
            return risk_score, risk_factors
        
        # Agency age and establishment
        agency_age_days = agency_data.get('agency_age_days', 0)
        if agency_age_days < 30:
            risk_score += 0.6
            risk_factors.append('very_new_agency')
        elif agency_age_days < 90:
            risk_score += 0.3
            risk_factors.append('new_agency')
        
        # Agency transaction history
        total_transactions = agency_data.get('total_transactions', 0)
        if total_transactions < 10:
            risk_score += 0.4
            risk_factors.append('limited_transaction_history')
        
        # Chargeback history
        chargeback_rate = agency_data.get('chargeback_rate', 0.0)
        if chargeback_rate > 0.05:  # >5% chargeback rate
            risk_score += 0.7
            risk_factors.append('high_chargeback_rate')
        elif chargeback_rate > 0.02:  # >2% chargeback rate
            risk_score += 0.3
            risk_factors.append('elevated_chargeback_rate')
        
        # License verification
        if not agency_data.get('license_verified', False):
            risk_score += 0.4
            risk_factors.append('unverified_license')
        
        # Rapid growth pattern (potential fraud)
        recent_volume_growth = agency_data.get('volume_growth_30d', 0.0)
        if recent_volume_growth > 5.0:  # >500% growth
            risk_score += 0.5
            risk_factors.append('suspicious_rapid_growth')
        
        return min(1.0, risk_score), risk_factors
    
    def analyze_booking_patterns(self, transaction_data):
        """Analyze travel booking patterns for fraud"""
        risk_score = 0.0
        risk_factors = []
        
        amount = transaction_data.get('amount', 0)
        booking_type = transaction_data.get('booking_type', 'unknown')
        
        # High-value bookings
        if amount > 100000:
            risk_score += 0.5
            risk_factors.append('extremely_high_value_booking')
        elif amount > 50000:
            risk_score += 0.3
            risk_factors.append('high_value_booking')
        
        # Last-minute bookings (higher fraud risk)
        days_to_travel = transaction_data.get('days_to_travel', 30)
        if days_to_travel < 1:
            risk_score += 0.6
            risk_factors.append('same_day_booking')
        elif days_to_travel < 3:
            risk_score += 0.4
            risk_factors.append('last_minute_booking')
        
        # International vs domestic
        is_international = transaction_data.get('is_international', False)
        if is_international:
            risk_score += 0.2
            risk_factors.append('international_booking')
        
        # Group size anomalies
        passenger_count = transaction_data.get('passenger_count', 1)
        if passenger_count > 20:
            risk_score += 0.4
            risk_factors.append('large_group_booking')
        elif passenger_count > 10:
            risk_score += 0.2
            risk_factors.append('group_booking')
        
        # Unusual destinations
        destination_risk = transaction_data.get('destination_risk_score', 0.0)
        risk_score += destination_risk * 0.3
        if destination_risk > 0.5:
            risk_factors.append('high_risk_destination')
        
        return min(1.0, risk_score), risk_factors
    
    def detect_travel_fraud_signatures(self, transaction_data):
        """Detect known travel fraud signatures"""
        risk_score = 0.0
        risk_factors = []
        
        # Credit card testing with small amounts
        amount = transaction_data.get('amount', 0)
        if amount < 50:
            risk_score += 0.7
            risk_factors.append('card_testing_amount')
        
        # Refund fraud patterns
        if transaction_data.get('refund_requested_immediately', False):
            risk_score += 0.8
            risk_factors.append('immediate_refund_request')
        
        # Fake booking patterns
        booking_details = transaction_data.get('booking_details', {})
        if not booking_details.get('passenger_names'):
            risk_score += 0.4
            risk_factors.append('missing_passenger_details')
        
        # Multiple bookings same customer
        customer_bookings_today = transaction_data.get('customer_bookings_today', 0)
        if customer_bookings_today > 5:
            risk_score += 0.6
            risk_factors.append('excessive_daily_bookings')
        elif customer_bookings_today > 2:
            risk_score += 0.3
            risk_factors.append('multiple_daily_bookings')
        
        # Stolen card indicators
        if transaction_data.get('card_country') != transaction_data.get('billing_country'):
            risk_score += 0.4
            risk_factors.append('card_country_mismatch')
        
        return min(1.0, risk_score), risk_factors
    
    def analyze_payment_methods(self, transaction_data):
        """Analyze payment method risks"""
        risk_score = 0.0
        risk_factors = []
        
        payment_method = transaction_data.get('payment_method', 'unknown')
        
        # High-risk payment methods
        if payment_method in ['prepaid_card', 'virtual_card', 'cryptocurrency']:
            risk_score += 0.6
            risk_factors.append(f'high_risk_payment_{payment_method}')
        elif payment_method in ['debit_card']:
            risk_score += 0.2
            risk_factors.append('debit_card_payment')
        
        # New card indicators
        if transaction_data.get('card_age_days', 365) < 30:
            risk_score += 0.3
            risk_factors.append('new_payment_card')
        
        # Card verification failures
        if transaction_data.get('cvv_verification', True) is False:
            risk_score += 0.5
            risk_factors.append('cvv_verification_failed')
        
        if transaction_data.get('avs_verification', True) is False:
            risk_score += 0.4
            risk_factors.append('address_verification_failed')
        
        return min(1.0, risk_score), risk_factors
    
    def run_realtime_monitoring(self, transaction_data):
        """Real-time behavioral monitoring"""
        
        agency_id = transaction_data.get('agency_id', 'unknown')
        current_time = datetime.now()
        
        # Get recent transactions for this agency
        recent_transactions = [
            t for t in self.transaction_history 
            if t['agency_id'] == agency_id and 
            datetime.fromisoformat(t['timestamp']) > current_time - timedelta(hours=6)
        ]
        
        realtime_risk = 0.0
        realtime_factors = []
        
        # Velocity analysis
        if len(recent_transactions) > 10:
            realtime_risk += 0.6
            realtime_factors.append('high_transaction_velocity')
        elif len(recent_transactions) > 5:
            realtime_risk += 0.3
            realtime_factors.append('elevated_transaction_velocity')
        
        # Amount concentration
        current_amount = transaction_data.get('amount', 0)
        recent_amounts = [t.get('amount', 0) for t in recent_transactions]
        
        if recent_amounts:
            total_recent = sum(recent_amounts) + current_amount
            if total_recent > 500000:  # >$500K in 6 hours
                realtime_risk += 0.7
                realtime_factors.append('high_volume_concentration')
            elif total_recent > 200000:  # >$200K in 6 hours
                realtime_risk += 0.4
                realtime_factors.append('elevated_volume_concentration')
        
        # Pattern deviation
        if recent_transactions:
            avg_amount = sum(recent_amounts) / len(recent_amounts)
            if current_amount > avg_amount * 5:
                realtime_risk += 0.4
                realtime_factors.append('amount_pattern_deviation')
        
        print(f"\n⚡ Real-time Monitoring:")
        print(f"   Recent Transactions (6h): {len(recent_transactions)}")
        print(f"   Realtime Risk: {realtime_risk:.3f}")
        print(f"   Risk Factors: {', '.join(realtime_factors) if realtime_factors else 'None'}")
        
        return {
            'realtime_risk_score': min(1.0, realtime_risk),
            'realtime_factors': realtime_factors,
            'recent_transaction_count': len(recent_transactions),
            'monitoring_window_hours': 6
        }
    
    def calculate_travel_fraud_risk(self, base_analysis, travel_analysis, realtime_analysis, security_analysis, transaction_data):
        """Calculate final travel fraud risk assessment"""
        
        # Weighted scoring for travel platform (Now includes App Security!)
        weights = {
            'network_intelligence': 0.15,    
            'behavioral': 0.15,             
            'financial': 0.10,              
            'travel_specific': 0.25,        
            'realtime_monitoring': 0.15,
            'app_security': 0.20            # 20% weight for VPN/Bot detection
        }
        
        # Extract scores
        network_score = base_analysis.get('network_intelligence', {}).get('fraud_probability', 0.5)
        behavioral_score = base_analysis.get('behavioral', {}).get('behavioral_risk_score', 0.5)
        financial_score = base_analysis.get('financial', {}).get('financial_risk_score', 0.5)
        travel_score = travel_analysis.get('travel_risk_score', 0.5)
        realtime_score = realtime_analysis.get('realtime_risk_score', 0.0)
        security_score = security_analysis.get('network_threat_score', 0.0)
        
        # Calculate weighted risk
        overall_risk = (
            network_score * weights['network_intelligence'] +
            behavioral_score * weights['behavioral'] +
            financial_score * weights['financial'] +
            travel_score * weights['travel_specific'] +
            realtime_score * weights['realtime_monitoring'] +
            security_score * weights['app_security']
        )
        
        # Travel-specific decision thresholds (MORE AGGRESSIVE)
        amount = transaction_data.get('amount', 0)
        
        if overall_risk >= 0.55 or amount > 200000:  # Very aggressive for high amounts
            recommendation = "⛔ BLOCK TRANSACTION IMMEDIATELY"
            risk_category = "FRAUD_DETECTED"
            confidence = 0.98
            action_code = "BLOCK"
        elif overall_risk >= 0.35 or amount > 100000:  # Aggressive for large amounts
            recommendation = "⚠️ MANUAL REVIEW REQUIRED"
            risk_category = "HIGH_RISK"
            confidence = 0.95
            action_code = "MANUAL_REVIEW"
        elif overall_risk >= 0.20:
            recommendation = "👀 ENHANCED MONITORING"
            risk_category = "MEDIUM_RISK"
            confidence = 0.90
            action_code = "MONITOR"
        else:
            recommendation = "✅ APPROVE TRANSACTION"
            risk_category = "LOW_RISK"
            confidence = 0.85
            action_code = "APPROVE"
        
        print(f"\n🎯 FINAL TRAVEL FRAUD ASSESSMENT:")
        print(f"   Overall Risk Score: {overall_risk:.4f}")
        print(f"   Recommendation: {recommendation}")
        print(f"   Risk Category: {risk_category}")
        print(f"   Confidence: {confidence:.3f}")
        print(f"   Action Code: {action_code}")
        
        # Generate alerts if needed
        if action_code in ["BLOCK", "MANUAL_REVIEW"]:
            self.generate_fraud_alert(transaction_data, overall_risk, recommendation)
        
        return {
            'overall_risk_score': overall_risk,
            'recommendation': recommendation,
            'risk_category': risk_category,
            'confidence': confidence,
            'action_code': action_code,
            'base_analysis': base_analysis,
            'travel_analysis': travel_analysis,
            'realtime_analysis': realtime_analysis,
            'security_analysis': security_analysis,
            'weights_used': weights,
            'assessment_timestamp': datetime.now().isoformat()
        }
    
    def update_agency_profile(self, agency_id, transaction_data, agency_data):
        """Update agency profile with new transaction"""
        if agency_id not in self.agency_profiles:
            self.agency_profiles[agency_id] = {
                'first_seen': datetime.now(),
                'total_transactions': 0,
                'total_volume': 0,
                'risk_alerts': 0,
                'last_activity': datetime.now()
            }
        
        profile = self.agency_profiles[agency_id]
        profile['total_transactions'] += 1
        profile['total_volume'] += transaction_data.get('amount', 0)
        profile['last_activity'] = datetime.now()
        
        # Update from agency_data if provided
        if agency_data:
            profile.update({k: v for k, v in agency_data.items() 
                          if k not in ['first_seen', 'total_transactions', 'total_volume']})
    
    def log_transaction(self, transaction_data, assessment):
        """Log transaction for continuous monitoring"""
        log_entry = {
            'transaction_id': transaction_data.get('transaction_id', 'unknown'),
            'agency_id': transaction_data.get('agency_id', 'unknown'),
            'amount': transaction_data.get('amount', 0),
            'risk_score': assessment['overall_risk_score'],
            'action_code': assessment['action_code'],
            'timestamp': datetime.now().isoformat()
        }
        
        self.transaction_history.append(log_entry)
        
        # Keep only recent transactions (last 24 hours)
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.transaction_history = [
            t for t in self.transaction_history 
            if datetime.fromisoformat(t['timestamp']) > cutoff_time
        ]
    
    def generate_fraud_alert(self, transaction_data, risk_score, recommendation):
        """Generate fraud alert for high-risk transactions"""
        alert = {
            'alert_id': f"ALERT_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'agency_id': transaction_data.get('agency_id', 'unknown'),
            'transaction_id': transaction_data.get('transaction_id', 'unknown'),
            'amount': transaction_data.get('amount', 0),
            'risk_score': risk_score,
            'recommendation': recommendation,
            'priority': 'CRITICAL' if 'BLOCK' in recommendation else 'HIGH',
            'timestamp': datetime.now().isoformat()
        }
        
        self.risk_alerts.append(alert)
        print(f"\n🚨 FRAUD ALERT GENERATED:")
        print(f"   Alert ID: {alert['alert_id']}")
        print(f"   Priority: {alert['priority']}")
        print(f"   Action Required: {recommendation}")
    
    def categorize_risk(self, score):
        """Categorize risk levels"""
        if score >= 0.7:
            return 'CRITICAL'
        elif score >= 0.5:
            return 'HIGH'
        elif score >= 0.3:
            return 'MEDIUM'
        elif score >= 0.15:
            return 'LOW'
        else:
            return 'MINIMAL'
    
    def get_agency_monitoring_dashboard(self, agency_id):
        """Get real-time agency monitoring dashboard"""
        profile = self.agency_profiles.get(agency_id, {})
        
        # Recent transactions
        recent_transactions = [
            t for t in self.transaction_history 
            if t['agency_id'] == agency_id
        ]
        
        # Risk metrics
        avg_risk_score = np.mean([t['risk_score'] for t in recent_transactions]) if recent_transactions else 0
        high_risk_count = len([t for t in recent_transactions if t['risk_score'] > 0.5])
        
        # Alerts
        agency_alerts = [a for a in self.risk_alerts if a['agency_id'] == agency_id]
        
        dashboard = {
            'agency_id': agency_id,
            'profile_summary': profile,
            'recent_transactions_24h': len(recent_transactions),
            'average_risk_score': avg_risk_score,
            'high_risk_transactions': high_risk_count,
            'active_alerts': len([a for a in agency_alerts if 
                               datetime.fromisoformat(a['timestamp']) > datetime.now() - timedelta(hours=1)]),
            'total_volume_24h': sum([t['amount'] for t in recent_transactions]),
            'risk_trend': 'INCREASING' if avg_risk_score > 0.4 else 'STABLE',
            'last_updated': datetime.now().isoformat()
        }
        
        return dashboard


def demo_travel_agency_fraud_system():
    """Demo the travel agency fraud detection system"""
    
    print("🛫 TRAVEL AGENCY FRAUD DETECTION SYSTEM DEMO")
    print("=" * 60)
    
    # Initialize system
    fraud_system = TravelAgencyFraudDetectionSystem()
    
    print("\n" + "="*60)
    print("SCENARIO 1: SUSPICIOUS NEW AGENCY (Credit Contraction/Pause)")
    print("="*60)
    
    # Mock agency data
    suspicious_agency_data = {
        'agency_age_days': 15,           # Very new agency
        'total_transactions': 5,         # Limited history
        'chargeback_rate': 0.08,         # High chargeback rate
        'license_verified': False,       # Unverified
        'volume_growth_30d': 8.0         # Suspicious growth
    }
    
    # Suspicious travel transaction
    suspicious_transaction = {
        'transaction_id': 'TXN_TRAVEL_SUSPICIOUS_001',
        'agency_id': 'AGENCY_NEW_001',
        'agency_name': 'QuickTravel LLC',
        'amount': 150000.0,              # High amount
        'booking_type': 'group_international',
        'passenger_count': 25,           # Large group
        'days_to_travel': 2,             # Last minute
        'is_international': True,
        'destination_risk_score': 0.7,   # High-risk destination
        'payment_method': 'prepaid_card', # Risky payment
        'card_age_days': 15,             # New card
        'cvv_verification': False,       # Failed CVV
        'customer_bookings_today': 3,    # Multiple bookings
        'type': 'TRANSFER',              # For base models
        'oldbalanceOrg': 0,              # For base models
        'newbalanceOrig': 0,
        'step': 150,
        'device_seen_count': 1,
        'is_device_shared': True,
        'velocity_score': 9.5,
        'burst_detected': True,
        'user_frequency': 25,
        'ip_address': '185.220.101.2',   # ADDED: Known Tor/VPN IP
        'geo_velocity_anomaly': True     # ADDED: Impossible travel
    }
    
    # Analyze the suspicious transaction
    result1 = fraud_system.analyze_travel_transaction(
        suspicious_transaction, 
        suspicious_agency_data
    )
    
    print("\n" + "="*60)
    print("SCENARIO 2: ESTABLISHED TRUSTED AGENCY (Credit Expansion)")
    print("="*60)
    
    # Pre-warm the credit manager to simulate an established agency nearing their limit
    fraud_system.credit_manager.initialize_agency('AGENCY_TRUSTED_002', initial_limit=50000.0)
    fraud_system.credit_manager.credit_states['AGENCY_TRUSTED_002']['trust_score'] = 0.85
    fraud_system.credit_manager.credit_states['AGENCY_TRUSTED_002']['utilized_credit'] = 45000.0
    
    trusted_agency_data = {
        'agency_age_days': 450,          
        'total_transactions': 1250,      
        'chargeback_rate': 0.00,         # Zero chargebacks
        'license_verified': True,        
        'volume_growth_30d': 1.1         # Steady growth
    }
    
    trusted_transaction = {
        'transaction_id': 'TXN_TRAVEL_TRUSTED_002',
        'agency_id': 'AGENCY_TRUSTED_002',
        'agency_name': 'Premium Corporate Travel',
        'amount': 12000.0,               
        'booking_type': 'corporate_domestic',
        'passenger_count': 2,           
        'days_to_travel': 45,            # Booked well in advance
        'is_international': False,
        'destination_risk_score': 0.1,   
        'payment_method': 'corporate_card', 
        'card_age_days': 300,            
        'cvv_verification': True,       
        'customer_bookings_today': 1,    
        'type': 'PAYMENT',              
        'oldbalanceOrg': 50000,          
        'newbalanceOrig': 38000,
        'step': 150,
        'device_seen_count': 50,
        'is_device_shared': False,
        'velocity_score': 2.0,
        'burst_detected': False,
        'user_frequency': 5
    }
    
    result2 = fraud_system.analyze_travel_transaction(
        trusted_transaction, 
        trusted_agency_data
    )
    
    print(f"\n" + "="*60)
    print(f"📊 TRAVEL FRAUD & CREDIT ASSESSMENT SUMMARY:")
    print(f"   Agency 1: {suspicious_transaction['agency_name']}")
    print(f"   Risk Score: {result1['overall_risk_score']:.4f} -> Action: {result1['action_code']}")
    print(f"   Credit Decision: {result1['credit_decision']['action']} ({result1['credit_decision']['reasoning']})")
    print(f"   ---")
    print(f"   Agency 2: {trusted_transaction['agency_name']}")
    print(f"   Risk Score: {result2['overall_risk_score']:.4f} -> Action: {result2['action_code']}")
    print(f"   Credit Decision: {result2['credit_decision']['action']} ({result2['credit_decision']['reasoning']})")
    print(f"="*60)
    
    return result1, result2

if __name__ == "__main__":
    demo_travel_agency_fraud_system()