# ============================================================
# DATA PIPELINE FOR TRAVEL PLATFORM LOG AGGREGATION
# Real-time log processing and feature extraction
# ============================================================

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from pathlib import Path
import aioredis
import asyncpg
from kafka import KafkaConsumer, KafkaProducer
from elasticsearch import AsyncElasticsearch
import re
from dataclasses import dataclass
from collections import defaultdict, deque

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# DATA MODELS FOR LOG PROCESSING
# ============================================================

@dataclass
class ApplicationLog:
    """Application log entry"""
    timestamp: datetime
    log_level: str
    service: str
    user_id: Optional[str]
    session_id: Optional[str]
    ip_address: Optional[str]
    endpoint: str
    method: str
    status_code: int
    response_time: float
    user_agent: Optional[str]
    message: str
    additional_data: Dict

@dataclass
class DatabaseLog:
    """Database transaction log"""
    timestamp: datetime
    query_type: str
    table_name: str
    user_id: Optional[str]
    transaction_id: Optional[str]
    execution_time: float
    rows_affected: int
    query_hash: str

@dataclass
class NetworkEvent:
    """Network-level event"""
    timestamp: datetime
    event_type: str  # login, transaction, api_call, etc.
    source_user: str
    target_entity: str
    amount: Optional[float]
    metadata: Dict

# ============================================================
# LOG AGGREGATION AND PROCESSING
# ============================================================

class TravelPlatformLogProcessor:
    """Process and aggregate logs from travel platform"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.redis = None
        self.postgres = None
        self.elasticsearch = None
        self.kafka_consumer = None
        self.kafka_producer = None
        
        # Real-time feature tracking
        self.user_sessions = defaultdict(dict)
        self.api_call_patterns = defaultdict(deque)
        self.transaction_flows = defaultdict(list)
        
        # Network graph updates
        self.network_updates = deque(maxlen=1000)
        
    async def initialize(self):
        """Initialize all connections"""
        
        # Redis connection
        self.redis = aioredis.from_url(
            f"redis://{self.config['redis']['host']}:{self.config['redis']['port']}"
        )
        
        # PostgreSQL connection
        self.postgres = await asyncpg.create_pool(
            host=self.config['postgres']['host'],
            database=self.config['postgres']['database'],
            user=self.config['postgres']['user'],
            password=self.config['postgres']['password']
        )
        
        # Elasticsearch connection
        self.elasticsearch = AsyncElasticsearch([
            f"{self.config['elasticsearch']['host']}:{self.config['elasticsearch']['port']}"
        ])
        
        # Kafka setup
        self.kafka_consumer = KafkaConsumer(
            'travel-platform-logs',
            bootstrap_servers=self.config['kafka']['brokers'],
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )
        
        self.kafka_producer = KafkaProducer(
            bootstrap_servers=self.config['kafka']['brokers'],
            value_serializer=lambda x: json.dumps(x).encode('utf-8')
        )
        
        logger.info("Log processor initialized")
    
    # ============================================================
    # LOG INGESTION FROM MULTIPLE SOURCES
    # ============================================================
    
    async def process_application_logs(self):
        """Process application logs from Kafka stream"""
        
        for message in self.kafka_consumer:
            try:
                log_data = message.value
                log_entry = self.parse_application_log(log_data)
                
                if log_entry:
                    # Extract network features
                    await self.extract_network_features_from_log(log_entry)
                    
                    # Update user session tracking
                    await self.update_user_session(log_entry)
                    
                    # Detect anomalous patterns
                    await self.detect_log_anomalies(log_entry)
                    
                    # Store in Elasticsearch for search
                    await self.store_log_elasticsearch(log_entry)
                    
            except Exception as e:
                logger.error(f"Error processing log: {e}")
    
    def parse_application_log(self, log_data: Dict) -> Optional[ApplicationLog]:
        """Parse raw log data into structured format"""
        
        try:
            # Extract user_id from various places
            user_id = (
                log_data.get('user_id') or
                self.extract_user_from_url(log_data.get('url', '')) or
                self.extract_user_from_headers(log_data.get('headers', {}))
            )
            
            return ApplicationLog(
                timestamp=datetime.fromisoformat(log_data['timestamp']),
                log_level=log_data.get('level', 'INFO'),
                service=log_data.get('service', 'unknown'),
                user_id=user_id,
                session_id=log_data.get('session_id'),
                ip_address=log_data.get('ip_address'),
                endpoint=log_data.get('endpoint', ''),
                method=log_data.get('method', 'GET'),
                status_code=log_data.get('status_code', 200),
                response_time=log_data.get('response_time', 0.0),
                user_agent=log_data.get('user_agent'),
                message=log_data.get('message', ''),
                additional_data=log_data.get('additional_data', {})
            )
        except Exception as e:
            logger.error(f"Error parsing log: {e}")
            return None
    
    def extract_user_from_url(self, url: str) -> Optional[str]:
        """Extract user ID from URL patterns"""
        patterns = [
            r'/users/([^/]+)/',
            r'/api/v\d+/users/([^/]+)',
            r'user_id=([^&]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None
    
    def extract_user_from_headers(self, headers: Dict) -> Optional[str]:
        """Extract user ID from request headers"""
        user_headers = ['X-User-ID', 'Authorization', 'X-Session-User']
        
        for header in user_headers:
            if header in headers:
                if header == 'Authorization':
                    # Extract from JWT token (simplified)
                    token = headers[header].replace('Bearer ', '')
                    # In real implementation, decode JWT
                    return self.decode_user_from_token(token)
                else:
                    return headers[header]
        return None
    
    def decode_user_from_token(self, token: str) -> Optional[str]:
        """Decode user ID from JWT token"""
        # Simplified - in real implementation, use proper JWT library
        try:
            # This would normally decode the JWT
            return f"user_from_token_{hash(token) % 10000}"
        except:
            return None
    
    # ============================================================
    # NETWORK FEATURE EXTRACTION FROM LOGS
    # ============================================================
    
    async def extract_network_features_from_log(self, log_entry: ApplicationLog):
        """Extract network intelligence features from log entries"""
        
        if not log_entry.user_id:
            return
        
        # Track API call patterns
        api_pattern = {
            'timestamp': log_entry.timestamp,
            'endpoint': log_entry.endpoint,
            'method': log_entry.method,
            'status': log_entry.status_code,
            'response_time': log_entry.response_time
        }
        
        # Update user's API call history
        user_api_calls = self.api_call_patterns[log_entry.user_id]
        user_api_calls.append(api_pattern)
        
        # Keep only recent calls (last 1000)
        if len(user_api_calls) > 1000:
            user_api_calls.popleft()
        
        # Extract network events
        network_event = await self.extract_network_event(log_entry)
        if network_event:
            await self.process_network_event(network_event)
        
        # Update Redis with real-time features
        await self.update_realtime_features(log_entry)
    
    async def extract_network_event(self, log_entry: ApplicationLog) -> Optional[NetworkEvent]:
        """Extract network events from log entries"""
        
        # Booking transaction
        if '/bookings' in log_entry.endpoint and log_entry.method == 'POST':
            # Extract target agency from additional data
            agency_id = log_entry.additional_data.get('agency_id')
            amount = log_entry.additional_data.get('amount')
            
            if agency_id:
                return NetworkEvent(
                    timestamp=log_entry.timestamp,
                    event_type='booking_transaction',
                    source_user=log_entry.user_id,
                    target_entity=agency_id,
                    amount=amount,
                    metadata={
                        'booking_type': log_entry.additional_data.get('booking_type'),
                        'destination': log_entry.additional_data.get('destination'),
                        'ip_address': log_entry.ip_address
                    }
                )
        
        # User-to-user interactions (referrals, reviews, etc.)
        elif '/users/' in log_entry.endpoint and 'review' in log_entry.endpoint:
            target_user = self.extract_target_user_from_endpoint(log_entry.endpoint)
            if target_user:
                return NetworkEvent(
                    timestamp=log_entry.timestamp,
                    event_type='user_interaction',
                    source_user=log_entry.user_id,
                    target_entity=target_user,
                    amount=None,
                    metadata={'interaction_type': 'review'}
                )
        
        # Agency interactions
        elif '/agencies/' in log_entry.endpoint:
            agency_id = self.extract_agency_from_endpoint(log_entry.endpoint)
            if agency_id:
                return NetworkEvent(
                    timestamp=log_entry.timestamp,
                    event_type='agency_interaction',
                    source_user=log_entry.user_id,
                    target_entity=agency_id,
                    amount=None,
                    metadata={'interaction_type': 'view'}
                )
        
        return None
    
    def extract_target_user_from_endpoint(self, endpoint: str) -> Optional[str]:
        """Extract target user ID from endpoint"""
        match = re.search(r'/users/([^/]+)', endpoint)
        return match.group(1) if match else None
    
    def extract_agency_from_endpoint(self, endpoint: str) -> Optional[str]:
        """Extract agency ID from endpoint"""
        match = re.search(r'/agencies/([^/]+)', endpoint)
        return match.group(1) if match else None
    
    async def process_network_event(self, event: NetworkEvent):
        """Process network events for graph updates"""
        
        self.network_updates.append(event)
        
        # Update network metrics in Redis
        await self.update_network_metrics(event)
        
        # If it's a transaction, update transaction flow tracking
        if event.event_type == 'booking_transaction':
            await self.update_transaction_flows(event)
    
    # ============================================================
    # REAL-TIME FEATURE UPDATES
    # ============================================================
    
    async def update_realtime_features(self, log_entry: ApplicationLog):
        """Update real-time features in Redis"""
        
        user_id = log_entry.user_id
        timestamp = log_entry.timestamp
        
        # Update user activity metrics
        activity_key = f"user_activity:{user_id}:daily"
        await self.redis.hincrby(activity_key, timestamp.strftime('%Y-%m-%d'), 1)
        await self.redis.expire(activity_key, 86400 * 7)  # 7 days
        
        # Update API call frequency
        api_freq_key = f"api_frequency:{user_id}:hourly"
        await self.redis.hincrby(api_freq_key, timestamp.strftime('%Y-%m-%d:%H'), 1)
        await self.redis.expire(api_freq_key, 86400)  # 24 hours
        
        # Update error rate
        if log_entry.status_code >= 400:
            error_key = f"user_errors:{user_id}:hourly"
            await self.redis.hincrby(error_key, timestamp.strftime('%Y-%m-%d:%H'), 1)
            await self.redis.expire(error_key, 86400)
        
        # Update response time patterns
        if log_entry.response_time > 0:
            response_time_key = f"response_times:{user_id}"
            await self.redis.lpush(response_time_key, log_entry.response_time)
            await self.redis.ltrim(response_time_key, 0, 100)  # Keep last 100
            await self.redis.expire(response_time_key, 3600)
    
    async def update_network_metrics(self, event: NetworkEvent):
        """Update network metrics in Redis"""
        
        # Update degree centrality approximations
        source_degree_key = f"node_degree:{event.source_user}"
        target_degree_key = f"node_degree:{event.target_entity}"
        
        await self.redis.incr(source_degree_key)
        await self.redis.incr(target_degree_key)
        await self.redis.expire(source_degree_key, 86400)
        await self.redis.expire(target_degree_key, 86400)
        
        # Update connection weights
        if event.amount:
            weight_key = f"connection_weight:{event.source_user}:{event.target_entity}"
            current_weight = await self.redis.get(weight_key)
            new_weight = float(current_weight or 0) + event.amount
            await self.redis.setex(weight_key, 86400, new_weight)
        
        # Update transaction counts
        txn_count_key = f"txn_count:{event.source_user}:{event.target_entity}"
        await self.redis.incr(txn_count_key)
        await self.redis.expire(txn_count_key, 86400)
    
    async def update_transaction_flows(self, event: NetworkEvent):
        """Update transaction flow patterns"""
        
        flow_data = {
            'timestamp': event.timestamp.isoformat(),
            'source': event.source_user,
            'target': event.target_entity,
            'amount': event.amount,
            'metadata': event.metadata
        }
        
        # Store in user's transaction flow
        flow_key = f"transaction_flow:{event.source_user}"
        await self.redis.lpush(flow_key, json.dumps(flow_data))
        await self.redis.ltrim(flow_key, 0, 500)  # Keep last 500 transactions
        await self.redis.expire(flow_key, 86400 * 30)  # 30 days
    
    # ============================================================
    # ANOMALY DETECTION FROM LOGS
    # ============================================================
    
    async def detect_log_anomalies(self, log_entry: ApplicationLog):
        """Detect anomalous patterns in log data"""
        
        anomalies = []
        
        # High error rate
        if log_entry.status_code >= 500:
            error_count = await self.get_recent_error_count(log_entry.user_id)
            if error_count > 10:  # More than 10 errors in last hour
                anomalies.append({
                    'type': 'high_error_rate',
                    'severity': 'medium',
                    'details': f'User has {error_count} errors in last hour'
                })
        
        # Unusual response times
        if log_entry.response_time > 5.0:  # More than 5 seconds
            avg_response_time = await self.get_average_response_time(log_entry.user_id)
            if log_entry.response_time > avg_response_time * 3:
                anomalies.append({
                    'type': 'unusual_response_time',
                    'severity': 'low',
                    'details': f'Response time {log_entry.response_time}s vs avg {avg_response_time}s'
                })
        
        # Rapid API calls
        api_call_rate = await self.get_api_call_rate(log_entry.user_id)
        if api_call_rate > 100:  # More than 100 calls per hour
            anomalies.append({
                'type': 'high_api_frequency',
                'severity': 'high',
                'details': f'API call rate: {api_call_rate} calls/hour'
            })
        
        # Store anomalies if found
        if anomalies:
            await self.store_anomalies(log_entry, anomalies)
    
    async def get_recent_error_count(self, user_id: str) -> int:
        """Get recent error count for user"""
        error_key = f"user_errors:{user_id}:hourly"
        current_hour = datetime.utcnow().strftime('%Y-%m-%d:%H')
        count = await self.redis.hget(error_key, current_hour)
        return int(count) if count else 0
    
    async def get_average_response_time(self, user_id: str) -> float:
        """Get average response time for user"""
        response_time_key = f"response_times:{user_id}"
        times = await self.redis.lrange(response_time_key, 0, -1)
        if times:
            return sum(float(t) for t in times) / len(times)
        return 1.0  # Default
    
    async def get_api_call_rate(self, user_id: str) -> int:
        """Get API call rate for user"""
        api_freq_key = f"api_frequency:{user_id}:hourly"
        current_hour = datetime.utcnow().strftime('%Y-%m-%d:%H')
        count = await self.redis.hget(api_freq_key, current_hour)
        return int(count) if count else 0
    
    async def store_anomalies(self, log_entry: ApplicationLog, anomalies: List[Dict]):
        """Store detected anomalies"""
        
        anomaly_data = {
            'timestamp': log_entry.timestamp.isoformat(),
            'user_id': log_entry.user_id,
            'log_entry': {
                'endpoint': log_entry.endpoint,
                'method': log_entry.method,
                'status_code': log_entry.status_code,
                'response_time': log_entry.response_time
            },
            'anomalies': anomalies
        }
        
        # Store in Elasticsearch
        await self.elasticsearch.index(
            index='travel-platform-anomalies',
            body=anomaly_data
        )
        
        # Alert if high severity
        high_severity_anomalies = [a for a in anomalies if a['severity'] == 'high']
        if high_severity_anomalies:
            await self.send_anomaly_alert(log_entry.user_id, high_severity_anomalies)
    
    # ============================================================
    # DATA STORAGE AND RETRIEVAL
    # ============================================================
    
    async def store_log_elasticsearch(self, log_entry: ApplicationLog):
        """Store log entry in Elasticsearch"""
        
        doc = {
            'timestamp': log_entry.timestamp,
            'log_level': log_entry.log_level,
            'service': log_entry.service,
            'user_id': log_entry.user_id,
            'session_id': log_entry.session_id,
            'ip_address': log_entry.ip_address,
            'endpoint': log_entry.endpoint,
            'method': log_entry.method,
            'status_code': log_entry.status_code,
            'response_time': log_entry.response_time,
            'message': log_entry.message,
            'additional_data': log_entry.additional_data
        }
        
        await self.elasticsearch.index(
            index=f"travel-platform-logs-{log_entry.timestamp.strftime('%Y-%m')}",
            body=doc
        )
    
    async def get_user_log_patterns(self, user_id: str, hours: int = 24) -> Dict:
        """Get user's log patterns for feature extraction"""
        
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)
        
        query = {
            "query": {
                "bool": {
                    "must": [
                        {"term": {"user_id": user_id}},
                        {"range": {
                            "timestamp": {
                                "gte": start_time.isoformat(),
                                "lte": end_time.isoformat()
                            }
                        }}
                    ]
                }
            },
            "size": 1000,
            "sort": [{"timestamp": {"order": "desc"}}]
        }
        
        response = await self.elasticsearch.search(
            index="travel-platform-logs-*",
            body=query
        )
        
        logs = response['hits']['hits']
        
        # Analyze patterns
        return {
            'total_requests': len(logs),
            'unique_endpoints': len(set(log['_source']['endpoint'] for log in logs)),
            'error_rate': len([log for log in logs if log['_source']['status_code'] >= 400]) / len(logs) if logs else 0,
            'avg_response_time': np.mean([log['_source']['response_time'] for log in logs]) if logs else 0,
            'request_distribution': self.analyze_request_distribution(logs),
            'temporal_pattern': self.analyze_temporal_pattern(logs)
        }
    
    def analyze_request_distribution(self, logs: List) -> Dict:
        """Analyze request distribution patterns"""
        endpoints = [log['_source']['endpoint'] for log in logs]
        from collections import Counter
        return dict(Counter(endpoints).most_common(10))
    
    def analyze_temporal_pattern(self, logs: List) -> Dict:
        """Analyze temporal access patterns"""
        hours = [datetime.fromisoformat(log['_source']['timestamp']).hour for log in logs]
        from collections import Counter
        return dict(Counter(hours))
    
    async def send_anomaly_alert(self, user_id: str, anomalies: List[Dict]):
        """Send alert for detected anomalies"""
        
        alert_data = {
            'type': 'LOG_ANOMALY',
            'user_id': user_id,
            'anomalies': anomalies,
            'timestamp': datetime.utcnow().isoformat(),
            'severity': 'HIGH'
        }
        
        # Send to alerting system
        self.kafka_producer.send('travel-platform-alerts', alert_data)
        logger.warning(f"Anomaly alert sent for user {user_id}: {anomalies}")

# ============================================================
# CONFIGURATION AND RUNNER
# ============================================================

async def main():
    """Main runner for log processor"""
    
    config = {
        'redis': {'host': 'localhost', 'port': 6379},
        'postgres': {
            'host': 'localhost',
            'database': 'travel_platform',
            'user': 'postgres',
            'password': 'password'
        },
        'elasticsearch': {'host': 'localhost', 'port': 9200},
        'kafka': {'brokers': ['localhost:9092']}
    }
    
    processor = TravelPlatformLogProcessor(config)
    await processor.initialize()
    
    # Start processing logs
    await processor.process_application_logs()

if __name__ == "__main__":
    asyncio.run(main())