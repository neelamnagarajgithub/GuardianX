# ============================================================
# TRAVEL PLATFORM API LAYER
# FastAPI implementation for real-time fraud detection
# ============================================================

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from typing import Optional, List, Dict, Any
import asyncio
import json
import logging
from datetime import datetime
import uvicorn
from real_time_travel_intelligence import TravelNetworkIntelligence, TravelTransaction

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Travel Platform Network Intelligence API",
    description="Real-time fraud detection for travel bookings",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global intelligence engine
travel_intel = None

@app.on_event("startup")
async def startup_event():
    """Initialize the intelligence engine on startup"""
    global travel_intel
    travel_intel = TravelNetworkIntelligence()
    logger.info("Travel Intelligence API started")

# ============================================================
# API MODELS
# ============================================================

class BookingRequest(BaseModel):
    """Booking request model"""
    transaction_id: str
    user_id: str
    agency_id: str
    booking_type: str
    amount: float
    currency: str = "USD"
    travel_date: str  # ISO format
    destination: str
    source_country: str
    payment_method: str
    device_fingerprint: str
    ip_address: str
    session_id: str
    user_agent: str
    
    @validator('booking_type')
    def validate_booking_type(cls, v):
        valid_types = ['flight', 'hotel', 'package', 'car_rental']
        if v not in valid_types:
            raise ValueError(f'booking_type must be one of {valid_types}')
        return v
    
    @validator('amount')
    def validate_amount(cls, v):
        if v <= 0:
            raise ValueError('amount must be positive')
        return v

class RiskResponse(BaseModel):
    """Risk scoring response"""
    transaction_id: str
    fraud_probability: float
    fraud_prediction: int
    risk_bucket: str
    risk_factors: List[str]
    recommendation: str
    timestamp: str
    processing_time_ms: Optional[float] = None

class BatchScoringRequest(BaseModel):
    """Batch scoring request"""
    transactions: List[BookingRequest]
    
class UserRiskProfile(BaseModel):
    """User risk profile request"""
    user_id: str
    days_lookback: int = 90

class AgencyRiskProfile(BaseModel):
    """Agency risk profile request"""
    agency_id: str
    days_lookback: int = 90

# ============================================================
# CORE API ENDPOINTS
# ============================================================

@app.post("/api/v1/score-transaction", response_model=RiskResponse)
async def score_transaction(request: BookingRequest):
    """Score a single transaction for fraud risk"""
    
    start_time = datetime.utcnow()
    
    try:
        # Convert to internal model
        transaction = TravelTransaction(
            transaction_id=request.transaction_id,
            user_id=request.user_id,
            agency_id=request.agency_id,
            booking_type=request.booking_type,
            amount=request.amount,
            currency=request.currency,
            booking_date=datetime.utcnow(),
            travel_date=datetime.fromisoformat(request.travel_date),
            destination=request.destination,
            source_country=request.source_country,
            payment_method=request.payment_method,
            device_fingerprint=request.device_fingerprint,
            ip_address=request.ip_address,
            session_id=request.session_id,
            user_agent=request.user_agent
        )
        
        # Score transaction
        result = await travel_intel.score_transaction_risk(transaction)
        
        # Calculate processing time
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return RiskResponse(
            transaction_id=result['transaction_id'],
            fraud_probability=result['fraud_probability'],
            fraud_prediction=result['fraud_prediction'],
            risk_bucket=result['risk_bucket'],
            risk_factors=result.get('risk_factors', []),
            recommendation=result.get('recommendation', 'Manual review required'),
            timestamp=result['timestamp'],
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error scoring transaction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/batch-score")
async def batch_score_transactions(request: BatchScoringRequest):
    """Score multiple transactions"""
    
    start_time = datetime.utcnow()
    results = []
    
    try:
        for booking in request.transactions:
            transaction = TravelTransaction(
                transaction_id=booking.transaction_id,
                user_id=booking.user_id,
                agency_id=booking.agency_id,
                booking_type=booking.booking_type,
                amount=booking.amount,
                currency=booking.currency,
                booking_date=datetime.utcnow(),
                travel_date=datetime.fromisoformat(booking.travel_date),
                destination=booking.destination,
                source_country=booking.source_country,
                payment_method=booking.payment_method,
                device_fingerprint=booking.device_fingerprint,
                ip_address=booking.ip_address,
                session_id=booking.session_id,
                user_agent=booking.user_agent
            )
            
            result = await travel_intel.score_transaction_risk(transaction)
            results.append(result)
        
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return {
            "results": results,
            "total_transactions": len(request.transactions),
            "processing_time_ms": processing_time,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in batch scoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/user-risk-profile/{user_id}")
async def get_user_risk_profile(user_id: str, days_lookback: int = 90):
    """Get risk profile for a specific user"""
    
    try:
        # Get user network metrics
        user_network = await travel_intel.get_user_network_metrics(user_id)
        
        # Get recent activity
        recent_activity = await travel_intel.get_recent_user_activity(user_id, hours=days_lookback*24)
        
        # Calculate risk indicators
        total_amount = sum([txn['amount'] for txn in recent_activity])
        avg_amount = total_amount / len(recent_activity) if recent_activity else 0
        
        high_risk_bookings = sum([
            1 for txn in recent_activity 
            if txn['amount'] > avg_amount * 3
        ])
        
        return {
            "user_id": user_id,
            "network_metrics": user_network,
            "activity_summary": {
                "total_transactions": len(recent_activity),
                "total_amount": total_amount,
                "average_amount": avg_amount,
                "high_risk_bookings": high_risk_bookings
            },
            "risk_score": min(user_network.get('degree_centrality', 0) * 100, 100),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting user profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/agency-risk-profile/{agency_id}")
async def get_agency_risk_profile(agency_id: str, days_lookback: int = 90):
    """Get risk profile for a specific agency"""
    
    try:
        # Get agency network metrics
        agency_network = await travel_intel.get_agency_network_metrics(agency_id)
        
        # Get recent transactions
        query = f"""
        SELECT COUNT(*) as transaction_count, 
               AVG(amount) as avg_amount,
               SUM(amount) as total_amount,
               COUNT(DISTINCT user_id) as unique_users
        FROM transactions 
        WHERE agency_id = '{agency_id}' 
        AND booking_date >= NOW() - INTERVAL '{days_lookback} days'
        """
        
        # This would be executed against your database
        # For demo purposes, using placeholder values
        
        return {
            "agency_id": agency_id,
            "network_metrics": agency_network,
            "activity_summary": {
                "transaction_count": 150,  # Placeholder
                "unique_users": 75,       # Placeholder
                "total_amount": 45000.0,  # Placeholder
                "average_amount": 300.0   # Placeholder
            },
            "risk_indicators": {
                "isolation_risk": 1.0 - agency_network.get('degree_centrality', 0),
                "volume_risk": 0.2,  # Based on transaction patterns
                "user_diversity": 0.5  # Based on user variety
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting agency profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================
# MONITORING AND ANALYTICS ENDPOINTS
# ============================================================

@app.get("/api/v1/analytics/risk-distribution")
async def get_risk_distribution(hours: int = 24):
    """Get risk distribution over time"""
    
    # This would query your analytics database
    # Returning mock data for demonstration
    
    return {
        "time_period": f"last_{hours}_hours",
        "risk_distribution": {
            "LOW": 1250,
            "MEDIUM": 180,
            "HIGH": 25
        },
        "total_transactions": 1455,
        "fraud_rate": 0.017,  # 1.7%
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/api/v1/analytics/top-risk-factors")
async def get_top_risk_factors(hours: int = 24):
    """Get most common risk factors"""
    
    return {
        "time_period": f"last_{hours}_hours",
        "risk_factors": [
            {"factor": "New device with high amount", "count": 45, "percentage": 18.2},
            {"factor": "High-risk destination", "count": 32, "percentage": 12.9},
            {"factor": "Unusual booking time", "count": 28, "percentage": 11.3},
            {"factor": "Last-minute booking", "count": 24, "percentage": 9.7},
            {"factor": "High booking frequency", "count": 19, "percentage": 7.7}
        ],
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint"""
    
    try:
        # Check system components
        redis_status = travel_intel.redis_client.ping()
        
        return {
            "status": "healthy",
            "components": {
                "api": "up",
                "redis": "up" if redis_status else "down",
                "model": "loaded"
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

# ============================================================
# WEBHOOK ENDPOINTS FOR REAL-TIME PROCESSING
# ============================================================

@app.post("/api/v1/webhooks/booking-created")
async def handle_booking_webhook(request: BookingRequest, background_tasks: BackgroundTasks):
    """Handle real-time booking events"""
    
    try:
        # Add to background processing queue
        background_tasks.add_task(process_booking_async, request)
        
        return {
            "status": "accepted",
            "transaction_id": request.transaction_id,
            "message": "Booking queued for processing",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error handling booking webhook: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def process_booking_async(booking: BookingRequest):
    """Asynchronously process booking for risk scoring"""
    
    try:
        # Convert and score
        transaction = TravelTransaction(
            transaction_id=booking.transaction_id,
            user_id=booking.user_id,
            agency_id=booking.agency_id,
            booking_type=booking.booking_type,
            amount=booking.amount,
            currency=booking.currency,
            booking_date=datetime.utcnow(),
            travel_date=datetime.fromisoformat(booking.travel_date),
            destination=booking.destination,
            source_country=booking.source_country,
            payment_method=booking.payment_method,
            device_fingerprint=booking.device_fingerprint,
            ip_address=booking.ip_address,
            session_id=booking.session_id,
            user_agent=booking.user_agent
        )
        
        result = await travel_intel.score_transaction_risk(transaction)
        
        # Take action based on risk level
        if result['risk_bucket'] == 'HIGH':
            await send_high_risk_alert(transaction, result)
        elif result['risk_bucket'] == 'MEDIUM':
            await queue_for_review(transaction, result)
        
        logger.info(f"Processed booking {booking.transaction_id}: {result['risk_bucket']}")
        
    except Exception as e:
        logger.error(f"Error processing booking async: {e}")

async def send_high_risk_alert(transaction: TravelTransaction, result: Dict):
    """Send alert for high-risk transactions"""
    
    alert_data = {
        "alert_type": "HIGH_RISK_TRANSACTION",
        "transaction_id": transaction.transaction_id,
        "user_id": transaction.user_id,
        "agency_id": transaction.agency_id,
        "amount": transaction.amount,
        "fraud_probability": result['fraud_probability'],
        "risk_factors": result['risk_factors'],
        "timestamp": datetime.utcnow().isoformat()
    }
    
    # Send to monitoring system, Slack, email, etc.
    logger.warning(f"HIGH RISK ALERT: {json.dumps(alert_data)}")

async def queue_for_review(transaction: TravelTransaction, result: Dict):
    """Queue medium-risk transactions for manual review"""
    
    review_data = {
        "transaction_id": transaction.transaction_id,
        "risk_data": result,
        "status": "pending_review",
        "created_at": datetime.utcnow().isoformat()
    }
    
    # Add to review queue
    logger.info(f"Queued for review: {transaction.transaction_id}")

# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":
    uvicorn.run(
        "travel_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )