"""
Pydantic models for request/response validation
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime

# Response Models

class HealthComponent(BaseModel):
    """Individual health component score"""
    network_activity: float = Field(..., ge=0, le=100)
    dex_volume: float = Field(..., ge=0, le=100)
    gaming_engagement: float = Field(..., ge=0, le=100)
    whale_activity: float = Field(..., ge=0, le=100)
    nft_market: float = Field(..., ge=0, le=100)

class Trend(BaseModel):
    """Trend information"""
    change_7d: float = Field(..., alias="7d_change")
    change_30d: float = Field(..., alias="30d_change")
    direction: str

    class Config:
        populate_by_name = True

class Alert(BaseModel):
    """Alert/notification model"""
    type: str  # positive, warning, critical
    severity: str  # info, medium, high
    message: str
    metric: str

class EcosystemHealthResponse(BaseModel):
    """Ecosystem health response"""
    health_score: float = Field(..., ge=0, le=100)
    status: str
    components: HealthComponent
    trends: Trend
    alerts: List[Alert]
    last_updated: str

class ConfidenceInterval(BaseModel):
    """Confidence interval for predictions"""
    lower: float
    upper: float

class VolumeForecast(BaseModel):
    """Single volume forecast"""
    date: str
    predicted_volume_usd: float
    confidence_interval: ConfidenceInterval

class VolumeForecastResponse(BaseModel):
    """Volume forecast response"""
    forecast: List[VolumeForecast]
    model_accuracy: float
    last_trained: str
    historical_avg: float

class WhalePatterns(BaseModel):
    """Whale trading patterns"""
    accumulation_phase: bool
    avg_trade_size_trend: str
    buy_sell_ratio: float

class WhalePredictions(BaseModel):
    """Whale behavior predictions"""
    likely_action_24h: str
    confidence: float

class WhaleBehaviorResponse(BaseModel):
    """Whale behavior response"""
    total_whales: int
    active_whales_24h: int
    whale_sentiment: str
    patterns: WhalePatterns
    predictions: WhalePredictions

class RetentionForecast(BaseModel):
    """Retention forecast by week"""
    week_1: float
    week_2: float
    week_4: float

class ChurnPredictionResponse(BaseModel):
    """Churn prediction response"""
    game: str
    churn_risk: str
    churn_probability: float
    at_risk_players: int
    retention_forecast: RetentionForecast
    recommendations: List[str]

class Anomaly(BaseModel):
    """Detected anomaly"""
    metric: str
    timestamp: str
    value: float
    expected_range: List[float]
    severity: str
    description: str

class AnomalyDetectionResponse(BaseModel):
    """Anomaly detection response"""
    anomalies_detected: int
    anomalies: List[Anomaly]
    overall_status: str

class HolderSegment(BaseModel):
    """Holder segment"""
    segment: str
    count: int
    avg_hold_duration_days: int
    avg_balance: float
    churn_risk: str
    percentage: float

class HolderSegmentationResponse(BaseModel):
    """Holder segmentation response"""
    segments: List[HolderSegment]
    total_holders: int
    analysis_date: str

class TrendingCollection(BaseModel):
    """Trending NFT collection"""
    collection: str
    momentum_score: int
    predicted_floor_price_7d: float
    volume_trend: str
    volume_change_24h: float

class MarketPredictions(BaseModel):
    """NFT market predictions"""
    total_volume_7d: float
    confidence: float
    expected_trend: str

class NFTTrendsResponse(BaseModel):
    """NFT trends response"""
    market_sentiment: str
    trending_collections: List[TrendingCollection]
    market_predictions: MarketPredictions

class CongestionForecast(BaseModel):
    """Network congestion forecast"""
    hour: str
    stress_level: str
    predicted_gas_gwei: float

class NetworkStressResponse(BaseModel):
    """Network stress test response"""
    current_stress_level: str
    current_gas_price: float
    predicted_gas_price_1h: float
    congestion_forecast: List[CongestionForecast]

# Generic Response Wrapper

class APIResponse(BaseModel):
    """Generic API response wrapper"""
    success: bool
    data: Optional[Dict] = None
    error: Optional[Dict] = None
    metadata: Dict

class Metadata(BaseModel):
    """Response metadata"""
    timestamp: str
    version: str = "1.0.0"