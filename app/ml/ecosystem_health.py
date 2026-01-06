"""
Ecosystem Health ML Model
Calculates overall health score and generates insights
"""
from typing import Dict, List
from app.services.data_fetcher import data_fetcher
from app.services.data_processor import data_processor
from app.utils.logger import logger
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

class EcosystemHealthModel:
    """ML model for ecosystem health analysis"""
    
    def __init__(self):
        self.thresholds = {
            "healthy": 80,
            "moderate": 60,
            "attention": 40
        }
    
    async def get_health_score(self) -> Dict:
        """
        Calculate comprehensive ecosystem health score
        
        Returns:
            Dict with health score, status, components, trends, and alerts
        """
        logger.info("🏥 Analyzing ecosystem health...")
        
        try:
            # Fetch all data
            all_data = await data_fetcher.fetch_all_data()
            
            # Calculate health score and component scores
            health_score, components = data_processor.calculate_ecosystem_health(all_data)
            
            # Determine status
            status = self._determine_status(health_score)
            
            # Calculate trends
            trends = await self._calculate_trends(all_data)
            
            # Generate alerts
            alerts = self._generate_alerts(all_data, components, trends)
            
            result = {
                "health_score": round(health_score, 1),
                "status": status,
                "components": {
                    k: round(v, 1) for k, v in components.items()
                },
                "trends": trends,
                "alerts": alerts,
                "last_updated": datetime.utcnow().isoformat() + "Z"
            }
            
            logger.success(f"✅ Health analysis complete: {status} ({health_score:.1f})")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error in health analysis: {e}")
            raise
    
    def _determine_status(self, score: float) -> str:
        """Determine health status based on score"""
        if score >= self.thresholds["healthy"]:
            return "Healthy"
        elif score >= self.thresholds["moderate"]:
            return "Moderate"
        elif score >= self.thresholds["attention"]:
            return "Needs Attention"
        else:
            return "Critical"
    
    async def _calculate_trends(self, all_data: Dict) -> Dict:
        """Calculate 7d and 30d trends"""
        try:
            # Get historical data
            network_df = data_processor.process_network_activity(
                all_data.get("network_activity", {})
            )
            
            if network_df.empty or len(network_df) < 30:
                return {
                    "7d_change": 0.0,
                    "30d_change": 0.0,
                    "direction": "stable"
                }
            
            # Calculate changes
            recent_7d = network_df.tail(7)["daily_transactions"].mean()
            previous_7d = network_df.tail(14).head(7)["daily_transactions"].mean()
            
            recent_30d = network_df.tail(30)["daily_transactions"].mean()
            previous_30d = network_df.tail(60).head(30)["daily_transactions"].mean()
            
            change_7d = ((recent_7d / previous_7d) - 1) * 100 if previous_7d > 0 else 0
            change_30d = ((recent_30d / previous_30d) - 1) * 100 if previous_30d > 0 else 0
            
            # Determine direction
            if change_7d > 5:
                direction = "growing"
            elif change_7d < -5:
                direction = "declining"
            else:
                direction = "stable"
            
            return {
                "7d_change": round(change_7d, 1),
                "30d_change": round(change_30d, 1),
                "direction": direction
            }
            
        except Exception as e:
            logger.error(f"Error calculating trends: {e}")
            return {
                "7d_change": 0.0,
                "30d_change": 0.0,
                "direction": "unknown"
            }
    
    def _generate_alerts(
        self,
        all_data: Dict,
        components: Dict,
        trends: Dict
    ) -> List[Dict]:
        """Generate dynamic alerts based on metrics"""
        alerts = []
        
        # Growth alerts
        if trends.get("7d_change", 0) > 10:
            alerts.append({
                "type": "positive",
                "severity": "info",
                "message": f"Network activity up {trends['7d_change']:.1f}% this week",
                "metric": "network_growth"
            })
        elif trends.get("7d_change", 0) < -10:
            alerts.append({
                "type": "warning",
                "severity": "medium",
                "message": f"Network activity down {abs(trends['7d_change']):.1f}% this week",
                "metric": "network_decline"
            })
        
        # Component-specific alerts
        for component, score in components.items():
            if score > 90:
                alerts.append({
                    "type": "positive",
                    "severity": "info",
                    "message": f"Excellent {component.replace('_', ' ')} performance",
                    "metric": component
                })
            elif score < 40:
                alerts.append({
                    "type": "warning",
                    "severity": "high",
                    "message": f"Low {component.replace('_', ' ')} score - attention needed",
                    "metric": component
                })
        
        # Whale activity alert
        whale_data = all_data.get("whales", {})
        if whale_data.get("success"):
            try:
                df = pd.DataFrame(whale_data.get("data", []))
                if not df.empty and "buy_volume" in df.columns:
                    buy_ratio = df["buy_volume"].sum() / (df["buy_volume"].sum() + df["sell_volume"].sum())
                    if buy_ratio > 0.65:
                        alerts.append({
                            "type": "positive",
                            "severity": "info",
                            "message": "Whales showing strong accumulation behavior",
                            "metric": "whale_sentiment"
                        })
                    elif buy_ratio < 0.35:
                        alerts.append({
                            "type": "warning",
                            "severity": "medium",
                            "message": "Whales showing distribution behavior",
                            "metric": "whale_sentiment"
                        })
            except Exception as e:
                logger.error(f"Error analyzing whale alerts: {e}")
        
        # Limit to top 5 most important alerts
        return sorted(
            alerts,
            key=lambda x: {"high": 3, "medium": 2, "info": 1}.get(x["severity"], 0),
            reverse=True
        )[:5]

# Global instance
ecosystem_health_model = EcosystemHealthModel()