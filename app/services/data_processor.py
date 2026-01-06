"""
Data processing and transformation service for ML models
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from app.utils.logger import logger

class DataProcessor:
    """Process and transform blockchain data for ML models"""
    
    def process_network_activity(self, raw_data: Dict) -> pd.DataFrame:
        """
        Convert network activity to DataFrame with proper types
        
        Args:
            raw_data: Raw API response
            
        Returns:
            Processed DataFrame
        """
        if not raw_data.get("success") or not raw_data.get("data"):
            logger.warning("No network activity data available")
            return pd.DataFrame()
        
        try:
            df = pd.DataFrame(raw_data["data"])
            if "day" in df.columns:
                df["day"] = pd.to_datetime(df["day"])
                df = df.sort_values("day")
            return df
        except Exception as e:
            logger.error(f"Error processing network activity: {e}")
            return pd.DataFrame()
    
    def process_volume_data(self, raw_data: Dict) -> pd.DataFrame:
        """Process volume/liquidity data"""
        if not raw_data.get("success") or not raw_data.get("data"):
            logger.warning("No volume data available")
            return pd.DataFrame()
        
        try:
            df = pd.DataFrame(raw_data["data"])
            if "Trade Day" in df.columns:
                df["Trade Day"] = pd.to_datetime(df["Trade Day"])
                df = df.sort_values("Trade Day")
            return df
        except Exception as e:
            logger.error(f"Error processing volume data: {e}")
            return pd.DataFrame()
    
    def process_games_data(self, raw_data: Dict) -> pd.DataFrame:
        """Process gaming metrics"""
        if not raw_data.get("success") or not raw_data.get("data"):
            return pd.DataFrame()
        
        try:
            df = pd.DataFrame(raw_data["data"])
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
            return df
        except Exception as e:
            logger.error(f"Error processing games data: {e}")
            return pd.DataFrame()
    
    def process_whale_data(self, raw_data: Dict) -> pd.DataFrame:
        """Process whale trader data"""
        if not raw_data.get("success") or not raw_data.get("data"):
            return pd.DataFrame()
        
        try:
            return pd.DataFrame(raw_data["data"])
        except Exception as e:
            logger.error(f"Error processing whale data: {e}")
            return pd.DataFrame()
    
    def calculate_ecosystem_health(self, all_data: Dict) -> Tuple[float, Dict]:
        """
        Calculate overall ecosystem health score (0-100)
        
        Args:
            all_data: Dict containing all 13 endpoint responses
            
        Returns:
            Tuple of (health_score, component_scores)
        """
        logger.info("🧮 Calculating ecosystem health score...")
        
        component_scores = {}
        
        # 1. Network Activity Score (25% weight)
        network_score = self._calculate_network_score(all_data.get("network_activity", {}))
        component_scores["network_activity"] = network_score
        
        # 2. DEX Volume Score (25% weight)
        volume_score = self._calculate_volume_score(all_data.get("volume_liquidity", {}))
        component_scores["dex_volume"] = volume_score
        
        # 3. Gaming Engagement Score (20% weight)
        gaming_score = self._calculate_gaming_score(
            all_data.get("games_daily", {}),
            all_data.get("games_overall", {})
        )
        component_scores["gaming_engagement"] = gaming_score
        
        # 4. Whale Activity Score (15% weight)
        whale_score = self._calculate_whale_score(all_data.get("whales", {}))
        component_scores["whale_activity"] = whale_score
        
        # 5. NFT Market Score (15% weight)
        nft_score = self._calculate_nft_score(all_data.get("nft_collections", {}))
        component_scores["nft_market"] = nft_score
        
        # Calculate weighted average
        weights = {
            "network_activity": 0.25,
            "dex_volume": 0.25,
            "gaming_engagement": 0.20,
            "whale_activity": 0.15,
            "nft_market": 0.15
        }
        
        health_score = sum(
            component_scores.get(key, 50) * weight
            for key, weight in weights.items()
        )
        
        logger.success(f"✅ Ecosystem health: {health_score:.1f}/100")
        return health_score, component_scores
    
    def _calculate_network_score(self, network_data: Dict) -> float:
        """Calculate network activity score"""
        df = self.process_network_activity(network_data)
        if df.empty or len(df) < 14:
            return 50.0  # Neutral score
        
        try:
            # Compare recent 7 days to previous 7 days
            recent = df.tail(7)
            previous = df.tail(14).head(7)
            
            # Metrics: daily_transactions, active_wallets
            txn_growth = recent["daily_transactions"].mean() / previous["daily_transactions"].mean()
            wallet_growth = recent["active_wallets"].mean() / previous["active_wallets"].mean()
            
            # Convert to 0-100 scale
            score = ((txn_growth - 1) * 50 + 50 + (wallet_growth - 1) * 50 + 50) / 2
            return np.clip(score, 0, 100)
        except Exception as e:
            logger.error(f"Error in network score: {e}")
            return 50.0
    
    def _calculate_volume_score(self, volume_data: Dict) -> float:
        """Calculate DEX volume score"""
        df = self.process_volume_data(volume_data)
        if df.empty or len(df) < 14:
            return 50.0
        
        try:
            recent_vol = df.tail(7)["WRON Volume (USD)"].sum()
            previous_vol = df.tail(14).head(7)["WRON Volume (USD)"].sum()
            
            if previous_vol == 0:
                return 50.0
            
            growth = recent_vol / previous_vol
            score = (growth - 1) * 50 + 50
            return np.clip(score, 0, 100)
        except Exception as e:
            logger.error(f"Error in volume score: {e}")
            return 50.0
    
    def _calculate_gaming_score(self, daily_data: Dict, overall_data: Dict) -> float:
        """Calculate gaming engagement score"""
        df = self.process_games_data(daily_data)
        if df.empty:
            return 50.0
        
        try:
            # Look at DAU trend
            if len(df) >= 7:
                recent_dau = df.tail(7)["daily_active_users"].mean()
                prev_dau = df.tail(14).head(7)["daily_active_users"].mean()
                
                if prev_dau == 0:
                    return 50.0
                
                growth = recent_dau / prev_dau
                score = (growth - 1) * 50 + 50
                return np.clip(score, 0, 100)
        except Exception as e:
            logger.error(f"Error in gaming score: {e}")
        
        return 50.0
    
    def _calculate_whale_score(self, whale_data: Dict) -> float:
        """Calculate whale activity score"""
        df = self.process_whale_data(whale_data)
        if df.empty:
            return 50.0
        
        try:
            # Analyze whale buy/sell ratio
            total_buys = df["buy_volume"].sum()
            total_sells = df["sell_volume"].sum()
            
            if total_sells == 0:
                return 75.0  # Bullish signal
            
            ratio = total_buys / total_sells
            # ratio > 1 = bullish, < 1 = bearish
            score = 50 + (ratio - 1) * 25
            return np.clip(score, 0, 100)
        except Exception as e:
            logger.error(f"Error in whale score: {e}")
        
        return 50.0
    
    def _calculate_nft_score(self, nft_data: Dict) -> float:
        """Calculate NFT market score"""
        if not nft_data.get("success") or not nft_data.get("data"):
            return 50.0
        
        try:
            df = pd.DataFrame(nft_data["data"])
            if df.empty:
                return 50.0
            
            # Look at volume trends
            if "volume" in df.columns:
                total_volume = df["volume"].sum()
                if total_volume > 1000000:  # Active market
                    return 75.0
                elif total_volume > 100000:
                    return 60.0
            
            return 50.0
        except Exception as e:
            logger.error(f"Error in NFT score: {e}")
        
        return 50.0
    
    def extract_features_for_ml(self, all_data: Dict) -> pd.DataFrame:
        """
        Extract features for ML models
        
        Returns:
            DataFrame with engineered features
        """
        features = {}
        
        # Network features
        network_df = self.process_network_activity(all_data.get("network_activity", {}))
        if not network_df.empty:
            features["avg_daily_txns"] = network_df["daily_transactions"].mean()
            features["avg_active_wallets"] = network_df["active_wallets"].mean()
            features["avg_gas_price"] = network_df.get("avg_gas_price_in_gwei", pd.Series([20])).mean()
            features["txn_7d_trend"] = self._calculate_trend(network_df["daily_transactions"].tail(7))
        
        # Volume features
        volume_df = self.process_volume_data(all_data.get("volume_liquidity", {}))
        if not volume_df.empty:
            features["total_volume_7d"] = volume_df.tail(7)["WRON Volume (USD)"].sum()
            features["avg_trades_per_day"] = volume_df["Number of Trades"].mean()
            features["volume_volatility"] = volume_df["WRON Volume (USD)"].std()
        
        # Gaming features
        games_df = self.process_games_data(all_data.get("games_daily", {}))
        if not games_df.empty:
            features["avg_dau"] = games_df["daily_active_users"].mean()
            features["dau_trend"] = self._calculate_trend(games_df["daily_active_users"].tail(7))
        
        return pd.DataFrame([features])
    
    def _calculate_trend(self, series: pd.Series) -> float:
        """Calculate trend direction (-1 to 1)"""
        if len(series) < 2:
            return 0.0
        
        x = np.arange(len(series))
        y = series.values
        
        # Simple linear regression
        slope = np.polyfit(x, y, 1)[0]
        mean_val = y.mean()
        
        if mean_val == 0:
            return 0.0
        
        # Normalize slope
        normalized_trend = np.clip(slope / mean_val * 10, -1, 1)
        return float(normalized_trend)

# Global instance
data_processor = DataProcessor()