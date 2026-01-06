"""
Data fetcher service for the 13 Ronin API endpoints
"""
import httpx
from typing import Dict, List, Optional
from app.config import settings
from app.utils.logger import logger
import asyncio
from datetime import datetime

class RoninDataFetcher:
    """Fetches data from the 13 open-source Ronin API endpoints"""
    
    # Map of endpoint names to their paths
    ENDPOINTS = {
        "network_activity": "/api/dune/network-activity",
        "volume_liquidity": "/api/dune/volume-liquidity",
        "games_daily": "/api/dune/games-daily",
        "weekly_segmentation": "/api/dune/weekly-segmentation",
        "hourly": "/api/dune/hourly",
        "whales": "/api/dune/whales",
        "ronin_daily": "/api/dune/ronin-daily",
        "retention": "/api/dune/retention",
        "trade_pairs": "/api/dune/trade-pairs",
        "games_overall": "/api/dune/games-overall",
        "holders": "/api/dune/holders",
        "segmented_holders": "/api/dune/segmented-holders",
        "nft_collections": "/api/dune/nft-collections"
    }
    
    def __init__(self):
        self.base_url = settings.RONIN_API_BASE_URL
        self.timeout = settings.HTTP_TIMEOUT_SECONDS
    
    async def fetch_endpoint(self, endpoint: str, endpoint_name: str = None) -> Dict:
        """
        Fetch data from a specific endpoint
        
        Args:
            endpoint: The API endpoint path
            endpoint_name: Human-readable name for logging
            
        Returns:
            Dict with response data or error information
        """
        url = f"{self.base_url}{endpoint}"
        name = endpoint_name or endpoint
        
        logger.info(f"📡 Fetching data from: {name}")
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.get(url)
                response.raise_for_status()
                data = response.json()
                
                logger.success(f"✅ Successfully fetched {name}")
                return {
                    "success": True,
                    "data": data.get("data", data),
                    "endpoint": endpoint,
                    "fetched_at": datetime.utcnow().isoformat()
                }
                
            except httpx.HTTPStatusError as e:
                logger.error(f"❌ HTTP error fetching {name}: {e.response.status_code}")
                return {
                    "success": False,
                    "data": None,
                    "error": f"HTTP {e.response.status_code}",
                    "endpoint": endpoint
                }
            except httpx.TimeoutException:
                logger.error(f"⏱️  Timeout fetching {name}")
                return {
                    "success": False,
                    "data": None,
                    "error": "Request timeout",
                    "endpoint": endpoint
                }
            except Exception as e:
                logger.error(f"❌ Error fetching {name}: {str(e)}")
                return {
                    "success": False,
                    "data": None,
                    "error": str(e),
                    "endpoint": endpoint
                }
    
    async def fetch_all_data(self) -> Dict[str, Dict]:
        """
        Fetch data from all 13 endpoints concurrently
        
        Returns:
            Dict mapping endpoint names to their response data
        """
        logger.info("🌐 Fetching data from all 13 Ronin endpoints...")
        
        # Create tasks for concurrent fetching
        tasks = [
            self.fetch_endpoint(path, name)
            for name, path in self.ENDPOINTS.items()
        ]
        
        # Execute all requests concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Map results to endpoint names
        data_map = {}
        for (name, _), result in zip(self.ENDPOINTS.items(), results):
            if isinstance(result, Exception):
                logger.error(f"❌ Exception for {name}: {result}")
                data_map[name] = {
                    "success": False,
                    "data": None,
                    "error": str(result)
                }
            else:
                data_map[name] = result
        
        successful = sum(1 for d in data_map.values() if d.get("success"))
        logger.info(f"📊 Fetched {successful}/{len(self.ENDPOINTS)} endpoints successfully")
        
        return data_map
    
    # Individual endpoint methods for specific use cases
    
    async def get_network_activity(self) -> Dict:
        """Fetch network activity metrics"""
        return await self.fetch_endpoint(
            self.ENDPOINTS["network_activity"],
            "network_activity"
        )
    
    async def get_volume_liquidity(self) -> Dict:
        """Fetch DEX volume and liquidity data"""
        return await self.fetch_endpoint(
            self.ENDPOINTS["volume_liquidity"],
            "volume_liquidity"
        )
    
    async def get_games_daily(self) -> Dict:
        """Fetch daily gaming metrics"""
        return await self.fetch_endpoint(
            self.ENDPOINTS["games_daily"],
            "games_daily"
        )
    
    async def get_whales(self) -> Dict:
        """Fetch whale trader data"""
        return await self.fetch_endpoint(
            self.ENDPOINTS["whales"],
            "whales"
        )
    
    async def get_hourly_data(self) -> Dict:
        """Fetch hourly trading data"""
        return await self.fetch_endpoint(
            self.ENDPOINTS["hourly"],
            "hourly"
        )
    
    async def get_retention(self) -> Dict:
        """Fetch user retention cohort data"""
        return await self.fetch_endpoint(
            self.ENDPOINTS["retention"],
            "retention"
        )
    
    async def get_holders(self) -> Dict:
        """Fetch token holders data"""
        return await self.fetch_endpoint(
            self.ENDPOINTS["holders"],
            "holders"
        )
    
    async def get_segmented_holders(self) -> Dict:
        """Fetch segmented holders data"""
        return await self.fetch_endpoint(
            self.ENDPOINTS["segmented_holders"],
            "segmented_holders"
        )
    
    async def get_nft_collections(self) -> Dict:
        """Fetch NFT collections data"""
        return await self.fetch_endpoint(
            self.ENDPOINTS["nft_collections"],
            "nft_collections"
        )
    
    async def get_trade_pairs(self) -> Dict:
        """Fetch trading pairs analytics"""
        return await self.fetch_endpoint(
            self.ENDPOINTS["trade_pairs"],
            "trade_pairs"
        )
    
    async def get_games_overall(self) -> Dict:
        """Fetch overall gaming statistics"""
        return await self.fetch_endpoint(
            self.ENDPOINTS["games_overall"],
            "games_overall"
        )
    
    async def get_ronin_daily(self) -> Dict:
        """Fetch daily Ronin metrics"""
        return await self.fetch_endpoint(
            self.ENDPOINTS["ronin_daily"],
            "ronin_daily"
        )
    
    async def get_weekly_segmentation(self) -> Dict:
        """Fetch weekly trade segmentation"""
        return await self.fetch_endpoint(
            self.ENDPOINTS["weekly_segmentation"],
            "weekly_segmentation"
        )

# Global instance
data_fetcher = RoninDataFetcher()