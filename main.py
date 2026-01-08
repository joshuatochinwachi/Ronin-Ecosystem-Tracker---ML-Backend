"""
Ronin Gaming ML Analytics API
Version: 1.0.0
Features:
- 13 Ronin Analytics endpoints 
- Multi-model ML ensemble with auto-selection
- Game health prediction (declining vs growing games)
- Network activity forecasting
- Automated retraining on data refresh
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import os
import time
import joblib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
from contextlib import asynccontextmanager
import json
import requests

from dotenv import load_dotenv
load_dotenv()

# ML imports
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
# from imblearn.over_sampling import SMOTE

# Try to import XGBoost and LightGBM
try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    
try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# Background task imports
import asyncio
from threading import Thread

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== HELPER FUNCTIONS ====================

def clean_dataframe_for_json(df: pd.DataFrame) -> pd.DataFrame:
    """Clean dataframe for JSON serialization"""
    if df.empty:
        return df
    
    df = df.replace([np.inf, -np.inf], None)
    df = df.where(pd.notna(df), None)
    
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].astype(str)
        elif df[col].dtype == 'object':
            df[col] = df[col].apply(
                lambda x: str(x) if pd.notna(x) and not isinstance(x, (str, int, float, bool, type(None))) else x
            )
    
    return df

def safe_float(value: Any) -> Optional[float]:
    """Convert float to JSON-safe value"""
    if isinstance(value, (int, float)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    return value

# ==================== CONFIGURATION ====================

class Config:
    def __init__(self):
        self.base_url = "https://web-production-4fae.up.railway.app"
        
        # All 13 Ronin endpoints
        self.endpoints = {
            'games_daily_activity': '/api/raw/dune/games_daily_activity',
            'games_overall_activity': '/api/raw/dune/games_overall_activity',
            'ronin_daily_activity': '/api/raw/dune/ronin_daily_activity',
            'user_activation_retention': '/api/raw/dune/user_activation_retention',
            'ron_current_holders': '/api/raw/dune/ron_current_holders',
            'ron_segmented_holders': '/api/raw/dune/ron_segmented_holders',
            'wron_active_trade_pairs': '/api/raw/dune/wron_active_trade_pairs',
            'wron_whale_tracking': '/api/raw/dune/wron_whale_tracking',
            'wron_volume_liquidity': '/api/raw/dune/wron_volume_liquidity',
            'wron_trading_hourly': '/api/raw/dune/wron_trading_hourly',
            'wron_weekly_segmentation': '/api/raw/dune/wron_weekly_segmentation',
            'nft_collections': '/api/raw/dune/nft_collections',
            'coingecko_ron': '/api/raw/coingecko/ron'
        }
        
        # Smart polling configuration
        self.check_interval_seconds = int(os.getenv('CHECK_INTERVAL_SECONDS', 900))  # 15 minutes
        self.min_training_samples = int(os.getenv('MIN_TRAINING_SAMPLES', 15))
        
        # Force retrain if data hasn't changed in X hours (safety fallback)
        self.max_stale_hours = int(os.getenv('MAX_STALE_HOURS', 24))  # 24 hours

config = Config()

# ==================== CACHE MANAGER ====================

class CacheManager:
    def __init__(self):
        self.cache_dir = "raw_data_cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        self.metadata_file = os.path.join(self.cache_dir, "cache_metadata.json")
        self.metadata = self._load_metadata()
        self.last_check_time = None
        self.last_data_hash = {}  # Store hash of each endpoint's data
    
    def _load_metadata(self) -> Dict:
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)
                    # Restore last check time and hashes
                    self.last_check_time = data.get('last_check_time')
                    self.last_data_hash = data.get('data_hashes', {})
                    return data
            except:
                return {}
        return {}
    
    def _save_metadata(self):
        try:
            metadata_to_save = self.metadata.copy()
            metadata_to_save['last_check_time'] = self.last_check_time
            metadata_to_save['data_hashes'] = self.last_data_hash
            
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata_to_save, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
    
    def _get_cache_path(self, endpoint_name: str) -> str:
        return os.path.join(self.cache_dir, f"{endpoint_name}.parquet")
    
    def _calculate_data_hash(self, df: pd.DataFrame) -> str:
        """Calculate hash of dataframe to detect changes"""
        try:
            # Use shape and a sample of data to create hash
            # This is faster than hashing entire dataframe
            hash_input = f"{df.shape}_{df.head(10).to_json()}_{df.tail(10).to_json()}"
            import hashlib
            return hashlib.md5(hash_input.encode()).hexdigest()
        except:
            return ""
    
    def should_check_for_updates(self) -> bool:
        """Check if it's time to poll for data updates"""
        if self.last_check_time is None:
            return True
        
        last_check = datetime.fromisoformat(self.last_check_time)
        elapsed = (datetime.now() - last_check).total_seconds()
        
        return elapsed >= config.check_interval_seconds
    
    def check_for_data_changes(self) -> Dict[str, bool]:
        """
        Check each endpoint for data changes
        Returns dict with endpoint names and whether they changed
        """
        changes = {}
        
        for endpoint_name in config.endpoints.keys():
            try:
                # Fetch fresh data
                url = config.base_url + config.endpoints[endpoint_name]
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                
                # Handle nested structure
                if isinstance(data, dict):
                    if 'data' in data:
                        df = pd.DataFrame(data['data'])
                    elif 'rows' in data:
                        df = pd.DataFrame(data['rows'])
                    else:
                        df = pd.DataFrame([data])
                elif isinstance(data, list):
                    df = pd.DataFrame(data)
                else:
                    continue
                
                # Calculate hash
                new_hash = self._calculate_data_hash(df)
                old_hash = self.last_data_hash.get(endpoint_name, "")
                
                # Check if changed
                if new_hash != old_hash:
                    logger.info(f"✓ {endpoint_name}: DATA CHANGED")
                    changes[endpoint_name] = True
                    # Save new data and hash
                    self.save_to_cache(endpoint_name, df)
                    self.last_data_hash[endpoint_name] = new_hash
                else:
                    logger.info(f"✓ {endpoint_name}: No changes")
                    changes[endpoint_name] = False
                    
            except Exception as e:
                logger.error(f"Error checking {endpoint_name}: {e}")
                changes[endpoint_name] = False
        
        # Update last check time
        self.last_check_time = datetime.now().isoformat()
        self._save_metadata()
        
        return changes
    
    def load_from_cache(self, endpoint_name: str) -> Optional[pd.DataFrame]:
        """Load data from cache"""
        try:
            cache_path = self._get_cache_path(endpoint_name)
            if os.path.exists(cache_path):
                df = pd.read_parquet(cache_path)
                logger.info(f"✓ Loaded {endpoint_name} from cache ({len(df)} rows)")
                return df
        except Exception as e:
            logger.error(f"Error loading cache for {endpoint_name}: {e}")
        return None
    
    def save_to_cache(self, endpoint_name: str, df: pd.DataFrame):
        """Save data to cache"""
        try:
            cache_path = self._get_cache_path(endpoint_name)
            df.to_parquet(cache_path, index=False)
            
            self.metadata[endpoint_name] = {
                'cached_at': datetime.now().isoformat(),
                'row_count': len(df),
                'columns': list(df.columns)
            }
            self._save_metadata()
            
            logger.info(f"✓ Cached {endpoint_name} ({len(df)} rows)")
        except Exception as e:
            logger.error(f"Error saving cache for {endpoint_name}: {e}")
    
    def fetch_data(self, endpoint_name: str, force: bool = False) -> pd.DataFrame:
        """Fetch data from API endpoint or load from cache"""
        # If not forcing and cache exists, load from cache
        if not force:
            cached_df = self.load_from_cache(endpoint_name)
            if cached_df is not None:
                return cached_df
        
        # Fetch from API
        try:
            url = config.base_url + config.endpoints[endpoint_name]
            logger.info(f"Fetching {endpoint_name} from API...")
            
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Handle nested structure
            if isinstance(data, dict):
                if 'data' in data:
                    df = pd.DataFrame(data['data'])
                elif 'rows' in data:
                    df = pd.DataFrame(data['rows'])
                else:
                    df = pd.DataFrame([data])
            elif isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                raise ValueError(f"Unexpected data format: {type(data)}")
            
            # Save to cache and update hash
            self.save_to_cache(endpoint_name, df)
            self.last_data_hash[endpoint_name] = self._calculate_data_hash(df)
            self._save_metadata()
            
            logger.info(f"✓ Fetched {endpoint_name} ({len(df)} rows)")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching {endpoint_name}: {e}")
            # Try to return stale cache if available
            cached_df = self.load_from_cache(endpoint_name)
            if cached_df is not None:
                logger.warning(f"Using cached data for {endpoint_name}")
                return cached_df
            raise
    
    def refresh_all_data(self) -> Dict[str, Any]:
        """
        Check all 13 endpoints and refresh only if data changed
        Returns: dict with 'any_changes' and 'changes' details
        """
        logger.info("Checking all endpoints for data changes...")
        changes = self.check_for_data_changes()
        
        any_changes = any(changes.values())
        changed_endpoints = [name for name, changed in changes.items() if changed]
        
        result = {
            'any_changes': any_changes,
            'changes': changes,
            'changed_endpoints': changed_endpoints,
            'total_changed': len(changed_endpoints)
        }
        
        if any_changes:
            logger.info(f"🔄 {len(changed_endpoints)} endpoint(s) changed: {', '.join(changed_endpoints)}")
        else:
            logger.info("✓ No data changes detected across all endpoints")
        
        return result
    
    def cache_data(self, name: str, df: pd.DataFrame):
        """Cache arbitrary dataframe (for predictions, etc)"""
        self.save_to_cache(name, df)
    
    def get_cached_data(self, name: str) -> Optional[pd.DataFrame]:
        """Get cached dataframe"""
        return self.load_from_cache(name)

# ==================== FEATURE SERVICE ====================

class FeatureService:
    def __init__(self):
        self.feature_columns = [
            'avg_players_7d',
            'avg_transactions_7d',
            'avg_volume_7d',
            'player_trend_7d',
            'transaction_trend_7d',
            'player_volatility',
            'days_since_peak',
            'peak_to_current_ratio',
            'consistency_score',
            'growth_momentum'
        ]
    
    def create_game_features(self, game_data: pd.DataFrame) -> Optional[Dict]:
        """
        Create features for a single game from its daily activity history
        Predicts if game will decline in next 7 days
        """
        if len(game_data) < 14:  # Need at least 2 weeks of data
            return None
        
        game_data = game_data.sort_values('day')
        
        # Define cutoff - we'll predict activity AFTER this date
        cutoff_date = game_data['day'].max() - pd.Timedelta(days=7)
        
        # Training period: data BEFORE cutoff
        training_period = game_data[game_data['day'] <= cutoff_date]
        
        # Target period: data AFTER cutoff (next 7 days)
        target_start = cutoff_date
        target_end = cutoff_date + pd.Timedelta(days=7)
        target_period = game_data[
            (game_data['day'] > target_start) & 
            (game_data['day'] <= target_end)
        ]
        
        if len(training_period) < 7 or len(target_period) < 3:
            return None
        
        features = {}
        
        try:
            # Last 7 days features
            last_7 = training_period.tail(7)
            features['avg_players_7d'] = last_7['unique_players'].mean()
            features['avg_transactions_7d'] = last_7['transaction_count'].mean()
            features['avg_volume_7d'] = last_7['total_volume_ron_sent_to_game'].mean()
            
            # Trends (recent vs previous)
            if len(training_period) >= 14:
                recent_7 = training_period.tail(7)
                previous_7 = training_period.iloc[-14:-7]
                
                recent_avg_players = recent_7['unique_players'].mean()
                previous_avg_players = previous_7['unique_players'].mean()
                
                if previous_avg_players > 0:
                    features['player_trend_7d'] = (recent_avg_players - previous_avg_players) / previous_avg_players
                else:
                    features['player_trend_7d'] = 0
                
                recent_avg_txns = recent_7['transaction_count'].mean()
                previous_avg_txns = previous_7['transaction_count'].mean()
                
                if previous_avg_txns > 0:
                    features['transaction_trend_7d'] = (recent_avg_txns - previous_avg_txns) / previous_avg_txns
                else:
                    features['transaction_trend_7d'] = 0
            else:
                features['player_trend_7d'] = 0
                features['transaction_trend_7d'] = 0
            
            # Volatility
            features['player_volatility'] = training_period['unique_players'].std()
            
            # Peak metrics
            peak_players = training_period['unique_players'].max()
            current_players = training_period['unique_players'].iloc[-1]
            peak_date = training_period.loc[training_period['unique_players'].idxmax(), 'day']
            
            features['days_since_peak'] = (training_period['day'].max() - peak_date).days
            features['peak_to_current_ratio'] = current_players / peak_players if peak_players > 0 else 0
            
            # Consistency score (inverse of coefficient of variation)
            mean_players = training_period['unique_players'].mean()
            std_players = training_period['unique_players'].std()
            cv = std_players / mean_players if mean_players > 0 else 0
            features['consistency_score'] = 1 / (1 + cv)
            
            # Growth momentum (first week vs last week)
            first_week = training_period.head(7)['unique_players'].mean()
            last_week = training_period.tail(7)['unique_players'].mean()
            
            if first_week > 0:
                features['growth_momentum'] = (last_week - first_week) / first_week
            else:
                features['growth_momentum'] = 0
            
            # TARGET: Did the game decline in the next 7 days?
            # Declining = average players in target period < average in last 7 of training
            avg_future_players = target_period['unique_players'].mean()
            avg_recent_players = training_period.tail(7)['unique_players'].mean()
            
            # Game is "declining" if future players drop by more than 10%
            features['will_decline'] = 1 if avg_future_players < (avg_recent_players * 0.9) else 0
            
            return features
            
        except Exception as e:
            logger.error(f"Error creating features: {e}")
            return None
    
    def create_training_dataset(self, games_daily_df: pd.DataFrame) -> pd.DataFrame:
        """Create training dataset from games daily activity"""
        training_data = []
        
        # Convert day to datetime
        games_daily_df['day'] = pd.to_datetime(games_daily_df['day'])
        
        # Group by game
        for game, group in games_daily_df.groupby('game_project'):
            if len(group) >= 14:  # Need at least 2 weeks of data
                features = self.create_game_features(group)
                if features:
                    features['game_project'] = game
                    training_data.append(features)
        
        df = pd.DataFrame(training_data)
        
        if not df.empty:
            # Log class distribution
            class_dist = df['will_decline'].value_counts()
            total = len(df)
            declining = class_dist.get(1, 0)
            growing = class_dist.get(0, 0)
            
            logger.info(f"✓ Created training dataset with {total} samples")
            logger.info(f"  - Declining (1): {declining} ({declining/total*100:.1f}%)")
            logger.info(f"  - Growing (0): {growing} ({growing/total*100:.1f}%)")
            
            if len(class_dist) == 1:
                logger.warning("⚠️ Training data has only 1 class!")
        
        return df
    
    def create_prediction_features(self, games_daily_df: pd.DataFrame) -> pd.DataFrame:
        """Create features for current game predictions"""
        prediction_data = []
        
        games_daily_df['day'] = pd.to_datetime(games_daily_df['day'])
        
        for game, group in games_daily_df.groupby('game_project'):
            if len(group) >= 7:
                group = group.sort_values('day')
                latest_date = group['day'].max()
                
                features = {}
                
                # Last 7 days
                last_7 = group.tail(7)
                features['avg_players_7d'] = last_7['unique_players'].mean()
                features['avg_transactions_7d'] = last_7['transaction_count'].mean()
                features['avg_volume_7d'] = last_7['total_volume_ron_sent_to_game'].mean()
                
                # Trends
                if len(group) >= 14:
                    recent_7 = group.tail(7)
                    previous_7 = group.iloc[-14:-7]
                    
                    recent_avg = recent_7['unique_players'].mean()
                    prev_avg = previous_7['unique_players'].mean()
                    features['player_trend_7d'] = (recent_avg - prev_avg) / prev_avg if prev_avg > 0 else 0
                    
                    recent_txn = recent_7['transaction_count'].mean()
                    prev_txn = previous_7['transaction_count'].mean()
                    features['transaction_trend_7d'] = (recent_txn - prev_txn) / prev_txn if prev_txn > 0 else 0
                else:
                    features['player_trend_7d'] = 0
                    features['transaction_trend_7d'] = 0
                
                features['player_volatility'] = group['unique_players'].std()
                
                peak_players = group['unique_players'].max()
                current_players = group['unique_players'].iloc[-1]
                peak_date = group.loc[group['unique_players'].idxmax(), 'day']
                
                features['days_since_peak'] = (latest_date - peak_date).days
                features['peak_to_current_ratio'] = current_players / peak_players if peak_players > 0 else 0
                
                mean_players = group['unique_players'].mean()
                std_players = group['unique_players'].std()
                cv = std_players / mean_players if mean_players > 0 else 0
                features['consistency_score'] = 1 / (1 + cv)
                
                first_week = group.head(7)['unique_players'].mean()
                last_week = group.tail(7)['unique_players'].mean()
                features['growth_momentum'] = (last_week - first_week) / first_week if first_week > 0 else 0
                
                features['game_project'] = game
                features['latest_players'] = int(current_players)
                features['latest_transactions'] = int(group['transaction_count'].iloc[-1])
                
                prediction_data.append(features)
        
        df = pd.DataFrame(prediction_data)
        logger.info(f"Created prediction dataset with {len(df)} games")
        return df

# ==================== ML MODEL MANAGER ====================

class MLModelManager:
    def __init__(self):
        self.models_dir = "ml_models"
        os.makedirs(self.models_dir, exist_ok=True)
        
        self.feature_columns = [
            'avg_players_7d', 'avg_transactions_7d', 'avg_volume_7d',
            'player_trend_7d', 'transaction_trend_7d', 'player_volatility',
            'days_since_peak', 'peak_to_current_ratio', 'consistency_score', 'growth_momentum'
        ]
        
        self.champion = None
        self.top_3_ensemble = []
        self.all_models = []
        self.scaler = None
        
        self.model_configs = {
            'logistic_regression': {
                'model': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
                'priority': 3
            },
            'random_forest': {
                'model': RandomForestClassifier(
                    n_estimators=100, max_depth=8, min_samples_split=10,
                    random_state=42, n_jobs=-1, class_weight='balanced'
                ),
                'priority': 2
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(
                    n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42
                ),
                'priority': 2
            }
        }
        
        if XGBOOST_AVAILABLE:
            self.model_configs['xgboost'] = {
                'model': XGBClassifier(
                    n_estimators=100, max_depth=6, learning_rate=0.1,
                    random_state=42, eval_metric='logloss'
                ),
                'priority': 1
            }
        
        if LIGHTGBM_AVAILABLE:
            self.model_configs['lightgbm'] = {
                'model': LGBMClassifier(
                    n_estimators=100, max_depth=6, learning_rate=0.1,
                    random_state=42, verbose=-1, class_weight='balanced'
                ),
                'priority': 1
            }
        
        self._load_models()
    
    def train_and_evaluate_all(self, training_df: pd.DataFrame) -> List[Dict]:
        """Train all models and select champion"""
        logger.info("=" * 60)
        logger.info("TRAINING GAME HEALTH PREDICTION MODELS")
        logger.info("=" * 60)
        
        X = training_df[self.feature_columns].fillna(0)
        y = training_df['will_decline']
        
        # Check if we have both classes
        if len(y.unique()) < 2:
            logger.error("âŒ Training data has only one class! Cannot train models.")
            return []
        
        # Temporal split - test on most recent 25% of games
        total_samples = len(X)
        train_size = int(total_samples * 0.75)

        X_train = X.iloc[:train_size]
        X_test = X.iloc[train_size:]
        y_train = y.iloc[:train_size]
        y_test = y.iloc[train_size:]

        # Check if we have both classes in train and test
        if len(y_train.unique()) < 2 or len(y_test.unique()) < 2:
            logger.warning("⚠️ Single class in split, falling back to stratified split")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=42, stratify=y
            )
        
        # Check class distribution
        decline_rate = sum(y_train) / len(y_train)
        logger.info(f"📊 Training set:")
        logger.info(f"   Declining (1): {sum(y_train)} ({decline_rate*100:.1f}%)")
        logger.info(f"   Growing (0): {len(y_train)-sum(y_train)} ({(1-decline_rate)*100:.1f}%)")
        
        # Use class weights instead of SMOTE for time series
        logger.info(f"ℹ️ Using class weights to handle imbalance (decline rate: {decline_rate*100:.1f}%)")
        X_train_balanced = X_train
        y_train_balanced = y_train
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train_balanced)
        X_test_scaled = self.scaler.transform(X_test)
        
        results = []
        
        for name, config in self.model_configs.items():
            try:
                logger.info(f"Training {name}...")
                start_time = time.time()
                
                model = config['model']
                
                # For XGBoost, set scale_pos_weight
                if name == 'xgboost' and XGBOOST_AVAILABLE:
                    neg_count = len(y_train_balanced) - sum(y_train_balanced)
                    pos_count = sum(y_train_balanced)
                    if pos_count > 0:
                        scale_weight = neg_count / pos_count
                        model.set_params(scale_pos_weight=scale_weight)
                
                model.fit(X_train_scaled, y_train_balanced)
                training_time = time.time() - start_time
                
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                y_pred = model.predict(X_test_scaled)
                
                metrics = {
                    'name': name,
                    'model': model,
                    'roc_auc': roc_auc_score(y_test, y_pred_proba),
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, zero_division=0),
                    'recall': recall_score(y_test, y_pred, zero_division=0),
                    'training_time': training_time,
                    'timestamp': datetime.now().isoformat()
                }
                
                results.append(metrics)
                logger.info(f"  ✓ {name}: ROC-AUC={metrics['roc_auc']:.4f}, Accuracy={metrics['accuracy']:.4f}")
                
            except Exception as e:
                logger.error(f"  ❌ Failed to train {name}: {e}")
        
        # Calculate composite scores
        for result in results:
            composite_score = (
                0.35 * result['roc_auc'] +
                0.35 * result['accuracy'] +
                0.15 * result.get('precision', 0) +
                0.15 * result.get('recall', 0)
            )
            result['composite_score'] = composite_score

        # Sort by composite score
        results.sort(key=lambda x: x['composite_score'], reverse=True)
        
        
        if results:
            self.champion = results[0]
            self.top_3_ensemble = results[:min(3, len(results))]
            self.all_models = results
            
            logger.info("=" * 60)
            logger.info(f"CHAMPION MODEL: {self.champion['name'].upper()}")
            logger.info(f"ROC-AUC: {self.champion['roc_auc']:.4f}")
            logger.info(f"Top 3: {', '.join([m['name'] for m in self.top_3_ensemble])}")
            logger.info("=" * 60)
            
            self._save_models()
        
        return results
    
    def predict_champion(self, prediction_df: pd.DataFrame) -> np.ndarray:
        """Predict using champion model"""
        if not self.champion or not self.scaler:
            raise ValueError("Models not trained yet")
        
        X = prediction_df[self.feature_columns].fillna(0)
        X_scaled = self.scaler.transform(X)
        decline_proba = self.champion['model'].predict_proba(X_scaled)[:, 1]
        return decline_proba
    
    def predict_ensemble(self, prediction_df: pd.DataFrame) -> np.ndarray:
        """Predict using ensemble of top 3 models"""
        if not self.top_3_ensemble or not self.scaler:
            raise ValueError("Models not trained yet")
        
        X = prediction_df[self.feature_columns].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        predictions = []
        weights = []
        
        for model_info in self.top_3_ensemble:
            pred = model_info['model'].predict_proba(X_scaled)[:, 1]
            predictions.append(pred)
            weights.append(model_info['composite_score'])  
        
        ensemble_pred = np.average(predictions, axis=0, weights=weights)
        return ensemble_pred
    
    def _save_models(self):
        """Save models to disk"""
        try:
            joblib.dump(self.scaler, os.path.join(self.models_dir, 'scaler.joblib'))
            
            for model_info in self.all_models:
                model_path = os.path.join(self.models_dir, f"{model_info['name']}.joblib")
                joblib.dump(model_info['model'], model_path)
            
            metadata = {
            'champion': self.champion['name'] if self.champion else None,
            'champion_roc_auc': self.champion['roc_auc'] if self.champion else 0,
            'champion_accuracy': self.champion.get('accuracy', 0) if self.champion else 0,
            'champion_composite_score': self.champion.get('composite_score', 0) if self.champion else 0,
            'top_3': [m['name'] for m in self.top_3_ensemble],
            'last_trained': datetime.now().isoformat()
        }
            
            with open(os.path.join(self.models_dir, 'metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info("✓ Models saved successfully")
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def _load_models(self):
        """Load models from disk"""
        try:
            metadata_path = os.path.join(self.models_dir, 'metadata.json')
            if not os.path.exists(metadata_path):
                return
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            scaler_path = os.path.join(self.models_dir, 'scaler.joblib')
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
            
            champion_name = metadata.get('champion')
            if champion_name:
                model_path = os.path.join(self.models_dir, f"{champion_name}.joblib")
                if os.path.exists(model_path):
                    model = joblib.load(model_path)
                    self.champion = {
                    'name': champion_name,
                    'model': model,
                    'roc_auc': metadata.get('champion_roc_auc', 0),
                    'loaded_from_disk': True,
                    'last_trained': metadata.get('last_trained', 'unknown')
                }
                logger.info(f"⚠️ Loaded champion from disk. Metrics may be outdated (trained: {metadata.get('last_trained', 'unknown')})")
            
            # Load top 3 models
            for model_name in metadata.get('top_3', []):
                model_path = os.path.join(self.models_dir, f"{model_name}.joblib")
                if os.path.exists(model_path):
                    model = joblib.load(model_path)
                    self.top_3_ensemble.append({
                        'name': model_name,
                        'model': model,
                        'roc_auc': metadata.get('champion_roc_auc', 0) if model_name == champion_name else 0
                    })
            
            logger.info(f"✓ Loaded existing models. Champion: {champion_name}")
        except Exception as e:
            logger.error(f"Error loading models: {e}")

# Global instances
cache_manager = CacheManager()
feature_service = FeatureService()
ml_manager = MLModelManager()

# Background polling
background_task_running = False
background_task_thread = None

def background_polling_task():
    """Background task that checks for data changes every 15 minutes"""
    global background_task_running
    
    logger.info("🔄 Background polling task started")
    
    consecutive_failures = 0
    max_failures = 5
    
    while background_task_running:
        try:
            # Check if it's time to poll
            if cache_manager.should_check_for_updates():
                logger.info("⏰ Time to check for data updates...")
                
                # Check for changes
                refresh_result = cache_manager.refresh_all_data()
                
                # If data changed, trigger retraining
                if refresh_result['any_changes']:
                    logger.info(f"ðŸ”„ AUTO-RETRAIN: Data changed in {refresh_result['total_changed']} endpoint(s)")
                    
                    try:
                        # Load data and retrain
                        games_daily = cache_manager.fetch_data('games_daily_activity', force=False)
                        training_df = feature_service.create_training_dataset(games_daily)
                        
                        if len(training_df) >= config.min_training_samples:
                            # Train models
                            ml_results = ml_manager.train_and_evaluate_all(training_df)
                            
                            if ml_results:
                                # Generate predictions
                                prediction_df = feature_service.create_prediction_features(games_daily)
                                
                                if not prediction_df.empty:
                                    # Champion predictions
                                    champion_pred = ml_manager.predict_champion(prediction_df)
                                    prediction_df_champion = prediction_df.copy()
                                    prediction_df_champion['decline_probability'] = champion_pred
                                    
                                    p75 = np.percentile(champion_pred, 75)
                                    p50 = np.percentile(champion_pred, 50)
                                    high_threshold = max(0.5, min(0.8, p75))
                                    medium_threshold = max(0.2, min(0.5, p50))
                                    
                                    prediction_df_champion['risk_level'] = prediction_df_champion['decline_probability'].apply(
                                        lambda x: 'High' if x > high_threshold else ('Medium' if x > medium_threshold else 'Low')
                                    )
                                    
                                    cache_manager.cache_data('predictions_champion', prediction_df_champion)
                                    
                                    # Ensemble predictions
                                    ensemble_pred = ml_manager.predict_ensemble(prediction_df)
                                    prediction_df_ensemble = prediction_df.copy()
                                    prediction_df_ensemble['decline_probability'] = ensemble_pred
                                    
                                    p75_ens = np.percentile(ensemble_pred, 75)
                                    p50_ens = np.percentile(ensemble_pred, 50)
                                    high_threshold_ens = max(0.5, min(0.8, p75_ens))
                                    medium_threshold_ens = max(0.2, min(0.5, p50_ens))
                                    
                                    prediction_df_ensemble['risk_level'] = prediction_df_ensemble['decline_probability'].apply(
                                        lambda x: 'High' if x > high_threshold_ens else ('Medium' if x > medium_threshold_ens else 'Low')
                                    )
                                    
                                    cache_manager.cache_data('predictions_ensemble', prediction_df_ensemble)
                                    
                                    logger.info("✓ AUTO-RETRAIN COMPLETE - New predictions available")
                        else:
                            logger.warning(f"⚠️ Insufficient training samples: {len(training_df)}")
                            
                    except Exception as e:
                        logger.error(f"❌ AUTO-RETRAIN FAILED: {e}")
                else:
                    logger.info("✓ No data changes - predictions unchanged")

            # Reset failure counter on success
            consecutive_failures = 0
            
            # Sleep for 60 seconds before checking again
            import time
            time.sleep(60)
            
        except Exception as e:
            consecutive_failures += 1
            logger.error(f"❌ Error in background polling (failure {consecutive_failures}/{max_failures}): {e}")
            
            if consecutive_failures >= max_failures:
                logger.critical(f"🚨 Background polling failed {max_failures} times in a row! Stopping polling.")
                background_task_running = False
                break
            
            # Exponential backoff: 60s, 120s, 240s, etc.
            backoff_time = min(60 * (2 ** consecutive_failures), 900)  # Max 15 min
            logger.warning(f"⏳ Backing off for {backoff_time}s before retry...")
            
            import time
            time.sleep(backoff_time)

# ==================== FASTAPI APP ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global background_task_running, background_task_thread
    
    logger.info("=" * 60)
    logger.info("Starting Ronin Gaming ML Analytics API v1.0")
    logger.info(f"XGBoost: {XGBOOST_AVAILABLE} | LightGBM: {LIGHTGBM_AVAILABLE}")
    logger.info(f"Smart Polling: Every {config.check_interval_seconds}s (15 min)")
    logger.info("=" * 60)
    
    # Check if models exist, if not, train on startup
    if ml_manager.champion is None:
        logger.info("ðŸš€ No existing models found. Training on startup...")
        try:
            # Fetch all data
            refresh_result = cache_manager.refresh_all_data()
            
            # Load games daily activity
            games_daily = cache_manager.fetch_data('games_daily_activity', force=False)
            
            # Create training dataset
            training_df = feature_service.create_training_dataset(games_daily)
            
            if len(training_df) >= config.min_training_samples:
                # Train models
                ml_results = ml_manager.train_and_evaluate_all(training_df)
                
                if ml_results:
                    # Generate predictions
                    prediction_df = feature_service.create_prediction_features(games_daily)
                    
                    if not prediction_df.empty:
                        # Champion predictions
                        champion_pred = ml_manager.predict_champion(prediction_df)
                        prediction_df_champion = prediction_df.copy()
                        prediction_df_champion['decline_probability'] = champion_pred
                        
                        p75 = np.percentile(champion_pred, 75)
                        p50 = np.percentile(champion_pred, 50)
                        high_threshold = max(0.5, min(0.8, p75))
                        medium_threshold = max(0.2, min(0.5, p50))
                        
                        prediction_df_champion['risk_level'] = prediction_df_champion['decline_probability'].apply(
                            lambda x: 'High' if x > high_threshold else ('Medium' if x > medium_threshold else 'Low')
                        )
                        
                        cache_manager.cache_data('predictions_champion', prediction_df_champion)
                        
                        # Ensemble predictions
                        ensemble_pred = ml_manager.predict_ensemble(prediction_df)
                        prediction_df_ensemble = prediction_df.copy()
                        prediction_df_ensemble['decline_probability'] = ensemble_pred
                        
                        p75_ens = np.percentile(ensemble_pred, 75)
                        p50_ens = np.percentile(ensemble_pred, 50)
                        high_threshold_ens = max(0.5, min(0.8, p75_ens))
                        medium_threshold_ens = max(0.2, min(0.5, p50_ens))
                        
                        prediction_df_ensemble['risk_level'] = prediction_df_ensemble['decline_probability'].apply(
                            lambda x: 'High' if x > high_threshold_ens else ('Medium' if x > medium_threshold_ens else 'Low')
                        )
                        
                        cache_manager.cache_data('predictions_ensemble', prediction_df_ensemble)
                        
                        logger.info("✓ STARTUP TRAINING COMPLETE - Models ready!")
                else:
                    logger.warning("⚠️ Startup training failed")
            else:
                logger.warning(f"⚠️ Insufficient training samples: {len(training_df)}")
        except Exception as e:
            logger.error(f"❌ Startup training failed: {e}")
    else:
        logger.info("✓ Existing models loaded from disk")
    
    # Start background polling task
    background_task_running = True
    background_task_thread = Thread(target=background_polling_task, daemon=True)
    background_task_thread.start()
    logger.info("✓ Background polling task started")
    
    yield
    
    # Stop background task
    logger.info("Stopping background polling task...")
    background_task_running = False
    if background_task_thread:
        background_task_thread.join(timeout=5)
    logger.info("Shutting down API")

app = FastAPI(
    title="Ronin Gaming ML Analytics API",
    description="Game health prediction and analytics for Ronin gaming ecosystem",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== ROOT ENDPOINT ====================

@app.get("/")
async def root():
    return {
        "message": "Ronin Gaming ML Analytics API",
        "version": "1.0.0",
        "status": "online",
        "documentation": "/docs",
        "features": [
            "Game health prediction",
            "13 Ronin analytics endpoints",
            "Smart polling (checks every 15 min)",
            "Multi-model ML ensemble",
            "Auto-training on data changes"
        ],
        "endpoints": {
            "system": {
                "health_check": "/api/health",
                "polling_status": "/api/polling/status",
                "api_docs": "/docs"
            },
            "data": {
                "games_daily_activity": "/api/data/games-daily",
                "user_retention": "/api/data/retention"
            },
            "ml_operations": {
                "refresh_and_train": "/api/refresh?force=true (POST)",
                "smart_refresh": "/api/refresh?force=false (POST)"
            },
            "predictions": {
                "game_health": "/api/predictions/game-health?method=ensemble",
                "games_at_risk": "/api/predictions/games-at-risk?limit=10",
                "trending_games": "/api/predictions/trending-games?limit=10",
                "all_predictions": "/api/bulk/predictions"
            },
            "models": {
                "model_info": "/api/models/info"
            }
        },
        "quick_start": {
            "1_check_health": "GET /api/health",
            "2_train_models": "POST /api/refresh?force=true",
            "3_get_predictions": "GET /api/predictions/game-health"
        }
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "ml_models_trained": ml_manager.champion is not None,
        "champion_model": ml_manager.champion['name'] if ml_manager.champion else None,
        "models_available": list(ml_manager.model_configs.keys()),
        "xgboost_available": XGBOOST_AVAILABLE,
        "lightgbm_available": LIGHTGBM_AVAILABLE,
        "background_polling": background_task_running
    }

@app.get("/api/polling/status")
async def get_polling_status():
    """Get status of background polling system"""
    last_check = cache_manager.last_check_time
    
    if last_check:
        last_check_dt = datetime.fromisoformat(last_check)
        seconds_since_check = (datetime.now() - last_check_dt).total_seconds()
        seconds_until_next = max(0, config.check_interval_seconds - seconds_since_check)
    else:
        seconds_since_check = None
        seconds_until_next = 0
    
    return {
        "polling_enabled": background_task_running,
        "check_interval_seconds": config.check_interval_seconds,
        "check_interval_minutes": config.check_interval_seconds / 60,
        "last_check": last_check,
        "seconds_since_last_check": seconds_since_check,
        "seconds_until_next_check": seconds_until_next,
        "minutes_until_next_check": seconds_until_next / 60 if seconds_until_next else 0,
        "data_hashes": {k: v[:8] + "..." for k, v in cache_manager.last_data_hash.items()},  # Show first 8 chars
        "predictions_available": cache_manager.get_cached_data('predictions_champion') is not None
    }

# ==================== DATA ENDPOINTS ====================

@app.get("/api/data/games-daily")
async def get_games_daily():
    """Get games daily activity data"""
    try:
        df = cache_manager.fetch_data('games_daily_activity')
        df_clean = clean_dataframe_for_json(df)
        
        return {
            "data": df_clean.to_dict('records'),
            "metadata": {
                "source": "games_daily_activity",
                "row_count": len(df),
                "columns": list(df.columns)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/data/retention")
async def get_retention():
    """Get user retention data"""
    try:
        df = cache_manager.fetch_data('user_activation_retention')
        df_clean = clean_dataframe_for_json(df)
        
        # Extra safety: convert to records and clean any remaining bad values
        records = df_clean.to_dict('records')
        
        # Clean each record
        for record in records:
            for key, value in record.items():
                if isinstance(value, float):
                    if np.isnan(value) or np.isinf(value):
                        record[key] = None
        
        return {
            "data": records,
            "metadata": {
                "source": "user_activation_retention",
                "row_count": len(df),
                "columns": list(df.columns)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== ML ENDPOINTS ====================

@app.post("/api/refresh")
async def force_refresh(force: bool = Query(False, description="Force retrain even if no data changed")):
    """
    Smart refresh: Check for data changes and only retrain if data changed
    Use force=true to retrain regardless of changes
    """
    try:
        start_time = time.time()
        logger.info("=" * 60)
        logger.info("SMART REFRESH INITIATED")
        logger.info("=" * 60)
        
        # Step 1: Check for data changes
        logger.info("Step 1: Checking for data changes...")
        refresh_result = cache_manager.refresh_all_data()
        
        # If no changes and not forcing, return early
        if not refresh_result['any_changes'] and not force:
            logger.info("✓ No data changes detected. Using existing predictions.")
            
            # Check if we have existing predictions
            predictions_exist = cache_manager.get_cached_data('predictions_champion') is not None
            
            elapsed_time = time.time() - start_time
            
            return {
                "status": "no_changes",
                "message": "No data changes detected. Serving cached predictions.",
                "timestamp": datetime.now().isoformat(),
                "elapsed_time_seconds": round(elapsed_time, 2),
                "any_changes": False,
                "predictions_available": predictions_exist,
                "next_check": "Will check again in 15 minutes"
            }
        
        # If we get here, either data changed or force=true
        if force:
            logger.info("✔️ Force retrain requested")
        else:
            logger.info(f"🔄 Data changed in {refresh_result['total_changed']} endpoint(s). Retraining...")
        
        # Step 2: Load games daily activity
        logger.info("Step 2: Loading games daily activity...")
        games_daily = cache_manager.fetch_data('games_daily_activity', force=False)
        
        # Step 3: Create training dataset
        logger.info("Step 3: Creating training dataset...")
        training_df = feature_service.create_training_dataset(games_daily)
        
        if len(training_df) < config.min_training_samples:
            return {
                "status": "partial_success",
                "message": f"Insufficient training samples ({len(training_df)} < {config.min_training_samples})",
                "any_changes": refresh_result['any_changes'],
                "changed_endpoints": refresh_result['changed_endpoints'],
                "models_trained": 0
            }
        
        # Step 4: Train models
        logger.info("Step 4: Training ML models...")
        ml_results = ml_manager.train_and_evaluate_all(training_df)
        
        if not ml_results:
            return {
                "status": "failed",
                "message": "Model training failed - check logs",
                "any_changes": refresh_result['any_changes'],
                "changed_endpoints": refresh_result['changed_endpoints'],
                "models_trained": 0
            }
        
        # Step 5: Generate predictions
        logger.info("Step 5: Generating predictions...")
        prediction_df = feature_service.create_prediction_features(games_daily)
        
        if not prediction_df.empty:
            # Champion predictions
            champion_pred = ml_manager.predict_champion(prediction_df)
            prediction_df_champion = prediction_df.copy()
            prediction_df_champion['decline_probability'] = champion_pred
            
            # Dynamic thresholds (percentile-based)
            p75 = np.percentile(champion_pred, 75)
            p50 = np.percentile(champion_pred, 50)
            high_threshold = max(0.5, min(0.8, p75))
            medium_threshold = max(0.2, min(0.5, p50))
            
            prediction_df_champion['risk_level'] = prediction_df_champion['decline_probability'].apply(
                lambda x: 'High' if x > high_threshold else ('Medium' if x > medium_threshold else 'Low')
            )
            
            cache_manager.cache_data('predictions_champion', prediction_df_champion)
            
            # Ensemble predictions
            ensemble_pred = ml_manager.predict_ensemble(prediction_df)
            prediction_df_ensemble = prediction_df.copy()
            prediction_df_ensemble['decline_probability'] = ensemble_pred
            
            p75_ens = np.percentile(ensemble_pred, 75)
            p50_ens = np.percentile(ensemble_pred, 50)
            high_threshold_ens = max(0.5, min(0.8, p75_ens))
            medium_threshold_ens = max(0.2, min(0.5, p50_ens))
            
            prediction_df_ensemble['risk_level'] = prediction_df_ensemble['decline_probability'].apply(
                lambda x: 'High' if x > high_threshold_ens else ('Medium' if x > medium_threshold_ens else 'Low')
            )
            
            cache_manager.cache_data('predictions_ensemble', prediction_df_ensemble)
        
        elapsed_time = time.time() - start_time
        
        logger.info("=" * 60)
        logger.info(f"REFRESH COMPLETE in {elapsed_time:.1f}s")
        logger.info(f"Champion: {ml_manager.champion['name']}")
        logger.info(f"ROC-AUC: {ml_manager.champion['roc_auc']:.4f}")
        logger.info("=" * 60)
        
        return {
            "status": "success",
            "message": "Data refreshed and ML models trained successfully",
            "timestamp": datetime.now().isoformat(),
            "elapsed_time_seconds": round(elapsed_time, 2),
            "any_changes": refresh_result['any_changes'],
            "changed_endpoints": refresh_result['changed_endpoints'],
            "total_changed": refresh_result['total_changed'],
            "total_endpoints": 13,
            "models_trained": len(ml_results),
            "champion_model": ml_manager.champion['name'],
            "champion_roc_auc": safe_float(ml_manager.champion['roc_auc']),
            "champion_accuracy": safe_float(ml_manager.champion.get('accuracy', 0)),
            "top_3_ensemble": [m['name'] for m in ml_manager.top_3_ensemble],
            "training_samples": len(training_df),
            "predictions_generated": len(prediction_df) if not prediction_df.empty else 0
        }
        
    except Exception as e:
        logger.error(f"Error in force refresh: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/predictions/game-health")
async def predict_game_health(method: str = Query('champion', pattern='^(champion|ensemble)$')):
    """
    Get game health predictions
    Returns decline probability for each game
    """
    try:
        cache_key = f'predictions_{method}'
        predictions_df = cache_manager.get_cached_data(cache_key)
        
        if predictions_df is None:
            raise HTTPException(
                status_code=404,
                detail="No predictions available. Run /api/refresh first."
            )
        
        predictions_df_clean = clean_dataframe_for_json(predictions_df)
        
        # Sort by decline probability (highest risk first)
        predictions_df_clean = predictions_df_clean.sort_values('decline_probability', ascending=False)
        
        return {
            "method": method,
            "timestamp": datetime.now().isoformat(),
            "predictions": predictions_df_clean.to_dict('records'),
            "summary": {
                "total_games": len(predictions_df),
                "high_risk": len(predictions_df[predictions_df['risk_level'] == 'High']),
                "medium_risk": len(predictions_df[predictions_df['risk_level'] == 'Medium']),
                "low_risk": len(predictions_df[predictions_df['risk_level'] == 'Low'])
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/predictions/games-at-risk")
async def get_games_at_risk(
    limit: int = Query(10, ge=1, le=50),
    method: str = Query('champion', pattern='^(champion|ensemble)$')  
):
    """Get top N games at risk of declining"""
    try:
        # Try champion first, fall back to ensemble
        cache_key = f'predictions_{method}'
        predictions_df = cache_manager.get_cached_data(cache_key)
        
        # If requested method not available, try the other one
        if predictions_df is None:
            alternative_method = 'ensemble' if method == 'champion' else 'champion'
            predictions_df = cache_manager.get_cached_data(f'predictions_{alternative_method}')
            if predictions_df is not None:
                method = alternative_method  # Update method to reflect what we're actually using
        
        if predictions_df is None:
            raise HTTPException(
                status_code=404,
                detail="No predictions available. Run /api/refresh first."
            )
        
        # Filter high risk games and sort
        at_risk = predictions_df[predictions_df['risk_level'] == 'High'].copy()
        at_risk = at_risk.sort_values('decline_probability', ascending=False).head(limit)
        
        at_risk_clean = clean_dataframe_for_json(at_risk)
        
        return {
            "method": method,
            "timestamp": datetime.now().isoformat(),
            "at_risk_games": at_risk_clean.to_dict('records'),
            "count": len(at_risk)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/predictions/trending-games")
async def get_trending_games(
    limit: int = Query(10, ge=1, le=50),
    method: str = Query('champion', pattern='^(champion|ensemble)$')
):
    """Get top N games that are growing (low decline probability)"""
    try:
        # Try champion first, fall back to ensemble
        cache_key = f'predictions_{method}'
        predictions_df = cache_manager.get_cached_data(cache_key)
        
        # If requested method not available, try the other one
        if predictions_df is None:
            alternative_method = 'ensemble' if method == 'champion' else 'champion'
            predictions_df = cache_manager.get_cached_data(f'predictions_{alternative_method}')
            if predictions_df is not None:
                method = alternative_method  # Update method to reflect what we're actually using
        
        if predictions_df is None:
            raise HTTPException(
                status_code=404,
                detail="No predictions available. Run /api/refresh first."
            )
        
        # Sort by lowest decline probability (most likely to grow)
        trending = predictions_df.sort_values('decline_probability', ascending=True).head(limit)
        
        trending_clean = clean_dataframe_for_json(trending)
        
        return {
            "method": method,
            "timestamp": datetime.now().isoformat(),
            "trending_games": trending_clean.to_dict('records'),
            "count": len(trending)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/models/info")
async def get_model_info():
    """Get information about trained models"""
    if not ml_manager.champion:
        raise HTTPException(
            status_code=404,
            detail="No models trained yet. Run /api/refresh first."
        )
    
    return {
        "champion": {
            "name": ml_manager.champion['name'],
            "roc_auc": safe_float(ml_manager.champion['roc_auc']),
            "accuracy": safe_float(ml_manager.champion.get('accuracy', 0)),
            "precision": safe_float(ml_manager.champion.get('precision', 0)),
            "recall": safe_float(ml_manager.champion.get('recall', 0))
        },
        "top_3_ensemble": [
            {
                "name": m['name'],
                "roc_auc": safe_float(m['roc_auc']),
                "accuracy": safe_float(m.get('accuracy', 0))
            }
            for m in ml_manager.top_3_ensemble
        ],
        "all_models": [
            {
                "name": m['name'],
                "roc_auc": safe_float(m['roc_auc']),
                "accuracy": safe_float(m.get('accuracy', 0)),
                "training_time": safe_float(m.get('training_time', 0))
            }
            for m in ml_manager.all_models
        ]
    }

# ==================== BULK ENDPOINTS ====================

@app.get("/api/bulk/predictions")
async def get_all_predictions():
    """Get all predictions at once"""
    try:
        result = {
            "timestamp": datetime.now().isoformat(),
            "predictions": {}
        }
        
        try:
            result['predictions']['game_health'] = await predict_game_health(method='ensemble')
        except HTTPException as e:
            logger.error(f"game_health failed: {e.detail}")
            result['predictions']['game_health'] = {"error": str(e.detail)}
        except Exception as e:
            logger.error(f"Unexpected error in game_health: {e}", exc_info=True)
            result['predictions']['game_health'] = {"error": "Service unavailable"}

        try:
            result['predictions']['games_at_risk'] = await get_games_at_risk(limit=10)
        except HTTPException as e:
            logger.error(f"games_at_risk failed: {e.detail}")
            result['predictions']['games_at_risk'] = {"error": str(e.detail)}
        except Exception as e:
            logger.error(f"Unexpected error in games_at_risk: {e}", exc_info=True)
            result['predictions']['games_at_risk'] = {"error": "Service unavailable"}

        try:
            result['predictions']['trending_games'] = await get_trending_games(limit=10)
        except HTTPException as e:
            logger.error(f"trending_games failed: {e.detail}")
            result['predictions']['trending_games'] = {"error": str(e.detail)}
        except Exception as e:
            logger.error(f"Unexpected error in trending_games: {e}", exc_info=True)
            result['predictions']['trending_games'] = {"error": "Service unavailable"}

        try:
            result['model_info'] = await get_model_info()
        except HTTPException as e:
            logger.error(f"model_info failed: {e.detail}")
            result['model_info'] = {"error": str(e.detail)}
        except Exception as e:
            logger.error(f"Unexpected error in model_info: {e}", exc_info=True)
            result['model_info'] = {"error": "Service unavailable"}
        
        return result
        
    except Exception as e:
        logger.error(f"Error in bulk predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== RUN ====================

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False
    )