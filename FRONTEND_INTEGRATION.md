# 🔗 Frontend Integration Guide

How to integrate the Ronin ML Backend with your Next.js frontend.

---

## Step 1: Create ML API Service

Create a new file in your frontend: `lib/ml-api.ts`

```typescript
// lib/ml-api.ts
const ML_API_BASE_URL = process.env.NEXT_PUBLIC_ML_API_URL || 
  'http://localhost:8000';

interface APIResponse<T> {
  success: boolean;
  data: T;
  metadata: {
    timestamp: string;
    version: string;
  };
}

// Generic fetch wrapper with error handling
async function fetchML<T>(endpoint: string): Promise<T> {
  const response = await fetch(`${ML_API_BASE_URL}${endpoint}`);
  
  if (!response.ok) {
    throw new Error(`ML API error: ${response.statusText}`);
  }
  
  const json: APIResponse<T> = await response.json();
  
  if (!json.success) {
    throw new Error('ML API returned unsuccessful response');
  }
  
  return json.data;
}

// ============================================
// ML API Functions
// ============================================

export interface EcosystemHealth {
  health_score: number;
  status: string;
  components: {
    network_activity: number;
    dex_volume: number;
    gaming_engagement: number;
    whale_activity: number;
    nft_market: number;
  };
  trends: {
    '7d_change': number;
    '30d_change': number;
    direction: string;
  };
  alerts: Array<{
    type: string;
    severity: string;
    message: string;
    metric: string;
  }>;
}

export async function getEcosystemHealth(): Promise<EcosystemHealth> {
  return fetchML<EcosystemHealth>('/ml/ecosystem-health');
}

export interface VolumeForecast {
  forecast: Array<{
    date: string;
    predicted_volume_usd: number;
    confidence_interval: {
      lower: number;
      upper: number;
    };
  }>;
  model_accuracy: number;
  last_trained: string;
  historical_avg: number;
}

export async function getVolumeForecast(days: number = 7): Promise<VolumeForecast> {
  return fetchML<VolumeForecast>(`/ml/volume-forecast?days=${days}`);
}

export interface WhaleBehavior {
  total_whales: number;
  active_whales_24h: number;
  whale_sentiment: string;
  patterns: {
    accumulation_phase: boolean;
    avg_trade_size_trend: string;
    buy_sell_ratio: number;
  };
  predictions: {
    likely_action_24h: string;
    confidence: number;
  };
}

export async function getWhaleBehavior(): Promise<WhaleBehavior> {
  return fetchML<WhaleBehavior>('/ml/whale-behavior');
}

export interface ChurnPrediction {
  game: string;
  churn_risk: string;
  churn_probability: number;
  at_risk_players: number;
  retention_forecast: {
    week_1: number;
    week_2: number;
    week_4: number;
  };
  recommendations: string[];
}

export async function getChurnPrediction(game?: string): Promise<ChurnPrediction> {
  const url = game 
    ? `/ml/game-churn-prediction?game=${encodeURIComponent(game)}`
    : '/ml/game-churn-prediction';
  return fetchML<ChurnPrediction>(url);
}

export interface AnomalyDetection {
  anomalies_detected: number;
  anomalies: Array<{
    metric: string;
    timestamp: string;
    value: number;
    expected_range: number[];
    severity: string;
    description: string;
  }>;
  overall_status: string;
}

export async function getAnomalyDetection(): Promise<AnomalyDetection> {
  return fetchML<AnomalyDetection>('/ml/anomaly-detection');
}

export interface HolderSegmentation {
  segments: Array<{
    segment: string;
    count: number;
    avg_hold_duration_days: number;
    avg_balance: number;
    churn_risk: string;
    percentage: number;
  }>;
  total_holders: number;
  analysis_date: string;
}

export async function getHolderSegmentation(): Promise<HolderSegmentation> {
  return fetchML<HolderSegmentation>('/ml/holder-segmentation');
}

export interface NFTTrends {
  market_sentiment: string;
  trending_collections: Array<{
    collection: string;
    momentum_score: number;
    predicted_floor_price_7d: number;
    volume_trend: string;
    volume_change_24h: number;
  }>;
  market_predictions: {
    total_volume_7d: number;
    confidence: number;
    expected_trend: string;
  };
}

export async function getNFTTrends(): Promise<NFTTrends> {
  return fetchML<NFTTrends>('/ml/nft-trends');
}

export interface NetworkStress {
  current_stress_level: string;
  current_gas_price: number;
  predicted_gas_price_1h: number;
  congestion_forecast: Array<{
    hour: string;
    stress_level: string;
    predicted_gas_gwei: number;
  }>;
}

export async function getNetworkStress(): Promise<NetworkStress> {
  return fetchML<NetworkStress>('/ml/network-stress-test');
}
```

---

## Step 2: Add Environment Variable

Add to your `.env.local`:

```bash
# Development
NEXT_PUBLIC_ML_API_URL=http://localhost:8000

# Production (update after Railway deployment)
# NEXT_PUBLIC_ML_API_URL=https://your-ml-backend.up.railway.app
```

---

## Step 3: Create React Components

### Ecosystem Health Dashboard

```typescript
// components/EcosystemHealthCard.tsx
'use client';

import { useEffect, useState } from 'react';
import { getEcosystemHealth, EcosystemHealth } from '@/lib/ml-api';

export function EcosystemHealthCard() {
  const [health, setHealth] = useState<EcosystemHealth | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function loadHealth() {
      try {
        const data = await getEcosystemHealth();
        setHealth(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load data');
      } finally {
        setLoading(false);
      }
    }
    
    loadHealth();
    // Refresh every 5 minutes
    const interval = setInterval(loadHealth, 5 * 60 * 1000);
    return () => clearInterval(interval);
  }, []);

  if (loading) return <div>Loading ecosystem health...</div>;
  if (error) return <div>Error: {error}</div>;
  if (!health) return null;

  return (
    <div className="p-6 bg-white rounded-lg shadow">
      <h2 className="text-2xl font-bold mb-4">Ecosystem Health</h2>
      
      {/* Health Score */}
      <div className="mb-6">
        <div className="text-4xl font-bold text-blue-600">
          {health.health_score.toFixed(1)}
        </div>
        <div className="text-lg text-gray-600">{health.status}</div>
      </div>

      {/* Components */}
      <div className="space-y-3">
        {Object.entries(health.components).map(([key, value]) => (
          <div key={key} className="flex items-center justify-between">
            <span className="capitalize">{key.replace('_', ' ')}</span>
            <div className="flex items-center gap-2">
              <div className="w-32 h-2 bg-gray-200 rounded">
                <div 
                  className="h-2 bg-blue-600 rounded"
                  style={{ width: `${value}%` }}
                />
              </div>
              <span className="text-sm font-medium">{value.toFixed(1)}</span>
            </div>
          </div>
        ))}
      </div>

      {/* Trends */}
      <div className="mt-6 grid grid-cols-2 gap-4">
        <div>
          <div className="text-sm text-gray-600">7d Change</div>
          <div className={`text-lg font-bold ${
            health.trends['7d_change'] > 0 ? 'text-green-600' : 'text-red-600'
          }`}>
            {health.trends['7d_change'] > 0 ? '+' : ''}
            {health.trends['7d_change'].toFixed(1)}%
          </div>
        </div>
        <div>
          <div className="text-sm text-gray-600">30d Change</div>
          <div className={`text-lg font-bold ${
            health.trends['30d_change'] > 0 ? 'text-green-600' : 'text-red-600'
          }`}>
            {health.trends['30d_change'] > 0 ? '+' : ''}
            {health.trends['30d_change'].toFixed(1)}%
          </div>
        </div>
      </div>

      {/* Alerts */}
      {health.alerts.length > 0 && (
        <div className="mt-6">
          <h3 className="font-semibold mb-2">Alerts</h3>
          <div className="space-y-2">
            {health.alerts.map((alert, i) => (
              <div 
                key={i}
                className={`p-3 rounded ${
                  alert.type === 'positive' ? 'bg-green-50 text-green-800' :
                  alert.type === 'warning' ? 'bg-yellow-50 text-yellow-800' :
                  'bg-red-50 text-red-800'
                }`}
              >
                {alert.message}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
```

### Volume Forecast Chart

```typescript
// components/VolumeForecastChart.tsx
'use client';

import { useEffect, useState } from 'react';
import { getVolumeForecast, VolumeForecast } from '@/lib/ml-api';

export function VolumeForecastChart() {
  const [forecast, setForecast] = useState<VolumeForecast | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadForecast() {
      try {
        const data = await getVolumeForecast(7);
        setForecast(data);
      } catch (err) {
        console.error('Failed to load forecast:', err);
      } finally {
        setLoading(false);
      }
    }
    
    loadForecast();
  }, []);

  if (loading) return <div>Loading forecast...</div>;
  if (!forecast) return null;

  return (
    <div className="p-6 bg-white rounded-lg shadow">
      <h2 className="text-2xl font-bold mb-4">7-Day Volume Forecast</h2>
      
      <div className="mb-4">
        <span className="text-sm text-gray-600">Model Accuracy: </span>
        <span className="font-semibold">
          {(forecast.model_accuracy * 100).toFixed(1)}%
        </span>
      </div>

      <div className="space-y-2">
        {forecast.forecast.map((day) => (
          <div key={day.date} className="flex items-center justify-between p-2 hover:bg-gray-50 rounded">
            <span className="text-sm">{day.date}</span>
            <div className="text-right">
              <div className="font-semibold">
                ${(day.predicted_volume_usd / 1000000).toFixed(2)}M
              </div>
              <div className="text-xs text-gray-500">
                ${(day.confidence_interval.lower / 1000000).toFixed(2)}M - 
                ${(day.confidence_interval.upper / 1000000).toFixed(2)}M
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
```

### Whale Behavior Widget

```typescript
// components/WhaleBehaviorWidget.tsx
'use client';

import { useEffect, useState } from 'react';
import { getWhaleBehavior, WhaleBehavior } from '@/lib/ml-api';

export function WhaleBehaviorWidget() {
  const [whales, setWhales] = useState<WhaleBehavior | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadWhales() {
      try {
        const data = await getWhaleBehavior();
        setWhales(data);
      } catch (err) {
        console.error('Failed to load whale data:', err);
      } finally {
        setLoading(false);
      }
    }
    
    loadWhales();
    const interval = setInterval(loadWhales, 2 * 60 * 1000); // Every 2 min
    return () => clearInterval(interval);
  }, []);

  if (loading) return <div>Loading whale data...</div>;
  if (!whales) return null;

  const sentimentColor = 
    whales.whale_sentiment === 'bullish' ? 'text-green-600' :
    whales.whale_sentiment === 'bearish' ? 'text-red-600' :
    'text-gray-600';

  return (
    <div className="p-6 bg-white rounded-lg shadow">
      <h2 className="text-2xl font-bold mb-4">🐋 Whale Behavior</h2>
      
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div>
          <div className="text-sm text-gray-600">Total Whales</div>
          <div className="text-2xl font-bold">{whales.total_whales}</div>
        </div>
        <div>
          <div className="text-sm text-gray-600">Active (24h)</div>
          <div className="text-2xl font-bold">{whales.active_whales_24h}</div>
        </div>
      </div>

      <div className="mb-4">
        <div className="text-sm text-gray-600">Sentiment</div>
        <div className={`text-xl font-bold capitalize ${sentimentColor}`}>
          {whales.whale_sentiment}
        </div>
      </div>

      <div className="space-y-2 text-sm">
        <div className="flex justify-between">
          <span>Buy/Sell Ratio:</span>
          <span className="font-semibold">
            {whales.patterns.buy_sell_ratio.toFixed(2)}
          </span>
        </div>
        <div className="flex justify-between">
          <span>Accumulation:</span>
          <span className="font-semibold">
            {whales.patterns.accumulation_phase ? 'Yes' : 'No'}
          </span>
        </div>
        <div className="flex justify-between">
          <span>Likely Action:</span>
          <span className="font-semibold capitalize">
            {whales.predictions.likely_action_24h}
          </span>
        </div>
      </div>
    </div>
  );
}
```

---

## Step 4: Use in Your Pages

```typescript
// app/dashboard/page.tsx
import { EcosystemHealthCard } from '@/components/EcosystemHealthCard';
import { VolumeForecastChart } from '@/components/VolumeForecastChart';
import { WhaleBehaviorWidget } from '@/components/WhaleBehaviorWidget';

export default function DashboardPage() {
  return (
    <div className="container mx-auto p-6">
      <h1 className="text-3xl font-bold mb-8">Ronin Ecosystem Dashboard</h1>
      
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <EcosystemHealthCard />
        <WhaleBehaviorWidget />
        <VolumeForecastChart />
        {/* Add more components as needed */}
      </div>
    </div>
  );
}
```

---

## Step 5: Error Handling & Loading States

```typescript
// hooks/useMLData.ts
import { useEffect, useState } from 'react';

export function useMLData<T>(
  fetcher: () => Promise<T>,
  refreshInterval?: number
) {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    async function load() {
      try {
        setLoading(true);
        const result = await fetcher();
        setData(result);
        setError(null);
      } catch (err) {
        setError(err instanceof Error ? err : new Error('Unknown error'));
      } finally {
        setLoading(false);
      }
    }

    load();

    if (refreshInterval) {
      const interval = setInterval(load, refreshInterval);
      return () => clearInterval(interval);
    }
  }, [fetcher, refreshInterval]);

  return { data, loading, error, refetch: () => {} };
}
```

---

## 🎯 Production Checklist

Before deploying to production:

- [ ] Update `NEXT_PUBLIC_ML_API_URL` to your Railway backend URL
- [ ] Add error boundaries for ML components
- [ ] Implement loading skeletons
- [ ] Add retry logic for failed requests
- [ ] Set up proper CORS in backend
- [ ] Test all ML endpoints from frontend
- [ ] Add analytics/monitoring
- [ ] Implement caching strategy

---

## 🚀 You're Ready!

Your frontend is now fully integrated with the ML backend! 🎉

**Features you now have:**
- ✅ Real-time ecosystem health monitoring
- ✅ Volume forecasting
- ✅ Whale behavior tracking
- ✅ Churn predictions
- ✅ Anomaly alerts
- ✅ And more!

Happy coding! 🚀