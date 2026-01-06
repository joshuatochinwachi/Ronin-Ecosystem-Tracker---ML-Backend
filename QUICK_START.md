# 🚀 Ronin ML Backend - Quick Start Guide

This guide will help you get the ML backend running in **5 minutes**.

## Prerequisites
- Python 3.11+ installed
- OR Docker installed

## Option 1: Quick Local Setup (Recommended for Development)

### Step 1: Clone and Setup
```bash
# Navigate to your project directory
cd ronin-ml-backend

# Make the run script executable
chmod +x run.sh

# Run the setup and start script
./run.sh
```

The script will:
- ✅ Check Python installation
- ✅ Create virtual environment
- ✅ Install all dependencies
- ✅ Create necessary directories
- ✅ Start the server

### Step 2: Verify It's Running
Open your browser and visit:
- **API**: http://localhost:8000
- **Health Check**: http://localhost:8000/health
- **API Docs**: http://localhost:8000/docs

You should see the Swagger UI with all 8 ML endpoints!

### Step 3: Test the Endpoints
```bash
# Make test script executable
chmod +x test_endpoints.sh

# Run tests
./test_endpoints.sh
```

This will test all ML endpoints and show you the responses.

---

## Option 2: Docker Setup (Recommended for Production)

### Step 1: Build and Run
```bash
# Build and start with Docker Compose
docker-compose up -d

# Check logs
docker-compose logs -f
```

### Step 2: Verify
Visit http://localhost:8000/docs

### Step 3: Stop
```bash
docker-compose down
```

---

## 🎯 Testing Individual Endpoints

### 1. Ecosystem Health
```bash
curl http://localhost:8000/ml/ecosystem-health | jq
```

### 2. Volume Forecast (7 days)
```bash
curl "http://localhost:8000/ml/volume-forecast?days=7" | jq
```

### 3. Whale Behavior Analysis
```bash
curl http://localhost:8000/ml/whale-behavior | jq
```

### 4. Game Churn Prediction
```bash
curl "http://localhost:8000/ml/game-churn-prediction?game=Axie%20Infinity" | jq
```

### 5. Anomaly Detection
```bash
curl http://localhost:8000/ml/anomaly-detection | jq
```

### 6. Holder Segmentation
```bash
curl http://localhost:8000/ml/holder-segmentation | jq
```

### 7. NFT Trends
```bash
curl http://localhost:8000/ml/nft-trends | jq
```

### 8. Network Stress Test
```bash
curl http://localhost:8000/ml/network-stress-test | jq
```

---

## 🔗 Connecting Your Frontend

In your Next.js frontend, you can now call these endpoints:

```typescript
// Example: Fetch ecosystem health
const response = await fetch('http://localhost:8000/ml/ecosystem-health');
const data = await response.json();

console.log(data.data.health_score); // 87.5
console.log(data.data.status);       // "Healthy"
```

---

## 📊 API Response Format

All endpoints return a standardized response:

```json
{
  "success": true,
  "data": {
    // Your ML insights here
  },
  "metadata": {
    "timestamp": "2026-01-06T12:00:00Z",
    "version": "1.0.0"
  }
}
```

---

## 🔧 Configuration

The backend is already configured to use your production Ronin API:

```
Base URL: https://web-production-4fae.up.railway.app
```

It will automatically fetch data from all 13 endpoints:
- ✅ /api/dune/network-activity
- ✅ /api/dune/volume-liquidity
- ✅ /api/dune/games-daily
- ✅ /api/dune/weekly-segmentation
- ✅ /api/dune/hourly
- ✅ /api/dune/whales
- ✅ /api/dune/ronin-daily
- ✅ /api/dune/retention
- ✅ /api/dune/trade-pairs
- ✅ /api/dune/games-overall
- ✅ /api/dune/holders
- ✅ /api/dune/segmented-holders
- ✅ /api/dune/nft-collections

---

## 🐛 Troubleshooting

### Port 8000 already in use?
```bash
# Change the port in .env
echo "PORT=8001" >> .env

# Restart
./run.sh
```

### Dependencies not installing?
```bash
# Upgrade pip first
pip install --upgrade pip

# Then install requirements
pip install -r requirements.txt
```

### Can't connect to Ronin API?
Check the logs for connection errors:
```bash
# Local
tail -f logs/*.log

# Docker
docker-compose logs -f
```

---

## 📈 Next Steps

1. **Integrate with Frontend**: Use the ML endpoints in your Next.js app
2. **Deploy**: Deploy to Railway, Render, or any cloud platform
3. **Monitor**: Check logs and health endpoint regularly
4. **Customize**: Add your own ML models in `app/ml/`

---

## 🆘 Need Help?

- Check logs in `logs/` directory
- Visit API docs at http://localhost:8000/docs
- Review the full README.md for detailed documentation

**You're all set! The ML backend is ready to power your Ronin Ecosystem Tracker! 🎉**