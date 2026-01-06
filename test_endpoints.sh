#!/bin/bash

# API Testing Script for Ronin ML Backend
# Tests all ML endpoints

BASE_URL="http://localhost:8000"

echo "🧪 Testing Ronin ML Backend Endpoints"
echo "======================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to test endpoint
test_endpoint() {
    local name=$1
    local url=$2
    
    echo -e "${BLUE}Testing: $name${NC}"
    echo "URL: $url"
    
    response=$(curl -s -w "\n%{http_code}" "$url")
    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | sed '$d')
    
    if [ "$http_code" -eq 200 ]; then
        echo -e "${GREEN}✅ SUCCESS (HTTP $http_code)${NC}"
        echo "$body" | jq '.' 2>/dev/null || echo "$body"
    else
        echo -e "${RED}❌ FAILED (HTTP $http_code)${NC}"
        echo "$body"
    fi
    
    echo ""
    echo "---"
    echo ""
}

# Check if jq is installed for JSON formatting
if ! command -v jq &> /dev/null; then
    echo "⚠️  jq not found - JSON output will not be formatted"
    echo "   Install jq: brew install jq (macOS) or apt-get install jq (Linux)"
    echo ""
fi

# Test health endpoint
test_endpoint "Health Check" "$BASE_URL/health"

# Test root endpoint
test_endpoint "Root API Info" "$BASE_URL/"

# Test ML endpoints
test_endpoint "1. Ecosystem Health" "$BASE_URL/ml/ecosystem-health"

test_endpoint "2. Volume Forecast (7 days)" "$BASE_URL/ml/volume-forecast?days=7"

test_endpoint "3. Whale Behavior" "$BASE_URL/ml/whale-behavior"

test_endpoint "4. Game Churn Prediction" "$BASE_URL/ml/game-churn-prediction"

test_endpoint "5. Game Churn (Specific Game)" "$BASE_URL/ml/game-churn-prediction?game=Axie%20Infinity"

test_endpoint "6. Anomaly Detection" "$BASE_URL/ml/anomaly-detection"

test_endpoint "7. Holder Segmentation" "$BASE_URL/ml/holder-segmentation"

test_endpoint "8. NFT Trends" "$BASE_URL/ml/nft-trends"

test_endpoint "9. Network Stress Test" "$BASE_URL/ml/network-stress-test"

echo "======================================"
echo "✨ Testing Complete!"
echo ""
echo "📊 View API Documentation:"
echo "   Swagger UI: $BASE_URL/docs"
echo "   ReDoc:      $BASE_URL/redoc"