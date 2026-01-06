# 🚂 Deploy to Railway - Step by Step

Deploy your Ronin ML Backend to Railway in minutes!

## Why Railway?

- ✅ Automatic deployments from GitHub
- ✅ Built-in domain and SSL
- ✅ Easy environment variables
- ✅ Scales automatically
- ✅ Free tier available

---

## Prerequisites

1. GitHub account
2. Railway account (https://railway.app)
3. Your code pushed to GitHub

---

## Step 1: Prepare Your Repository

Make sure your repo has these files:
- ✅ `Dockerfile`
- ✅ `requirements.txt`
- ✅ `main.py`
- ✅ All `app/` directory files

---

## Step 2: Push to GitHub

```bash
# Initialize git (if not already done)
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit: Ronin ML Backend"

# Add remote (replace with your repo URL)
git remote add origin https://github.com/YOUR_USERNAME/ronin-ml-backend.git

# Push
git push -u origin main
```

---

## Step 3: Deploy on Railway

### Option A: Using Railway Dashboard

1. **Go to Railway**: https://railway.app
2. **Click "New Project"**
3. **Select "Deploy from GitHub repo"**
4. **Authorize Railway** to access your GitHub
5. **Select your repository**: `ronin-ml-backend`
6. **Railway will auto-detect** the Dockerfile and start building!

### Option B: Using Railway CLI

```bash
# Install Railway CLI
npm i -g @railway/cli

# Login
railway login

# Initialize project
railway init

# Deploy
railway up
```

---

## Step 4: Configure Environment Variables

In Railway Dashboard:

1. **Go to your project**
2. **Click "Variables" tab**
3. **Add these variables**:

```
PORT=8000
DEBUG=False
ENVIRONMENT=production
RONIN_API_BASE_URL=https://web-production-4fae.up.railway.app
CORS_ORIGINS=["https://your-frontend.vercel.app","http://localhost:3000"]
CACHE_TTL_SECONDS=300
LOG_LEVEL=INFO
```

4. **Click "Redeploy"** to apply changes

---

## Step 5: Get Your Backend URL

After deployment, Railway will give you a URL like:
```
https://ronin-ml-backend-production.up.railway.app
```

**Test it**:
```bash
curl https://your-railway-url.up.railway.app/health
```

You should see:
```json
{
  "status": "healthy",
  "service": "ronin-ml-backend",
  "version": "1.0.0"
}
```

---

## Step 6: Update Frontend to Use Railway Backend

In your Next.js frontend, update the API URL:

```typescript
// lib/api.ts or wherever you configure your API
const ML_API_BASE_URL = process.env.NEXT_PUBLIC_ML_API_URL || 
  'https://your-railway-url.up.railway.app';

// Example usage
export async function getEcosystemHealth() {
  const response = await fetch(`${ML_API_BASE_URL}/ml/ecosystem-health`);
  return response.json();
}
```

Add to your frontend's `.env.local`:
```bash
NEXT_PUBLIC_ML_API_URL=https://your-railway-url.up.railway.app
```

---

## Step 7: Enable Automatic Deployments

Railway automatically deploys on every push to `main` branch!

```bash
# Make a change
git add .
git commit -m "Update ML models"
git push

# Railway will automatically redeploy! 🎉
```

---

## 🔧 Railway Configuration Tips

### Custom Domain (Optional)

1. Go to **Settings** → **Domains**
2. Click **Generate Domain** or **Custom Domain**
3. Use your custom domain: `ml-api.yourdomain.com`

### Scaling

1. Go to **Settings** → **Resources**
2. Adjust:
   - **Memory**: 512MB - 4GB
   - **CPU**: Shared or Dedicated
   - **Replicas**: Scale horizontally

### Monitoring

Railway provides:
- **Metrics**: CPU, Memory, Network usage
- **Logs**: Real-time application logs
- **Deployments**: History and rollback

---

## 🔍 Testing Your Deployment

### Test Health Endpoint
```bash
curl https://your-railway-url.up.railway.app/health
```

### Test ML Endpoint
```bash
curl https://your-railway-url.up.railway.app/ml/ecosystem-health
```

### View API Docs
Visit: `https://your-railway-url.up.railway.app/docs`

---

## 🚨 Troubleshooting

### Build Fails?

**Check Build Logs** in Railway dashboard:
- Look for Python version errors
- Check dependency conflicts
- Verify Dockerfile syntax

**Common fixes**:
```dockerfile
# If build fails, try pinning Python version
FROM python:3.11.7-slim
```

### Port Issues?

Railway automatically sets `PORT` environment variable. Our app uses it:
```python
# main.py already handles this
port = settings.PORT
```

### Connection Timeouts?

Check CORS configuration:
```python
# app/config.py
CORS_ORIGINS = [
    "https://your-frontend.vercel.app",
    "https://your-frontend.netlify.app"
]
```

### High Memory Usage?

Reduce cache size:
```bash
# In Railway variables
CACHE_TTL_SECONDS=60  # Reduce from 300
```

---

## 💰 Cost Optimization

### Free Tier Limits
- $5 free credit/month
- 500 hours/month
- Shared CPU
- 512MB RAM

### Tips to Stay in Free Tier
1. **Use caching** to reduce API calls
2. **Optimize ML models** for memory
3. **Sleep on inactivity** (Railway does this automatically)
4. **Monitor usage** in Railway dashboard

### Scaling Beyond Free Tier
When you need more:
- **Hobby Plan**: $5/month
- **Pay-as-you-go**: $0.000463/GB-hour
- **Dedicated CPU**: Better performance

---

## 📊 Production Checklist

Before going live:

- [ ] Environment variables set correctly
- [ ] CORS origins include your production frontend
- [ ] Health endpoint responding
- [ ] All 8 ML endpoints working
- [ ] Logs are clean (no errors)
- [ ] Frontend can connect to backend
- [ ] API docs accessible
- [ ] Set up monitoring/alerts

---

## 🔄 CI/CD Pipeline

Railway provides automatic CI/CD:

```
┌─────────────┐
│  Git Push   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Railway   │
│   Builds    │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Deploy    │
│  (Auto)     │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│    Live!    │
└─────────────┘
```

---

## 🎯 Post-Deployment

### Monitor Your Backend

**Railway Dashboard**:
- Check metrics daily
- Review logs for errors
- Monitor memory usage

**Set up alerts**:
```bash
# Install Railway CLI
railway logs --follow

# Or check in dashboard
```

### Update CORS in Frontend

After deployment, update your frontend:

```typescript
// In your frontend API config
const API_ENDPOINTS = {
  ml: 'https://your-railway-url.up.railway.app/ml',
  health: 'https://your-railway-url.up.railway.app/health'
};
```

### Test from Frontend

```typescript
// Example: Fetch ecosystem health from your frontend
async function testMLBackend() {
  try {
    const response = await fetch(
      'https://your-railway-url.up.railway.app/ml/ecosystem-health'
    );
    const data = await response.json();
    console.log('✅ ML Backend connected!', data);
  } catch (error) {
    console.error('❌ Connection failed:', error);
  }
}
```

---

## 🎉 Success!

Your Ronin ML Backend is now:
- ✅ Deployed on Railway
- ✅ Automatically deploying on git push
- ✅ Secured with HTTPS
- ✅ Scalable and monitored
- ✅ Connected to your frontend

**Railway URL**: `https://your-project.up.railway.app`

**Next Steps**:
1. Share the API URL with your team
2. Monitor performance and logs
3. Scale as needed
4. Enjoy your ML-powered Ronin Ecosystem! 🚀

---

## 📚 Additional Resources

- Railway Docs: https://docs.railway.app
- Railway Discord: https://discord.gg/railway
- FastAPI Docs: https://fastapi.tiangolo.com
- Railway Status: https://railway.instatus.com

**Happy Deploying! 🚂✨**