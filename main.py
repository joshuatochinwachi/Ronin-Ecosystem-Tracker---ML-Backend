"""
Ronin Ecosystem ML Backend - FastAPI Entry Point
Provides ML predictions and insights for the Ronin blockchain ecosystem
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import ml_routes
from app.config import settings
from app.utils.logger import logger
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="Ronin Ecosystem ML API",
    description="Machine Learning predictions and analytics for Ronin blockchain",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include ML routes
app.include_router(ml_routes.router, prefix="/ml", tags=["Machine Learning"])

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("🚀 Ronin ML Backend starting up...")
    logger.info(f"📍 Ronin API Base URL: {settings.RONIN_API_BASE_URL}")
    logger.info(f"🔧 Debug Mode: {settings.DEBUG}")
    logger.info(f"⏱️  Cache TTL: {settings.CACHE_TTL_SECONDS}s")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("👋 Ronin ML Backend shutting down...")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "Ronin Ecosystem ML Backend",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "ml_apis": "/ml/*"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "service": "ronin-ml-backend",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )