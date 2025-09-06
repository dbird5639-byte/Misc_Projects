"""
Main FastAPI application for AI Ad Exchange
"""

import logging
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

from config.settings import API_SETTINGS, LOGGING_SETTINGS
from src.exchange.ad_exchange import AdExchange

# Setup logging
logging.basicConfig(
    level=getattr(logging, LOGGING_SETTINGS['level']),
    format=LOGGING_SETTINGS['format'],
    handlers=[
        logging.FileHandler(LOGGING_SETTINGS['file']),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=API_SETTINGS['title'],
    version=API_SETTINGS['version'],
    debug=API_SETTINGS['debug']
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize ad exchange
ad_exchange = AdExchange()

# Mount static files for frontend
app.mount("/static", StaticFiles(directory="src/frontend"), name="static")

@app.on_event("startup")
async def startup_event():
    """Application startup event"""
    logger.info("AI Ad Exchange API starting up...")
    logger.info(f"API Title: {API_SETTINGS['title']}")
    logger.info(f"API Version: {API_SETTINGS['version']}")
    logger.info(f"Debug Mode: {API_SETTINGS['debug']}")

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event"""
    logger.info("AI Ad Exchange API shutting down...")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AI Ad Exchange API",
        "version": API_SETTINGS['version'],
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Get exchange stats as health indicator
        stats = ad_exchange.get_exchange_stats()
        return {
            "status": "healthy",
            "exchange_stats": stats,
            "timestamp": "2024-01-01T00:00:00Z"
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Health check failed")

@app.get("/api/v1/stats")
async def get_stats():
    """Get exchange statistics"""
    try:
        stats = ad_exchange.get_exchange_stats()
        return {
            "success": True,
            "data": stats
        }
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get statistics")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=API_SETTINGS['host'],
        port=API_SETTINGS['port'],
        reload=API_SETTINGS['debug']
    ) 