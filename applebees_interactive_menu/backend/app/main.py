"""
Applebee's Interactive Menu - Main FastAPI Application
Advanced features including AI recommendations, AR experiences, voice ordering, and real-time tracking.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn

from app.core.config import settings
from app.core.database import init_db, get_db
from app.core.security import get_current_user
from app.api.v1.api import api_router
from app.services.ai.recommendation import RecommendationService
from app.services.ai.voice import VoiceProcessingService
from app.services.ar.experience import ARExperienceService
from app.services.realtime.websocket import WebSocketManager
from app.services.payment.stripe_service import StripeService
from app.services.notification.push_service import PushNotificationService
from app.utils.logger import setup_logger

# Setup logging
logger = setup_logger(__name__)

# Global service instances
recommendation_service = None
voice_service = None
ar_service = None
websocket_manager = None
stripe_service = None
push_service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    global recommendation_service, voice_service, ar_service, websocket_manager, stripe_service, push_service
    
    # Startup
    logger.info("🚀 Starting Applebee's Interactive Menu System...")
    
    try:
        # Initialize database
        await init_db()
        logger.info("✅ Database initialized")
        
        # Initialize AI services
        recommendation_service = RecommendationService()
        voice_service = VoiceProcessingService()
        ar_service = ARExperienceService()
        logger.info("✅ AI services initialized")
        
        # Initialize real-time services
        websocket_manager = WebSocketManager()
        await websocket_manager.start()
        logger.info("✅ WebSocket manager started")
        
        # Initialize payment services
        stripe_service = StripeService()
        logger.info("✅ Payment services initialized")
        
        # Initialize notification services
        push_service = PushNotificationService()
        logger.info("✅ Notification services initialized")
        
        # Start background tasks
        asyncio.create_task(websocket_manager.broadcast_heartbeat())
        logger.info("✅ Background tasks started")
        
        logger.info("🎉 Applebee's Interactive Menu System started successfully!")
        
    except Exception as e:
        logger.error(f"❌ Failed to start application: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Applebee's Interactive Menu System...")
    
    try:
        # Cleanup services
        if websocket_manager:
            await websocket_manager.stop()
        logger.info("✅ Services cleaned up")
        
    except Exception as e:
        logger.error(f"❌ Error during shutdown: {e}")


# Create FastAPI application
app = FastAPI(
    title="Applebee's Interactive Menu API",
    description="Advanced interactive menu system with AI, AR, and real-time features",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_HOSTS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.ALLOWED_HOSTS
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Include API routes
app.include_router(api_router, prefix="/api/v1")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with system information."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Applebee's Interactive Menu API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: linear-gradient(135deg, #D2232A, #008000); color: white; }
            .container { max-width: 800px; margin: 0 auto; }
            .header { text-align: center; margin-bottom: 40px; }
            .feature { background: rgba(255,255,255,0.1); padding: 20px; margin: 10px 0; border-radius: 10px; }
            .status { display: inline-block; padding: 5px 10px; border-radius: 5px; margin: 5px; }
            .online { background: #4CAF50; }
            .offline { background: #f44336; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🍎 Applebee's Interactive Menu API</h1>
                <p>Advanced restaurant menu system with AI, AR, and real-time features</p>
            </div>
            
            <div class="feature">
                <h2>🤖 AI Features</h2>
                <div class="status online">Recommendation Engine</div>
                <div class="status online">Voice Processing</div>
                <div class="status online">Dietary Assistant</div>
                <div class="status online">Smart Upselling</div>
            </div>
            
            <div class="feature">
                <h2>🥽 AR/VR Features</h2>
                <div class="status online">3D Food Visualization</div>
                <div class="status online">AR Menu Navigation</div>
                <div class="status online">Virtual Food Tours</div>
                <div class="status online">Interactive Overlays</div>
            </div>
            
            <div class="feature">
                <h2>📱 Real-Time Features</h2>
                <div class="status online">Live Order Tracking</div>
                <div class="status online">WebSocket Communication</div>
                <div class="status online">Kitchen Integration</div>
                <div class="status online">Wait Time Predictions</div>
            </div>
            
            <div class="feature">
                <h2>🔗 API Endpoints</h2>
                <p><a href="/docs" style="color: white;">📚 Interactive API Documentation</a></p>
                <p><a href="/redoc" style="color: white;">📖 ReDoc Documentation</a></p>
                <p><a href="/api/v1/health" style="color: white;">🏥 Health Check</a></p>
            </div>
        </div>
    </body>
    </html>
    """


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Applebee's Interactive Menu API",
        "version": "2.0.0",
        "features": {
            "ai_recommendations": recommendation_service is not None,
            "voice_processing": voice_service is not None,
            "ar_experiences": ar_service is not None,
            "websocket_manager": websocket_manager is not None,
            "payment_processing": stripe_service is not None,
            "push_notifications": push_service is not None
        }
    }


@app.get("/api/v1/health")
async def api_health_check():
    """API health check with detailed status."""
    try:
        # Check database connection
        db = get_db()
        await db.execute("SELECT 1")
        db_status = "healthy"
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        db_status = "unhealthy"
    
    # Check AI services
    ai_status = {
        "recommendation_service": recommendation_service is not None,
        "voice_service": voice_service is not None,
        "ar_service": ar_service is not None
    }
    
    # Check real-time services
    realtime_status = {
        "websocket_manager": websocket_manager is not None and websocket_manager.is_running,
        "active_connections": len(websocket_manager.active_connections) if websocket_manager else 0
    }
    
    return {
        "status": "healthy" if db_status == "healthy" else "degraded",
        "timestamp": "2024-01-15T12:00:00Z",
        "services": {
            "database": db_status,
            "ai_services": ai_status,
            "realtime_services": realtime_status,
            "payment_services": stripe_service is not None,
            "notification_services": push_service is not None
        },
        "system_info": {
            "python_version": "3.11.0",
            "fastapi_version": "0.104.0",
            "uvicorn_version": "0.24.0"
        }
    }


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Global HTTP exception handler."""
    logger.error(f"HTTP Exception: {exc.status_code} - {exc.detail}")
    return {
        "error": {
            "code": exc.status_code,
            "message": exc.detail,
            "timestamp": "2024-01-15T12:00:00Z"
        }
    }


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled Exception: {exc}")
    return {
        "error": {
            "code": 500,
            "message": "Internal server error",
            "timestamp": "2024-01-15T12:00:00Z"
        }
    }


# WebSocket endpoint for real-time communication
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket, client_id: str):
    """WebSocket endpoint for real-time communication."""
    if websocket_manager:
        await websocket_manager.connect(websocket, client_id)
        try:
            while True:
                # Receive message from client
                data = await websocket.receive_text()
                
                # Process message
                response = await websocket_manager.process_message(client_id, data)
                
                # Send response back to client
                await websocket.send_text(response)
                
        except Exception as e:
            logger.error(f"WebSocket error for client {client_id}: {e}")
        finally:
            await websocket_manager.disconnect(client_id)


# Background task endpoints
@app.post("/api/v1/tasks/process-voice")
async def process_voice_task(audio_data: bytes):
    """Process voice input asynchronously."""
    if voice_service:
        result = await voice_service.process_audio_async(audio_data)
        return {"status": "success", "result": result}
    else:
        raise HTTPException(status_code=503, detail="Voice service not available")


@app.post("/api/v1/tasks/generate-recommendations")
async def generate_recommendations_task(user_id: str, context: Dict[str, Any]):
    """Generate AI recommendations asynchronously."""
    if recommendation_service:
        recommendations = await recommendation_service.generate_recommendations_async(user_id, context)
        return {"status": "success", "recommendations": recommendations}
    else:
        raise HTTPException(status_code=503, detail="Recommendation service not available")


@app.post("/api/v1/tasks/ar-experience")
async def start_ar_experience_task(experience_data: Dict[str, Any]):
    """Start AR experience asynchronously."""
    if ar_service:
        experience = await ar_service.start_experience_async(experience_data)
        return {"status": "success", "experience": experience}
    else:
        raise HTTPException(status_code=503, detail="AR service not available")


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    ) 