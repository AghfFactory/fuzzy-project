"""
FastAPI WebSocket server for real-time speaker recognition.
Handles audio streaming and returns fuzzy logic-based speaker detection results.
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
import json
import base64
import logging
from typing import Dict, List
import asyncio
import io
import librosa
import tempfile
import os

from database import Database, UserProfile
from feature_extractor import FeatureExtractor
from fuzzy_engine import FuzzyEngine
from audio_converter import AudioConverter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Fuzzy Speaker Recognition API")

# CORS middleware for React Native app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Log validation errors for debugging."""
    body = await request.body()
    logger.error(f"Validation error: {exc.errors()}")
    logger.error(f"Request URL: {request.url}")
    logger.error(f"Request method: {request.method}")
    logger.error(f"Request headers: {dict(request.headers)}")
    logger.error(f"Request body (first 500 chars): {body[:500] if len(body) > 500 else body}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body_preview": str(body[:500]) if len(body) > 500 else str(body)}
    )

# Initialize components
db = Database()
feature_extractor = FeatureExtractor()
fuzzy_engine = FuzzyEngine()
audio_converter = AudioConverter(target_sample_rate=16000, target_channels=1)

# Audio buffer for each WebSocket connection
audio_buffers: Dict[str, List[bytes]] = {}
BUFFER_CHUNK_SIZE = 2  # Process every 2 seconds of audio


# Pydantic models for request/response
class RegisterRequest(BaseModel):
    name: str
    audio: str  # Base64 encoded audio data


@app.on_event("startup")
async def startup_event():
    """Initialize database and load fuzzy models."""
    db.initialize()
    fuzzy_engine.initialize(db)
    logger.info("Backend initialized successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    db.close()
    logger.info("Backend shutdown complete")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "online", "service": "fuzzy-speaker-recognition"}


@app.get("/users")
async def get_users():
    """Get all registered users."""
    users = db.get_all_users()
    return {"users": [{"id": u.id, "name": u.name} for u in users]}


@app.get("/users/{user_id}/profile")
async def get_user_profile(user_id: int):
    """Get detailed profile for a specific user including feature values."""
    user = db.get_user_profile(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return {
        "id": user.id,
        "name": user.name,
        "mfcc_mean": user.mfcc_mean,
        "spectral_centroid": user.spectral_centroid,
        "zero_crossing_rate": user.zero_crossing_rate
    }


@app.get("/users/all/profiles")
async def get_all_profiles():
    """Get all user profiles with their feature values for debugging."""
    users = db.get_all_users()
    return {
        "users": [
            {
                "id": u.id,
                "name": u.name,
                "mfcc_mean": u.mfcc_mean,
                "spectral_centroid": u.spectral_centroid,
                "zero_crossing_rate": u.zero_crossing_rate
            }
            for u in users
        ],
        "total": len(users)
    }


@app.post("/register")
async def register_user(request: RegisterRequest):
    """
    Register a new user with a 1-minute voice sample.
    
    Args:
        request: RegisterRequest containing name and base64-encoded audio data
    """
    try:
        # Decode base64 audio
        audio_bytes = base64.b64decode(request.audio)
        
        # Convert audio to WAV format using audio converter
        # This handles M4A, 3GP, and other formats automatically
        try:
            audio_array, sample_rate = audio_converter.convert_to_wav(audio_bytes)
            logger.info(f"Successfully converted audio: {len(audio_array)} samples at {sample_rate}Hz")
        except ValueError as e:
            logger.error(f"Audio conversion failed: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail=f"Failed to process audio file: {str(e)}"
            )
        
        # Extract features from the full audio sample
        features = feature_extractor.extract_features(audio_array, sample_rate=sample_rate)
        
        if features is None:
            raise HTTPException(status_code=400, detail="Failed to extract features")
        
        # Store user profile in database
        user_id = db.create_user_profile(
            name=request.name,
            mfcc_mean=float(features['mfcc_mean']),
            spectral_centroid=float(features['spectral_centroid']),
            zero_crossing_rate=float(features['zero_crossing_rate'])
        )
        
        # Update fuzzy engine with new user
        fuzzy_engine.add_user_profile(user_id, request.name, features)
        
        logger.info(f"User registered: {request.name} (ID: {user_id})")
        return {"success": True, "user_id": user_id, "name": request.name, "message": "Registration successful"}
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")


@app.websocket("/ws/recognize")
async def websocket_recognize(websocket: WebSocket):
    """
    WebSocket endpoint for real-time audio streaming and speaker recognition.
    """
    await websocket.accept()
    connection_id = id(websocket)
    audio_buffers[connection_id] = []
    buffer_samples = []
    sample_rate = 16000
    
    logger.info(f"WebSocket connection established: {connection_id}")
    
    try:
        while True:
            # Receive JSON message with base64-encoded audio
            message = await websocket.receive_json()
            
            # Extract base64 audio from message
            if "audio" not in message:
                logger.warning("Message missing 'audio' field")
                continue
            
            base64_audio = message["audio"]
            
            # Decode base64 to get audio bytes
            try:
                audio_bytes = base64.b64decode(base64_audio)
            except Exception as e:
                logger.warning(f"Failed to decode base64 audio: {e}")
                continue
            
            audio_buffers[connection_id].append(audio_bytes)
            
            # Convert audio chunk to WAV format using audio converter
            audio_chunk = None
            try:
                audio_chunk, chunk_sr = audio_converter.convert_to_wav(audio_bytes)
                sample_rate = chunk_sr
            except ValueError as e:
                logger.warning(f"Failed to convert audio chunk: {e}")
                continue
            except Exception as e:
                logger.warning(f"Failed to process audio chunk: {type(e).__name__}: {e}")
                continue
            
            if audio_chunk is not None and len(audio_chunk) > 0:
                buffer_samples.extend(audio_chunk.tolist())
            
            # Process when we have enough audio (2 seconds at 16kHz = 32000 samples)
            chunk_size_samples = sample_rate * BUFFER_CHUNK_SIZE
            
            if len(buffer_samples) >= chunk_size_samples:
                # Extract the chunk to process
                process_chunk = np.array(buffer_samples[:chunk_size_samples])
                buffer_samples = buffer_samples[chunk_size_samples:]
                
                logger.debug(f"Processing audio chunk: {len(process_chunk)} samples")
                
                # Extract features
                features = feature_extractor.extract_features(process_chunk, sample_rate=sample_rate)
                
                if features:
                    logger.debug(f"Extracted features: MFCC={features['mfcc_mean']:.2f}, "
                               f"Centroid={features['spectral_centroid']:.2f}, "
                               f"ZCR={features['zero_crossing_rate']:.4f}")
                
                if features is not None:
                    # Run fuzzy inference
                    result = fuzzy_engine.recognize_speaker(features)
                    
                    # Send result back to client
                    response = {
                        "speaker": result["speaker"],
                        "confidence": round(result["confidence"], 2)
                    }
                    await websocket.send_json(response)
                    logger.info(f"Recognition result: {response['speaker']} (confidence: {response['confidence']:.2f})")
                else:
                    logger.warning("Failed to extract features from audio chunk")
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {connection_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        try:
            await websocket.send_json({"error": str(e)})
        except:
            pass
    finally:
        # Cleanup
        if connection_id in audio_buffers:
            del audio_buffers[connection_id]
        logger.info(f"Connection cleaned up: {connection_id}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

