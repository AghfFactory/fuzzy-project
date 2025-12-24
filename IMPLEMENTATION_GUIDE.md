# Fuzzy Logic-Based Speaker Recognition System - Implementation Guide

## Table of Contents
1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Prerequisites and Setup](#prerequisites-and-setup)
4. [Detailed Implementation Steps](#detailed-implementation-steps)
5. [Component Deep Dive](#component-deep-dive)
6. [API Documentation](#api-documentation)
7. [Testing and Validation](#testing-and-validation)
8. [Deployment Guide](#deployment-guide)
9. [Troubleshooting](#troubleshooting)

---

## Project Overview

### What is This Project?
This is a **real-time speaker recognition system** that uses **fuzzy logic** to identify speakers based on their voice characteristics. The system:

- Accepts audio recordings from mobile devices (React Native app)
- Extracts acoustic features from audio (MFCC, spectral centroid, zero-crossing rate)
- Uses fuzzy logic inference to match speakers against registered profiles
- Provides real-time recognition via WebSocket connections
- Stores speaker profiles in a SQLite database

### Key Technologies
- **Backend Framework**: FastAPI (Python)
- **Audio Processing**: librosa, pydub
- **Fuzzy Logic**: scikit-fuzzy
- **Database**: SQLite
- **Real-time Communication**: WebSocket
- **Audio Formats**: Supports M4A, 3GP, WAV, MP3, OGG

---

## System Architecture

### High-Level Architecture

```
┌─────────────────┐
│  React Native   │
│     Mobile App   │
└────────┬────────┘
         │
         │ HTTP/WebSocket
         │
┌────────▼─────────────────────────────────────┐
│         FastAPI Backend Server               │
│  ┌────────────────────────────────────────┐ │
│  │  main.py (API Endpoints)                │ │
│  │  - /register (POST)                     │ │
│  │  - /ws/recognize (WebSocket)            │ │
│  │  - /users (GET)                         │ │
│  └────────────────────────────────────────┘ │
│                                              │
│  ┌────────────────────────────────────────┐ │
│  │  audio_converter.py                    │ │
│  │  - Format detection                    │ │
│  │  - Audio conversion (M4A/3GP → WAV)   │ │
│  └────────────────────────────────────────┘ │
│                                              │
│  ┌────────────────────────────────────────┐ │
│  │  feature_extractor.py                   │ │
│  │  - MFCC extraction                     │ │
│  │  - Spectral centroid                    │ │
│  │  - Zero-crossing rate                  │ │
│  └────────────────────────────────────────┘ │
│                                              │
│  ┌────────────────────────────────────────┐ │
│  │  fuzzy_engine.py                       │ │
│  │  - Membership functions                │ │
│  │  - Fuzzy rules                         │ │
│  │  - Mamdani inference                   │ │
│  └────────────────────────────────────────┘ │
│                                              │
│  ┌────────────────────────────────────────┐ │
│  │  database.py                           │ │
│  │  - SQLite operations                   │ │
│  │  - User profile management             │ │
│  └────────────────────────────────────────┘ │
└──────────────────────────────────────────────┘
         │
         │
┌────────▼────────┐
│  SQLite Database │
│ speaker_profiles │
│      .db         │
└──────────────────┘
```

### Data Flow

#### Registration Flow
1. Client sends base64-encoded audio + name → `/register`
2. Audio converter processes audio → WAV format
3. Feature extractor extracts MFCC, spectral centroid, ZCR
4. Database stores user profile
5. Fuzzy engine creates membership functions for new user
6. Response: `{success: true, user_id: X}`

#### Recognition Flow (WebSocket)
1. Client connects to `/ws/recognize`
2. Client streams base64-encoded audio chunks
3. Server buffers 2 seconds of audio
4. Audio converter processes chunk
5. Feature extractor extracts features
6. Fuzzy engine runs inference against all users
7. Server sends `{speaker: "Name", confidence: 0.85}`

---

## Prerequisites and Setup

### Step 1: Install Python and Dependencies

#### 1.1 Python Installation
- **Required Version**: Python 3.8 or higher
- **Download**: https://www.python.org/downloads/
- **Verify Installation**:
  ```bash
  python --version
  # Should show Python 3.8+
  ```

#### 1.2 Create Virtual Environment
```bash
# Navigate to project directory
cd C:\Users\aryan\AndroidStudioProjects\backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1
# On Windows CMD:
venv\Scripts\activate.bat
# On Linux/Mac:
source venv/bin/activate
```

#### 1.3 Install Python Packages
```bash
pip install -r requirements.txt
```

**Expected Output**: All packages from `requirements.txt` installed:
- fastapi==0.104.1
- uvicorn[standard]==0.24.0
- websockets==12.0
- numpy>=1.26.0,<2.1.0
- librosa==0.10.1
- scikit-fuzzy==0.4.2
- python-multipart==0.0.6
- soundfile==0.12.1
- pydub==0.25.1

### Step 2: Install FFmpeg (Required for Audio Conversion)

FFmpeg is required by `pydub` to convert M4A and 3GP formats.

#### 2.1 Windows Installation
1. **Download FFmpeg**:
   - Visit: https://ffmpeg.org/download.html
   - Or use direct link: https://www.gyan.dev/ffmpeg/builds/
   - Download "ffmpeg-release-essentials.zip"

2. **Extract and Install**:
   ```powershell
   # Extract to C:\ffmpeg
   # Add to PATH:
   # - Right-click "This PC" → Properties → Advanced System Settings
   # - Environment Variables → System Variables → Path → Edit
   # - Add: C:\ffmpeg\bin
   ```

3. **Verify Installation**:
   ```bash
   ffmpeg -version
   # Should show version information
   ```

#### 2.2 Alternative: Using Chocolatey (Windows)
```powershell
choco install ffmpeg
```

#### 2.3 Linux Installation
```bash
sudo apt-get update
sudo apt-get install ffmpeg
```

#### 2.4 macOS Installation
```bash
brew install ffmpeg
```

### Step 3: Verify Project Structure

Ensure your project has these files:
```
backend/
├── __init__.py
├── main.py
├── database.py
├── audio_converter.py
├── feature_extractor.py
├── fuzzy_engine.py
├── requirements.txt
└── venv/
```

---

## Detailed Implementation Steps

### Phase 1: Database Setup

#### Step 1.1: Initialize Database Module (`database.py`)

**Purpose**: Create SQLite database to store speaker profiles.

**Implementation Details**:

1. **Import Required Libraries**:
   ```python
   import sqlite3
   from typing import List, Optional
   from dataclasses import dataclass
   import logging
   ```

2. **Create UserProfile Dataclass**:
   - Stores: `id`, `name`, `mfcc_mean`, `spectral_centroid`, `zero_crossing_rate`
   - Used for type-safe data handling

3. **Database Class Structure**:
   - `__init__()`: Sets database path (default: `speaker_profiles.db`)
   - `initialize()`: Creates connection and tables
   - `_create_tables()`: Creates `users` table with schema:
     ```sql
     CREATE TABLE users (
         id INTEGER PRIMARY KEY AUTOINCREMENT,
         name TEXT NOT NULL UNIQUE,
         mfcc_mean REAL NOT NULL,
         spectral_centroid REAL NOT NULL,
         zero_crossing_rate REAL NOT NULL,
         created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
     )
     ```

4. **Key Methods**:
   - `create_user_profile()`: Insert/update user
   - `get_user_profile(user_id)`: Retrieve by ID
   - `get_user_by_name(name)`: Retrieve by name
   - `get_all_users()`: Get all profiles
   - `delete_user(user_id)`: Remove user
   - `close()`: Close connection

**Testing**:
```python
from database import Database

db = Database()
db.initialize()
user_id = db.create_user_profile(
    name="John Doe",
    mfcc_mean=-200.5,
    spectral_centroid=2500.0,
    zero_crossing_rate=0.15
)
print(f"Created user ID: {user_id}")
```

---

### Phase 2: Audio Processing

#### Step 2.1: Audio Converter Module (`audio_converter.py`)

**Purpose**: Convert various audio formats (M4A, 3GP, etc.) to standardized WAV format.

**Implementation Details**:

1. **Format Detection** (`detect_format()`):
   - Reads file signature (first 4-20 bytes)
   - Checks for:
     - M4A/MP4: `ftyp` + `m4a`/`MP4`/`isom`
     - WAV: `RIFF` + `WAVE`
     - 3GP: `ftyp` at offset 4
     - OGG: `OggS`
   - Returns file extension

2. **Audio Conversion** (`convert_to_wav()`):
   - **Input**: Raw audio bytes (any format)
   - **Output**: `(numpy_array, sample_rate)`
   - **Process**:
     ```
     1. Detect format from bytes
     2. Write bytes to temporary file
     3. Try librosa.load() first (works for WAV, MP3)
     4. If fails, use pydub + ffmpeg (for M4A, 3GP)
        a. Load with AudioSegment.from_file()
        b. Convert to mono (1 channel)
        c. Resample to 16000 Hz
        d. Export to temporary WAV
        e. Load WAV with librosa
     5. Clean up temporary files
     ```

3. **FFmpeg Detection**:
   - Tries multiple methods:
     - `pydub.utils.which("ffmpeg")`
     - `subprocess.run(["where", "ffmpeg"])` (Windows)
     - Common Windows paths (`C:\ffmpeg\bin\ffmpeg.exe`)
     - Direct execution test

4. **Error Handling**:
   - Catches `ImportError` (pydub not installed)
   - Catches conversion errors
   - Provides helpful error messages
   - Retries file deletion (Windows file locking)

**Testing**:
```python
from audio_converter import AudioConverter

converter = AudioConverter(target_sample_rate=16000, target_channels=1)

# Read audio file
with open("test.m4a", "rb") as f:
    audio_bytes = f.read()

# Convert
audio_array, sample_rate = converter.convert_to_wav(audio_bytes)
print(f"Converted: {len(audio_array)} samples at {sample_rate}Hz")
```

---

#### Step 2.2: Feature Extractor Module (`feature_extractor.py`)

**Purpose**: Extract acoustic features from audio for speaker recognition.

**Implementation Details**:

1. **MFCC (Mel-Frequency Cepstral Coefficients)**:
   - **What**: Represents spectral envelope (vocal tract shape)
   - **Extraction**:
     ```python
     mfccs = librosa.feature.mfcc(
         y=audio,
         sr=16000,
         n_mfcc=13,      # 13 coefficients
         n_fft=2048,     # FFT window size
         hop_length=512  # Frame hop length
     )
     mfcc_mean = np.mean(mfccs[0])  # Use first coefficient
     ```
   - **Why**: Captures speaker-specific vocal characteristics

2. **Spectral Centroid**:
   - **What**: "Brightness" of sound (center of mass of spectrum)
   - **Extraction**:
     ```python
     spectral_centroids = librosa.feature.spectral_centroid(
         y=audio,
         sr=16000,
         n_fft=2048,
         hop_length=512
     )
     spectral_centroid = np.mean(spectral_centroids)
     ```
   - **Why**: Different speakers have different voice brightness

3. **Zero-Crossing Rate (ZCR)**:
   - **What**: Rate of sign changes in audio signal
   - **Extraction**:
     ```python
     zcr = librosa.feature.zero_crossing_rate(
         audio,
         frame_length=2048,
         hop_length=512
     )
     zero_crossing_rate = np.mean(zcr)
     ```
   - **Why**: Indicates voice quality and pitch characteristics

4. **Resampling**:
   - If input sample rate ≠ 16000 Hz, resample using `librosa.resample()`
   - Ensures consistent feature extraction

5. **Output Format**:
   ```python
   {
       'mfcc_mean': float,
       'spectral_centroid': float,
       'zero_crossing_rate': float
   }
   ```

**Testing**:
```python
from feature_extractor import FeatureExtractor
import numpy as np

extractor = FeatureExtractor()
audio = np.random.randn(16000 * 2)  # 2 seconds of audio
features = extractor.extract_features(audio, sample_rate=16000)
print(features)
```

---

### Phase 3: Fuzzy Logic Engine

#### Step 3.1: Fuzzy Engine Module (`fuzzy_engine.py`)

**Purpose**: Use fuzzy logic to match audio features to speaker profiles.

**Fuzzy Logic Concepts**:

1. **Membership Functions**: Define how "close" a feature value is to a user's profile
2. **Fuzzy Rules**: If-then rules for inference
3. **Mamdani Inference**: Defuzzification to get confidence score

**Implementation Details**:

1. **Input Variables** (Antecedents):
   - `mfcc_mean`: Range [-500, 500]
   - `spectral_centroid`: Range [0, 8000]
   - `zero_crossing_rate`: Range [0, 0.5]

2. **Output Variable** (Consequent):
   - `confidence`: Range [0, 1]

3. **Membership Functions** (Per User):
   For each user, create triangular membership functions centered on their profile values:

   **MFCC Membership**:
   ```python
   mfcc_input['low'] = trimf([user.mfcc_mean - 200, user.mfcc_mean - 100, user.mfcc_mean])
   mfcc_input['medium'] = trimf([user.mfcc_mean - 100, user.mfcc_mean, user.mfcc_mean + 100])
   mfcc_input['high'] = trimf([user.mfcc_mean, user.mfcc_mean + 100, user.mfcc_mean + 200])
   ```

   **Spectral Centroid Membership**:
   ```python
   centroid_input['low'] = trimf([centroid - 2000, centroid - 1000, centroid])
   centroid_input['medium'] = trimf([centroid - 1000, centroid, centroid + 1000])
   centroid_input['high'] = trimf([centroid, centroid + 1000, centroid + 2000])
   ```

   **ZCR Membership**:
   ```python
   zcr_input['low'] = trimf([zcr - 0.1, zcr - 0.05, zcr])
   zcr_input['medium'] = trimf([zcr - 0.05, zcr, zcr + 0.05])
   zcr_input['high'] = trimf([zcr, zcr + 0.05, zcr + 0.1])
   ```

4. **Fuzzy Rules**:
   ```python
   # Rule 1: All features match → High confidence
   IF mfcc IS medium AND centroid IS medium AND zcr IS medium
   THEN confidence IS high

   # Rule 2: Some features match → Medium confidence
   IF (features partially match) AND NOT (all match)
   THEN confidence IS medium

   # Rule 3: Features don't match → Low confidence
   IF features don't match
   THEN confidence IS low
   ```

5. **Inference Process** (`recognize_speaker()`):
   ```
   For each registered user:
     1. Set input values (mfcc_mean, spectral_centroid, zcr)
     2. Run fuzzy inference (simulator.compute())
     3. Get output confidence [0, 1]
     4. Track best match (highest confidence)
   
   If best_confidence >= 0.5:
      Return {speaker: best_match, confidence: best_confidence}
   Else:
      Return {speaker: "Unknown", confidence: best_confidence}
   ```

6. **System Initialization**:
   - Load all users from database on startup
   - Create fuzzy system for each user
   - Store simulators in dictionary

**Testing**:
```python
from fuzzy_engine import FuzzyEngine
from database import Database

db = Database()
db.initialize()

engine = FuzzyEngine()
engine.initialize(db)

# Test recognition
features = {
    'mfcc_mean': -200.0,
    'spectral_centroid': 2500.0,
    'zero_crossing_rate': 0.15
}

result = engine.recognize_speaker(features)
print(f"Speaker: {result['speaker']}, Confidence: {result['confidence']}")
```

---

### Phase 4: FastAPI Server

#### Step 4.1: Main Application (`main.py`)

**Purpose**: HTTP and WebSocket API endpoints for client communication.

**Implementation Details**:

1. **FastAPI Setup**:
   ```python
   app = FastAPI(title="Fuzzy Speaker Recognition API")
   
   # CORS middleware (for React Native)
   app.add_middleware(
       CORSMiddleware,
       allow_origins=["*"],
       allow_credentials=True,
       allow_methods=["*"],
       allow_headers=["*"],
   )
   ```

2. **Component Initialization**:
   ```python
   db = Database()
   feature_extractor = FeatureExtractor()
   fuzzy_engine = FuzzyEngine()
   audio_converter = AudioConverter(target_sample_rate=16000, target_channels=1)
   ```

3. **Startup Event**:
   ```python
   @app.on_event("startup")
   async def startup_event():
       db.initialize()
       fuzzy_engine.initialize(db)
   ```

4. **HTTP Endpoints**:

   **GET `/`** - Health Check:
   ```python
   @app.get("/")
   async def root():
       return {"status": "online", "service": "fuzzy-speaker-recognition"}
   ```

   **GET `/users`** - List All Users:
   ```python
   @app.get("/users")
   async def get_users():
       users = db.get_all_users()
       return {"users": [{"id": u.id, "name": u.name} for u in users]}
   ```

   **GET `/users/{user_id}/profile`** - Get User Profile:
   ```python
   @app.get("/users/{user_id}/profile")
   async def get_user_profile(user_id: int):
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
   ```

   **POST `/register`** - Register New Speaker:
   ```python
   @app.post("/register")
   async def register_user(request: RegisterRequest):
       # 1. Decode base64 audio
       audio_bytes = base64.b64decode(request.audio)
       
       # 2. Convert to WAV
       audio_array, sample_rate = audio_converter.convert_to_wav(audio_bytes)
       
       # 3. Extract features
       features = feature_extractor.extract_features(audio_array, sample_rate)
       
       # 4. Save to database
       user_id = db.create_user_profile(
           name=request.name,
           mfcc_mean=features['mfcc_mean'],
           spectral_centroid=features['spectral_centroid'],
           zero_crossing_rate=features['zero_crossing_rate']
       )
       
       # 5. Update fuzzy engine
       fuzzy_engine.add_user_profile(user_id, request.name, features)
       
       return {"success": True, "user_id": user_id, "name": request.name}
   ```

5. **WebSocket Endpoint** (`/ws/recognize`):

   **Connection Flow**:
   ```
   1. Client connects → websocket.accept()
   2. Create connection_id and audio buffer
   3. Enter receive loop:
      a. Receive JSON message: {"audio": "base64_string"}
      b. Decode base64 to bytes
      c. Convert audio chunk to WAV
      d. Buffer samples
      e. When buffer >= 2 seconds:
         - Extract features
         - Run fuzzy inference
         - Send result: {"speaker": "...", "confidence": 0.85}
   4. On disconnect → cleanup buffer
   ```

   **Implementation**:
   ```python
   @app.websocket("/ws/recognize")
   async def websocket_recognize(websocket: WebSocket):
       await websocket.accept()
       connection_id = id(websocket)
       audio_buffers[connection_id] = []
       buffer_samples = []
       sample_rate = 16000
       
       try:
           while True:
               message = await websocket.receive_json()
               base64_audio = message["audio"]
               
               # Decode and convert
               audio_bytes = base64.b64decode(base64_audio)
               audio_chunk, chunk_sr = audio_converter.convert_to_wav(audio_bytes)
               buffer_samples.extend(audio_chunk.tolist())
               
               # Process when buffer is full (2 seconds)
               chunk_size_samples = sample_rate * 2
               if len(buffer_samples) >= chunk_size_samples:
                   process_chunk = np.array(buffer_samples[:chunk_size_samples])
                   buffer_samples = buffer_samples[chunk_size_samples:]
                   
                   # Extract features
                   features = feature_extractor.extract_features(process_chunk, sample_rate)
                   
                   # Recognize
                   result = fuzzy_engine.recognize_speaker(features)
                   
                   # Send result
                   await websocket.send_json({
                       "speaker": result["speaker"],
                       "confidence": round(result["confidence"], 2)
                   })
                   
       except WebSocketDisconnect:
           # Cleanup
           if connection_id in audio_buffers:
               del audio_buffers[connection_id]
   ```

6. **Error Handling**:
   - Validation errors → 422 with details
   - Audio conversion errors → 400 with message
   - Database errors → 500 with error
   - WebSocket errors → Logged and sent to client

---

## Component Deep Dive

### Database Schema

**Table: `users`**
| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | INTEGER | PRIMARY KEY, AUTOINCREMENT | Unique user ID |
| name | TEXT | NOT NULL, UNIQUE | Speaker name |
| mfcc_mean | REAL | NOT NULL | Mean MFCC coefficient |
| spectral_centroid | REAL | NOT NULL | Spectral centroid value |
| zero_crossing_rate | REAL | NOT NULL | Zero-crossing rate |
| created_at | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | Registration time |

### Audio Processing Pipeline

```
Raw Audio Bytes (M4A/3GP/WAV)
    ↓
Format Detection (file signature)
    ↓
Temporary File Write
    ↓
librosa/pydub Conversion
    ↓
WAV Format (16kHz, Mono)
    ↓
numpy array (normalized [-1, 1])
    ↓
Feature Extraction
    ↓
Features Dictionary
```

### Fuzzy Logic Inference Flow

```
Input Features (mfcc, centroid, zcr)
    ↓
For Each User:
    ↓
Fuzzification (Membership Functions)
    ↓
Rule Evaluation (IF-THEN)
    ↓
Aggregation (OR/AND operations)
    ↓
Defuzzification (Centroid method)
    ↓
Confidence Score [0, 1]
    ↓
Select Best Match (highest confidence)
    ↓
Output: {speaker, confidence}
```

---

## API Documentation

### Base URL
```
http://localhost:8000
```

### Endpoints

#### 1. Health Check
```http
GET /
```

**Response**:
```json
{
  "status": "online",
  "service": "fuzzy-speaker-recognition"
}
```

---

#### 2. Register Speaker
```http
POST /register
Content-Type: application/json
```

**Request Body**:
```json
{
  "name": "John Doe",
  "audio": "base64_encoded_audio_string"
}
```

**Response** (Success):
```json
{
  "success": true,
  "user_id": 1,
  "name": "John Doe",
  "message": "Registration successful"
}
```

**Response** (Error):
```json
{
  "detail": "Failed to process audio file: ..."
}
```

**Requirements**:
- Audio should be ~1 minute long for best results
- Supported formats: M4A, 3GP, WAV, MP3, OGG
- Audio is base64-encoded in request

---

#### 3. List All Users
```http
GET /users
```

**Response**:
```json
{
  "users": [
    {"id": 1, "name": "John Doe"},
    {"id": 2, "name": "Jane Smith"}
  ]
}
```

---

#### 4. Get User Profile
```http
GET /users/{user_id}/profile
```

**Response**:
```json
{
  "id": 1,
  "name": "John Doe",
  "mfcc_mean": -200.5,
  "spectral_centroid": 2500.0,
  "zero_crossing_rate": 0.15
}
```

---

#### 5. Get All Profiles
```http
GET /users/all/profiles
```

**Response**:
```json
{
  "users": [
    {
      "id": 1,
      "name": "John Doe",
      "mfcc_mean": -200.5,
      "spectral_centroid": 2500.0,
      "zero_crossing_rate": 0.15
    }
  ],
  "total": 1
}
```

---

#### 6. Real-Time Recognition (WebSocket)
```http
WS /ws/recognize
```

**Connection**:
```javascript
const ws = new WebSocket("ws://localhost:8000/ws/recognize");
```

**Send Message** (JSON):
```json
{
  "audio": "base64_encoded_audio_chunk"
}
```

**Receive Message** (JSON):
```json
{
  "speaker": "John Doe",
  "confidence": 0.85
}
```

**Or**:
```json
{
  "speaker": "Unknown",
  "confidence": 0.35
}
```

**Notes**:
- Send audio chunks continuously (e.g., every 0.5 seconds)
- Server processes 2-second buffers
- Results sent after each 2-second chunk is processed
- Confidence threshold: 0.5 (below = "Unknown")

---

## Testing and Validation

### Step 1: Start the Server

```bash
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Run server
python main.py
# Or:
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**Expected Output**:
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Step 2: Test Health Check

```bash
curl http://localhost:8000/
```

**Expected Response**:
```json
{"status":"online","service":"fuzzy-speaker-recognition"}
```

### Step 3: Test Registration

**Using Python**:
```python
import requests
import base64

# Read audio file
with open("test_audio.m4a", "rb") as f:
    audio_bytes = f.read()

# Encode to base64
audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

# Register
response = requests.post(
    "http://localhost:8000/register",
    json={
        "name": "Test User",
        "audio": audio_base64
    }
)

print(response.json())
```

**Using curl** (with base64 file):
```bash
# Encode audio to base64
base64 -i test_audio.m4a -o audio_base64.txt

# Create JSON payload
echo '{"name":"Test User","audio":"' > payload.json
cat audio_base64.txt >> payload.json
echo '"}' >> payload.json

# Send request
curl -X POST http://localhost:8000/register \
  -H "Content-Type: application/json" \
  -d @payload.json
```

### Step 4: Test User List

```bash
curl http://localhost:8000/users
```

### Step 5: Test WebSocket Recognition

**Using Python**:
```python
import asyncio
import websockets
import json
import base64

async def test_websocket():
    uri = "ws://localhost:8000/ws/recognize"
    
    async with websockets.connect(uri) as websocket:
        # Read and send audio chunk
        with open("test_audio.m4a", "rb") as f:
            audio_bytes = f.read()
        
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        # Send audio
        await websocket.send(json.dumps({"audio": audio_base64}))
        
        # Wait for response
        response = await websocket.recv()
        print(json.loads(response))

asyncio.run(test_websocket())
```

**Using JavaScript (Browser)**:
```javascript
const ws = new WebSocket("ws://localhost:8000/ws/recognize");

ws.onopen = () => {
  console.log("Connected");
  
  // Send audio chunk (base64)
  ws.send(JSON.stringify({
    audio: "base64_audio_string_here"
  }));
};

ws.onmessage = (event) => {
  const result = JSON.parse(event.data);
  console.log("Speaker:", result.speaker);
  console.log("Confidence:", result.confidence);
};

ws.onerror = (error) => {
  console.error("WebSocket error:", error);
};
```

### Step 6: Verify Database

```bash
# Using sqlite3 command-line tool
sqlite3 speaker_profiles.db

# Query users
SELECT * FROM users;

# Exit
.quit
```

---

## Deployment Guide

### Development Deployment

**Run Locally**:
```bash
python main.py
# Server runs on http://localhost:8000
```

**With Auto-Reload**:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Production Deployment

#### Option 1: Using Uvicorn with Gunicorn Workers

```bash
# Install gunicorn
pip install gunicorn

# Run with multiple workers
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

#### Option 2: Using Docker

**Create `Dockerfile`**:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Run server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Build and Run**:
```bash
docker build -t speaker-recognition-api .
docker run -p 8000:8000 speaker-recognition-api
```

#### Option 3: Cloud Deployment (Heroku, AWS, etc.)

**For Heroku**:
1. Create `Procfile`:
   ```
   web: uvicorn main:app --host 0.0.0.0 --port $PORT
   ```

2. Create `runtime.txt`:
   ```
   python-3.11
   ```

3. Deploy:
   ```bash
   heroku create your-app-name
   git push heroku main
   ```

**Environment Variables**:
- `PORT`: Server port (auto-set by platform)
- `DATABASE_URL`: Optional (if using external DB)

---

## Troubleshooting

### Issue 1: FFmpeg Not Found

**Error**:
```
ValueError: ffmpeg not found. Please ensure ffmpeg is installed...
```

**Solution**:
1. Install FFmpeg (see Prerequisites)
2. Add to PATH
3. Verify: `ffmpeg -version`

### Issue 2: Audio Conversion Fails

**Error**:
```
Failed to process audio file: ...
```

**Solutions**:
- Ensure audio file is valid
- Check file format is supported
- Verify FFmpeg is installed
- Check audio file size (not empty)

### Issue 3: Database Locked

**Error**:
```
sqlite3.OperationalError: database is locked
```

**Solution**:
- Ensure only one instance of server is running
- Check for open database connections
- Restart server

### Issue 4: WebSocket Connection Fails

**Error**:
```
WebSocket connection failed
```

**Solutions**:
- Check server is running
- Verify URL: `ws://localhost:8000/ws/recognize`
- Check firewall settings
- Ensure CORS is configured

### Issue 5: Low Recognition Accuracy

**Solutions**:
- Use longer audio samples for registration (1+ minute)
- Ensure quiet environment during recording
- Register multiple samples per user
- Adjust fuzzy membership function ranges in `fuzzy_engine.py`
- Lower confidence threshold (currently 0.5)

### Issue 6: Import Errors

**Error**:
```
ModuleNotFoundError: No module named 'librosa'
```

**Solution**:
```bash
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Issue 7: Port Already in Use

**Error**:
```
Address already in use
```

**Solution**:
```bash
# Change port in main.py:
uvicorn.run(app, host="0.0.0.0", port=8001)

# Or kill existing process:
# Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

---

## Performance Optimization

### 1. Audio Buffer Size
- Current: 2 seconds
- Adjust `BUFFER_CHUNK_SIZE` in `main.py`
- Smaller = faster response, less accurate
- Larger = slower response, more accurate

### 2. Feature Extraction
- Consider caching features for registered users
- Use multiprocessing for parallel feature extraction

### 3. Database
- Add indexes on `name` column (already UNIQUE)
- Consider connection pooling for high traffic

### 4. Fuzzy Engine
- Cache fuzzy systems (already done)
- Optimize membership function ranges
- Consider reducing number of rules

---

## Future Enhancements

1. **Multiple Audio Samples per User**: Average features from multiple recordings
2. **Feature Normalization**: Normalize features across users for better comparison
3. **Machine Learning**: Replace fuzzy logic with neural network (e.g., LSTM)
4. **Voice Activity Detection**: Skip silence in audio processing
5. **Real-time Streaming**: Process audio in smaller chunks for lower latency
6. **User Management**: Add authentication and user sessions
7. **API Rate Limiting**: Prevent abuse
8. **Logging and Monitoring**: Add structured logging and metrics

---

## Conclusion

This implementation guide provides a complete walkthrough of the Fuzzy Logic-Based Speaker Recognition System. The system is designed to be:

- **Modular**: Each component is independent and testable
- **Extensible**: Easy to add new features or replace components
- **Robust**: Error handling and validation throughout
- **Real-time**: WebSocket support for live recognition

For questions or issues, refer to the Troubleshooting section or review the component source code.

---

**Last Updated**: 2024
**Version**: 1.0

