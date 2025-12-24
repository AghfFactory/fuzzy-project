# Fuzzy Logic-Based Speaker Recognition System

A real-time speaker recognition API built with FastAPI that uses fuzzy logic to identify speakers based on their voice characteristics. Perfect for integration with mobile apps (React Native) or web applications.

## Features

- 🎤 **Real-time Speaker Recognition** - WebSocket-based live audio streaming and recognition
- 🧠 **Fuzzy Logic Engine** - Uses scikit-fuzzy for intelligent speaker matching
- 🎵 **Multi-Format Audio Support** - Handles M4A, 3GP, WAV, MP3, OGG formats
- 📊 **Feature Extraction** - Extracts MFCC, spectral centroid, and zero-crossing rate
- 💾 **SQLite Database** - Lightweight storage for speaker profiles
- 🚀 **FastAPI Backend** - High-performance async API with automatic documentation

## Quick Start

### Prerequisites

- **Python 3.8+** - [Download Python](https://www.python.org/downloads/)
- **FFmpeg** - Required for audio format conversion
  - **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html) or use `choco install ffmpeg`
  - **Linux**: `sudo apt-get install ffmpeg`
  - **macOS**: `brew install ffmpeg`

### Installation

1. **Clone the repository** (or navigate to project directory):
   ```bash
   cd backend
   ```

2. **Create and activate virtual environment**:
   ```bash
   # Create virtual environment
   python -m venv venv

   # Activate (Windows PowerShell)
   .\venv\Scripts\Activate.ps1

   # Activate (Windows CMD)
   venv\Scripts\activate.bat

   # Activate (Linux/macOS)
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify FFmpeg installation**:
   ```bash
   ffmpeg -version
   ```

### Running the Server

```bash
# Make sure virtual environment is activated
python main.py
```

The server will start on `http://localhost:8000`

**Alternative**: Run with auto-reload for development:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### API Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Usage

### 1. Register a Speaker

Register a new speaker with a voice sample (recommended: 1 minute of audio).

**Python Example**:
```python
import requests
import base64

# Read audio file
with open("speaker_audio.m4a", "rb") as f:
    audio_bytes = f.read()

# Encode to base64
audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

# Register speaker
response = requests.post(
    "http://localhost:8000/register",
    json={
        "name": "John Doe",
        "audio": audio_base64
    }
)

print(response.json())
# {"success": true, "user_id": 1, "name": "John Doe", "message": "Registration successful"}
```

**cURL Example**:
```bash
# First, encode audio to base64
base64 -i speaker_audio.m4a > audio_base64.txt

# Create JSON payload
echo '{"name":"John Doe","audio":"' > payload.json
cat audio_base64.txt >> payload.json
echo '"}' >> payload.json

# Send request
curl -X POST http://localhost:8000/register \
  -H "Content-Type: application/json" \
  -d @payload.json
```

### 2. List Registered Users

```bash
curl http://localhost:8000/users
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

### 3. Real-Time Speaker Recognition (WebSocket)

Connect to the WebSocket endpoint and stream audio chunks for real-time recognition.

**Python Example**:
```python
import asyncio
import websockets
import json
import base64

async def recognize_speaker():
    uri = "ws://localhost:8000/ws/recognize"
    
    async with websockets.connect(uri) as websocket:
        # Read audio file
        with open("test_audio.m4a", "rb") as f:
            audio_bytes = f.read()
        
        # Encode to base64
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        # Send audio chunk
        await websocket.send(json.dumps({"audio": audio_base64}))
        
        # Receive recognition result
        response = await websocket.recv()
        result = json.loads(response)
        
        print(f"Speaker: {result['speaker']}")
        print(f"Confidence: {result['confidence']}")

asyncio.run(recognize_speaker())
```

**JavaScript Example**:
```javascript
const ws = new WebSocket("ws://localhost:8000/ws/recognize");

ws.onopen = () => {
  console.log("Connected to recognition server");
  
  // Convert audio file to base64 and send
  // (You'll need to implement file reading in your app)
  const audioBase64 = "your_base64_audio_string";
  
  ws.send(JSON.stringify({
    audio: audioBase64
  }));
};

ws.onmessage = (event) => {
  const result = JSON.parse(event.data);
  console.log(`Speaker: ${result.speaker}`);
  console.log(`Confidence: ${result.confidence}`);
  
  // Handle recognition result in your app
  if (result.confidence > 0.7) {
    console.log(`High confidence match: ${result.speaker}`);
  }
};

ws.onerror = (error) => {
  console.error("WebSocket error:", error);
};

ws.onclose = () => {
  console.log("Connection closed");
};
```

**React Native Example**:
```javascript
import { WebSocket } from 'react-native';

const ws = new WebSocket('ws://your-server-ip:8000/ws/recognize');

ws.onopen = () => {
  // Record audio and convert to base64
  // Then send chunks continuously
  const audioChunk = base64AudioString;
  ws.send(JSON.stringify({ audio: audioChunk }));
};

ws.onmessage = (event) => {
  const result = JSON.parse(event.data);
  console.log(`Recognized: ${result.speaker} (${result.confidence})`);
};
```

### 4. Get User Profile

Retrieve detailed feature information for a registered user:

```bash
curl http://localhost:8000/users/1/profile
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

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `POST` | `/register` | Register a new speaker |
| `GET` | `/users` | List all registered users |
| `GET` | `/users/{user_id}/profile` | Get user profile details |
| `GET` | `/users/all/profiles` | Get all user profiles with features |
| `WS` | `/ws/recognize` | WebSocket for real-time recognition |

## How It Works

1. **Registration**: 
   - User provides name and audio sample
   - Audio is converted to WAV format (16kHz, mono)
   - Features are extracted (MFCC, spectral centroid, zero-crossing rate)
   - Profile is stored in database
   - Fuzzy logic membership functions are created

2. **Recognition**:
   - Audio chunks are streamed via WebSocket
   - Server buffers 2 seconds of audio
   - Features are extracted from buffered audio
   - Fuzzy inference runs against all registered users
   - Best match is returned with confidence score

3. **Fuzzy Logic**:
   - Each user has personalized membership functions
   - Triangular membership functions centered on user's feature values
   - Mamdani inference system with if-then rules
   - Confidence threshold: 0.5 (below = "Unknown")

## Supported Audio Formats

- **M4A** (requires FFmpeg)
- **3GP** (requires FFmpeg)
- **WAV**
- **MP3**
- **OGG**

## Project Structure

```
backend/
├── main.py                 # FastAPI application and endpoints
├── database.py            # SQLite database operations
├── audio_converter.py     # Audio format conversion
├── feature_extractor.py   # MFCC, spectral centroid, ZCR extraction
├── fuzzy_engine.py        # Fuzzy logic inference engine
├── requirements.txt       # Python dependencies
├── speaker_profiles.db    # SQLite database (created automatically)
└── README.md             # This file
```

## Configuration

### Adjust Recognition Sensitivity

Edit `fuzzy_engine.py` to modify:
- Membership function ranges (tolerance)
- Confidence threshold (currently 0.5)
- Fuzzy rules

### Change Audio Buffer Size

Edit `main.py`:
```python
BUFFER_CHUNK_SIZE = 2  # Process every 2 seconds (adjust as needed)
```

## Troubleshooting

### FFmpeg Not Found
- Ensure FFmpeg is installed and in PATH
- Verify with: `ffmpeg -version`
- See [Prerequisites](#prerequisites) for installation instructions

### Audio Conversion Fails
- Check audio file is valid and not corrupted
- Ensure FFmpeg is properly installed
- Try a different audio format (WAV works without FFmpeg)

### Low Recognition Accuracy
- Use longer audio samples for registration (1+ minute recommended)
- Record in quiet environment
- Ensure good audio quality
- Try registering multiple samples per user

### Port Already in Use
```bash
# Change port in main.py or use:
uvicorn main:app --host 0.0.0.0 --port 8001
```

## Development

### Running Tests

```python
# Test database
from database import Database
db = Database()
db.initialize()

# Test feature extraction
from feature_extractor import FeatureExtractor
import numpy as np
extractor = FeatureExtractor()
audio = np.random.randn(16000 * 2)  # 2 seconds
features = extractor.extract_features(audio)
print(features)
```

### Code Structure

- **Modular Design**: Each component is independent and testable
- **Type Hints**: Full type annotations for better IDE support
- **Error Handling**: Comprehensive error handling throughout
- **Logging**: Structured logging for debugging

## Production Deployment

### Using Uvicorn with Gunicorn

```bash
pip install gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Using Docker

See `IMPLEMENTATION_GUIDE.md` for Docker setup instructions.

## Performance Tips

- **Buffer Size**: Smaller buffers = faster response, less accuracy
- **Feature Caching**: Consider caching features for registered users
- **Database Indexing**: Already optimized with UNIQUE constraints
- **Connection Pooling**: Use for high-traffic scenarios

## License

This project is open source and available for use.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For detailed implementation documentation, see [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md).

For issues or questions:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review the implementation guide
3. Check server logs for error messages

## Acknowledgments

- **FastAPI** - Modern web framework
- **librosa** - Audio analysis library
- **scikit-fuzzy** - Fuzzy logic toolkit
- **pydub** - Audio manipulation

---

**Made with ❤️ for speaker recognition**

