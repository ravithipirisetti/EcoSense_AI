---
title: EcoSense AI Server
emoji: 🦅
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
short_description: Production REST AI Inference Server for Bird Sound Classification
---

# EcoSense AI Server

Official Production AI Inference Server for Bird Species Sound Classification.  
Deployed on **Render** to serve REST API inference calls simultaneously for the EcoSense Website, Raspberry Pi IoT nodes, Mobile Applications, and Desktop clients.

```
                Render
        --------------------
        EcoSense AI Server
        audio_model_yamnet.keras
        FastAPI REST Engine
        --------------------
              HTTPS API
                   │
      ┌────────────┼────────────┐
      │            │            │
      ▼            ▼            ▼
 Website      Raspberry Pi   Mobile App
```

---

## 🌐 Production Deployment Info

- **Render Web Service URL**: `https://ecosense-ai-server.onrender.com`
- **API Base URL**: `https://ecosense-ai-server.onrender.com`
- **OpenAPI / Swagger Docs**: `https://ecosense-ai-server.onrender.com/docs`

---

## 🔑 Required Headers

All prediction requests (`/predict`, `/identify`, `/api/v1/predict`) require the API key header:

| Header | Description | Required | Example |
| :--- | :--- | :--- | :--- |
| `x-api-key` | API key configured in server environment variable `API_KEY` | **Yes** | `ecosense_secret_api_key_2026` |
| `Content-Type` | Multipart Form Data boundary | **Yes** | `multipart/form-data` |

---

## 🛰️ REST API Endpoints

### 1. Root Status (`GET /`)
```json
{
  "name": "EcoSense AI Server",
  "status": "running"
}
```

### 2. Health Check (`GET /health`)
```json
{
  "status": "healthy",
  "model_loaded": true,
  "encoder_loaded": true,
  "classes": 66,
  "version": "1.0.0"
}
```

### 3. Server Info (`GET /info`)
```json
{
  "server": "EcoSense AI",
  "framework": "FastAPI",
  "model": "audio_model_yamnet.keras",
  "version": "1.0.0",
  "classes": 66
}
```

### 4. Audio Prediction (`POST /predict`, `POST /identify`, `POST /api/v1/predict`)
Accepts `multipart/form-data` with field name `audio` (`.wav`, `.mp3`, `.flac`, `.ogg`).

---

## 💻 Code Examples

### 1. cURL Example Request
```bash
curl -X POST "https://ecosense-ai-server.onrender.com/predict" \
  -H "x-api-key: ecosense_secret_api_key_2026" \
  -F "audio=@/path/to/recording.wav"
```

### 2. Example API Response (JSON)
```json
{
  "status": "success",
  "prediction": {
    "common_name": "Eurasian Collared Dove",
    "scientific_name": "Streptopelia decaocto",
    "confidence": 80.82
  },
  "top_predictions": [
    {
      "common_name": "Eurasian Collared Dove",
      "scientific_name": "Streptopelia decaocto",
      "confidence": 80.82
    },
    {
      "common_name": "Spotted Dove",
      "scientific_name": "Spilopelia chinensis",
      "confidence": 10.55
    }
  ],
  "processing_time_ms": 482.15,
  "model_version": "1.0.0",
  "request_id": "7d2ef67a-8b1e-43a2-9b2f-38e910245a1c"
}
```

### 3. Python Client Example
```python
import requests

SERVER_URL = "https://ecosense-ai-server.onrender.com/predict"
API_KEY = "ecosense_secret_api_key_2026"

headers = {"x-api-key": API_KEY}
files = {"audio": ("recording.wav", open("recording.wav", "rb"), "audio/wav")}

response = requests.post(SERVER_URL, headers=headers, files=files)
data = response.json()

print(f"Status: {data['status']}")
print(f"Predicted Bird: {data['prediction']['common_name']} ({data['prediction']['scientific_name']})")
print(f"Confidence: {data['prediction']['confidence']}%")
```

### 4. JavaScript / Browser / Node.js Client Example
```javascript
const formData = new FormData();
formData.append('audio', fileInputElement.files[0]);

fetch('https://ecosense-ai-server.onrender.com/predict', {
  method: 'POST',
  headers: {
    'x-api-key': 'ecosense_secret_api_key_2026'
  },
  body: formData
})
.then(res => res.json())
.then(data => {
  console.log('Prediction:', data.prediction.common_name);
  console.log('Scientific Name:', data.prediction.scientific_name);
  console.log('Confidence:', data.prediction.confidence + '%');
})
.catch(err => console.error('API Error:', err));
```

### 5. Raspberry Pi Client Example (Python `requests`)
```python
import time
import requests

API_URL = "https://ecosense-ai-server.onrender.com/predict"
API_KEY = "ecosense_secret_api_key_2026"
AUDIO_FILE = "/home/pi/audio_clips/recorded.wav"

def send_pi_audio_inference(audio_path):
    headers = {"x-api-key": API_KEY}
    with open(audio_path, "rb") as audio_file:
        files = {"audio": audio_file}
        response = requests.post(API_URL, headers=headers, files=files, timeout=30)
    
    if response.status_code == 200:
        result = response.json()
        pred = result.get("prediction", {})
        print(f"[RPI Node] Bird: {pred.get('common_name')} | Conf: {pred.get('confidence')}% | Latency: {result.get('processing_time_ms')}ms")
        return result
    else:
        print(f"[RPI Node] Error {response.status_code}: {response.text}")
        return None

if __name__ == "__main__":
    send_pi_audio_inference(AUDIO_FILE)
```

---

## 🚀 Render Deployment Instructions

1. Push this repository to GitHub.
2. Go to **Render Dashboard** -> **New Web Service**.
3. Connect your repository.
4. Set settings:
   - **Environment**: `Python`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Configure Environment Variables in Render:
   - `API_KEY`: Your secret API key
   - `ALLOWED_ORIGINS`: `https://ecosense.onrender.com,https://ecosense-ai.web.app`
   - `PORT`: Automatically set by Render
