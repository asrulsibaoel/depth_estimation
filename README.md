# YOLO + MiDAS Object Detection & Distance Estimation Service

This project bundles a YOLO-based object detection model with MiDAS depth estimation to provide real-time detection and distance measurement. The service is wrapped inside a FastAPI backend for easy deployment and integration.

---

## Features

- **Object Detection** using YOLO (You Only Look Once) for fast and accurate detection.
- **Distance Estimation** using MiDAS depth estimation model to estimate object distances in meters.
- Exposed as a **REST API** via FastAPI.
- Easy to run with a simple `run.sh` script.
- Dependencies listed in `requirements.txt` for easy environment setup.

---

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/asrulsibaoel/depth_estimation.git
   cd depth_estimation
   ```

2. Create a Python virtual environment (optional but recommended):

   ```bash
   python3 -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. Install required packages:

   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

Run the FastAPI service with the provided `run.sh` script:

```bash
./run.sh
```

By default, the server will start at:

```
http://localhost:8000
```

---

## API Endpoints

- `POST /predict`  
  Accepts an image file along with camera parameters and returns detected objects with estimated distances.

---

## Camera Focal Length Calculation

To estimate distance correctly, the system uses the camera's focal length. You can calculate the focal length (in pixels) from the camera's field of view (FOV) angle and the image width:

\[
f = \frac{w}{2 \times \tan\left(\frac{\theta}{2}\right)}
\]

where:  
- \( f \) = focal length in pixels  
- \( w \) = image width in pixels  
- \( \theta \) = camera horizontal field of view angle (in radians)

This formula is used internally based on the `camera_pov` parameter you provide (in degrees).

---

## Example Request

Here’s an example Python request using the `requests` library to send an image with camera parameters:

```python
import requests

API_URL = "http://localhost:8000/predict"
IMAGE_PATH = "path/to/your/image.jpg"

with open(IMAGE_PATH, "rb") as f:
    files = {"file": (IMAGE_PATH, f, "image/jpeg")}
    data = {
        "camera_pov": 80,   # Camera field of view in degrees
        "height": 5.05      # Average object height in meters (e.g. human height)
    }
    response = requests.post(API_URL, files=files, data=data)

print(response.json())
```

---

## Example Response

```json
[
  {'bbox': [125, 189, 247, 272], 'distance_m': 17.499951771},
  {'bbox': [125, 211, 241, 272], 'distance_m': 23.552397515588748}
]
```

---

## Project Structure

- `app/` — FastAPI application source code  
- `models/` — YOLO and MiDAS model files and utilities  
- `run.sh` — Shell script to launch the FastAPI server  
- `requirements.txt` — Python dependencies  

---

## Contributing

Contributions are welcome! Please open issues or pull requests for bug fixes, improvements, or new features.

---

## License

[MIT License](LICENSE)

---

## Contact

For questions or support, please contact:  
**Asrul Sani Ariesandy** — asrulsibaoel@gmail.com  
GitHub: [asrulsibaoel](https://github.com/asrulsibaoel)
