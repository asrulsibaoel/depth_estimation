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
   cd your-repo-name
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

- `POST /detect`  
  Accepts an image file and returns detected objects along with estimated distances.

### Example Request (using `curl`):

```bash
curl -X POST "http://localhost:8000/detect" -F "file=@path_to_your_image.jpg"
```

### Example Response:

```json
[
  {
    "bbox": [x_min, y_min, x_max, y_max],
    "label": "person",
    "confidence": 0.98,
    "distance_m": 3.45
  },
  ...
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
