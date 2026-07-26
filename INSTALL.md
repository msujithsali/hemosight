# HemoSight — Installation Guide (Hardware + Software)

Author: **M Sujith Sali**, ISE Dept, VTU Karnataka.

> **SCREENING AID ONLY — NOT A DIAGNOSTIC DEVICE.** Setup instructions below do
> not make HemoSight a medical device. Real clinical use needs `[TARGET]`
> validation, ethics approval, and clinician oversight.

---

## Part A — Software installation

### A1. Prerequisites
- Python 3.11+ and `git`.
- Node.js 18+ (only for the web UI).
- On Linux, native libs for OpenCV + WeasyPrint PDF:
  ```bash
  sudo apt-get update && sudo apt-get install -y \
      libgl1-mesa-glx libglib2.0-0 libpango-1.0-0 libpangocairo-1.0-0 \
      libgdk-pixbuf-2.0-0 libffi-dev
  ```

### A2. Get the code and install
```bash
git clone <your-repo-url> hemosight   # or unzip HemoSight.zip
cd hemosight
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

### A3. Make it fully functional (real weights, no manual data)
```bash
python -m scripts.train_toy      # builds synthetic data + trains TinyCNNs (CPU, ~1 min)
                                 # -> results/wbc_tiny.pt, malaria_tiny.pt (+ .onnx)
```
This produces **real model weights** so the whole pipeline runs. (Synthetic
data => accuracy is not clinical; see MODEL_CARD.)

### A4. Run the tests (proves the software works)
```bash
pytest -q            # 22 tests pass (determinism, schema, egress, quality gate,
                     # MC-Dropout, attention gate, ONNX parity, signed PDF,
                     # audit chain, federated smoke, full end-to-end pipeline)
```

### A5. Run the API and analyze an image
```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000
# In another shell — get a token in Python:
#   from api.auth import create_token; print(create_token("asha01"))
curl -H "Authorization: Bearer <TOKEN>" -F "file=@smear.png" \
     http://127.0.0.1:8000/analyze
# -> JSON with cell counts, WBC differential, malaria flag, uncertainty,
#    attention-gate status, and a path to a signed PDF report.
```

### A6. Run the web UI (Kannada / Hindi / English)
```bash
cd web && npm install && npm run dev   # http://localhost:5173
```

### A7. Optional — real datasets for real accuracy
`make download-data` (BCCD + NIH auto; Raabin needs a one-time site
registration), then `make train-wbc / train-malaria / train-yolo /
federated-simulate / eval-all`. These log real metrics to MLflow and replace
the synthetic weights.

### A8. Docker (everything at once)
```bash
docker compose up --build    # api + mlflow + postgres + aggregator
```

---

## Part B — Hardware assembly (Raspberry Pi 5)

### B1. Bill of materials (see README for Indian sellers / indicative INR)
Raspberry Pi 5 8GB · 27W USB-C PD supply · Active Cooler · 64/128GB U3 microSD
· official case · UVC USB microscope (40x–1000x) · prepared teaching smear
slides. Optional: Hailo-8L AI Kit, 7-inch touchscreen.

### B2. Flash the OS
1. Download **Raspberry Pi Imager** on your laptop.
2. Choose **Raspberry Pi OS (64-bit, Bookworm)** → your microSD.
3. In Imager settings: set hostname `hemosight`, enable SSH, set username/password,
   configure Wi-Fi. Write.

### B3. Assemble
1. Fit the **Active Cooler** onto the Pi 5 (it clips + plugs into the fan header).
2. Seat the Pi in the case, insert the flashed microSD.
3. Plug the **UVC USB microscope** into a USB 3.0 (blue) port.
4. Connect the **27W USB-C** supply last (this powers on).

### B4. First boot + software
```bash
ssh hemosight@hemosight.local
sudo apt-get update && sudo apt-get install -y python3-venv git \
     libgl1-mesa-glx libglib2.0-0 libpango-1.0-0 libpangocairo-1.0-0
git clone <your-repo-url> /opt/hemosight && cd /opt/hemosight
python3 -m venv .venv && source .venv/bin/activate
pip install -e "."           # ONNX Runtime path is edge-friendly
python -m scripts.train_toy  # or copy trained .onnx from your dev machine
```

### B5. Verify the microscope
```bash
python -c "import cv2; c=cv2.VideoCapture(0); ok,_=c.read(); print('camera ok:', ok)"
```
Set `HEMOSIGHT_CAMERA=uvc` (default) or `picamera` for the Pi Camera.

### B6. Run as a service (auto-start on boot)
```bash
sudo cp edge/hemosight.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now hemosight
systemctl status hemosight        # should be active (running)
```
The service serves the API on `127.0.0.1:8000`. Open the web UI (or a browser
on the Pi in kiosk mode with the 7-inch touchscreen) to capture and screen a
smear. The inference path runs **fully offline** — the egress guard blocks any
outbound call except the sanctioned federated weight-delta sync.

### B7. Benchmark latency on the Pi
```bash
python -c "from edge.infer import EdgeInferencer, benchmark; import cv2; \
img=cv2.imread('smear.png'); print('median s/img:', benchmark(EdgeInferencer(), img))"
```
Report the **actual** number (target < 3 s/image); do not assume it.

---

## Troubleshooting
- **WeasyPrint import error** → install the `libpango`/`libgdk-pixbuf` packages in A1/B4.
- **`camera ok: False`** → try index 1/2, confirm the scope is UVC, check `lsusb`.
- **ONNX provider warning** → XNNPACK falls back to CPU automatically; still works.
- **401 from /analyze** → include a valid `Authorization: Bearer <token>` header.
