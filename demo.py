import cv2, io, json, httpx
from api.auth import create_token
from scripts.make_toy_dataset import make_field
print("Creating test blood-smear image...")
img = make_field(14, seed=3)
cv2.imwrite("test_smear.png", img)
ok, buf = cv2.imencode(".png", img)
token = create_token("demo")
print("Sending to HemoSight for analysis...\n")
r = httpx.post("http://127.0.0.1:8000/analyze", files={"file": ("smear.png", io.BytesIO(buf.tobytes()), "image/png")}, headers={"Authorization": f"Bearer {token}"}, timeout=120)
print("========== RESULT FROM YOUR PROJECT ==========")
print(json.dumps(r.json(), indent=2))
