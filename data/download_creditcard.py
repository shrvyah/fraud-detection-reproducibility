from pathlib import Path
from urllib.request import urlretrieve
import hashlib

URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
EXPECTED_SHA256 = "76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89"
out = Path(__file__).resolve().parent / "creditcard.csv"
out.parent.mkdir(parents=True, exist_ok=True)
if not out.exists():
    print("Downloading", URL)
    urlretrieve(URL, out)
digest = hashlib.sha256(out.read_bytes()).hexdigest()
if digest != EXPECTED_SHA256:
    raise SystemExit(f"SHA-256 mismatch: {digest} != {EXPECTED_SHA256}")
print(f"Verified {out}: {digest}")
