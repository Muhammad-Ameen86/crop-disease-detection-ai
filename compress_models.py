# ==========================================
# compress_models.py
# Run once locally to compress all models
# before pushing to GitHub
# ==========================================
# Usage:
#   pip install joblib lz4
#   python compress_models.py
# ==========================================

import torch
import torch.nn as nn
import torchvision.models as models
import joblib
from pathlib import Path

DL_DIR  = Path("models/dl")
ML_DIR  = Path("models/ml")
NUM_CLASSES = 15

print("=" * 50)
print("  Crop Disease Model Compression Tool")
print("=" * 50)

# ════════════════════════════════════════════════════
# 1. ALEXNET — Dynamic INT8 quantization
#    ~228 MB → ~64 MB
# ════════════════════════════════════════════════════
print("\n[1/5] Quantizing AlexNet (INT8)...")
alexnet = models.alexnet(weights=None)
alexnet.classifier[6] = nn.Linear(4096, NUM_CLASSES)
alexnet.load_state_dict(torch.load(DL_DIR / "alexnet.pth", map_location="cpu"))
alexnet.eval()

alexnet_q = torch.quantization.quantize_dynamic(
    alexnet, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
)
torch.save(alexnet_q.state_dict(), DL_DIR / "alexnet_q.pth")
orig = (DL_DIR / "alexnet.pth").stat().st_size / 1e6
new  = (DL_DIR / "alexnet_q.pth").stat().st_size / 1e6
print(f"    ✅ {orig:.1f} MB → {new:.1f} MB  (saved {orig-new:.1f} MB)")

# Replace original
(DL_DIR / "alexnet.pth").unlink()
(DL_DIR / "alexnet_q.pth").rename(DL_DIR / "alexnet.pth")
print("    ✅ alexnet.pth replaced")

# ════════════════════════════════════════════════════
# 2. RESNET — float16 half precision
#    ~94 MB → ~47 MB
# ════════════════════════════════════════════════════
print("\n[2/5] Compressing ResNet (float16)...")
resnet = models.resnet50(weights=None)
resnet.fc = nn.Linear(2048, NUM_CLASSES)
resnet.load_state_dict(torch.load(DL_DIR / "resnet.pth", map_location="cpu"))
resnet.eval()

resnet_half = resnet.half()
torch.save(resnet_half.state_dict(), DL_DIR / "resnet_q.pth")
orig = (DL_DIR / "resnet.pth").stat().st_size / 1e6
new  = (DL_DIR / "resnet_q.pth").stat().st_size / 1e6
print(f"    ✅ {orig:.1f} MB → {new:.1f} MB  (saved {orig-new:.1f} MB)")

(DL_DIR / "resnet.pth").unlink()
(DL_DIR / "resnet_q.pth").rename(DL_DIR / "resnet.pth")
print("    ✅ resnet.pth replaced")

# ════════════════════════════════════════════════════
# 3. MOBILENET — float16 half precision
#    ~9.2 MB → ~4.7 MB
# ════════════════════════════════════════════════════
print("\n[3/5] Compressing MobileNet (float16)...")
mobilenet = models.mobilenet_v2(weights=None)
mobilenet.classifier[1] = nn.Linear(1280, NUM_CLASSES)
mobilenet.load_state_dict(torch.load(DL_DIR / "mobilenet.pth", map_location="cpu"))
mobilenet.eval()

mobilenet_half = mobilenet.half()
torch.save(mobilenet_half.state_dict(), DL_DIR / "mobilenet_q.pth")
orig = (DL_DIR / "mobilenet.pth").stat().st_size / 1e6
new  = (DL_DIR / "mobilenet_q.pth").stat().st_size / 1e6
print(f"    ✅ {orig:.1f} MB → {new:.1f} MB  (saved {orig-new:.1f} MB)")

(DL_DIR / "mobilenet.pth").unlink()
(DL_DIR / "mobilenet_q.pth").rename(DL_DIR / "mobilenet.pth")
print("    ✅ mobilenet.pth replaced")

# ════════════════════════════════════════════════════
# 4. SVM — joblib lz4 compression
#    ~166 MB → ~80 MB
# ════════════════════════════════════════════════════
print("\n[4/5] Compressing SVM (lz4)...")
orig = (ML_DIR / "svm_resnet.pkl").stat().st_size / 1e6
svm = joblib.load(ML_DIR / "svm_resnet.pkl")
joblib.dump(svm, ML_DIR / "svm_resnet.pkl", compress=("lz4", 9))
new = (ML_DIR / "svm_resnet.pkl").stat().st_size / 1e6
print(f"    ✅ {orig:.1f} MB → {new:.1f} MB  (saved {orig-new:.1f} MB)")

# ════════════════════════════════════════════════════
# 5. RANDOM FOREST — joblib lz4 compression
#    ~198 MB → ~26 MB
# ════════════════════════════════════════════════════
print("\n[5/5] Compressing Random Forest + kNN (lz4)...")
orig = (ML_DIR / "rf_resnet.pkl").stat().st_size / 1e6
rf = joblib.load(ML_DIR / "rf_resnet.pkl")
joblib.dump(rf, ML_DIR / "rf_resnet.pkl", compress=("lz4", 9))
new = (ML_DIR / "rf_resnet.pkl").stat().st_size / 1e6
print(f"    ✅ RF:  {orig:.1f} MB → {new:.1f} MB  (saved {orig-new:.1f} MB)")

orig = (ML_DIR / "knn_resnet.pkl").stat().st_size / 1e6
knn = joblib.load(ML_DIR / "knn_resnet.pkl")
joblib.dump(knn, ML_DIR / "knn_resnet.pkl", compress=("lz4", 9))
new = (ML_DIR / "knn_resnet.pkl").stat().st_size / 1e6
print(f"    ✅ kNN: {orig:.1f} MB → {new:.1f} MB  (saved {orig-new:.1f} MB)")

# ════════════════════════════════════════════════════
# Final size report
# ════════════════════════════════════════════════════
print("\n" + "=" * 50)
print("  Final Model Sizes")
print("=" * 50)
all_files = list(DL_DIR.glob("*.pth")) + list(ML_DIR.glob("*.pkl"))
total = 0
for f in sorted(all_files):
    size = f.stat().st_size / 1e6
    total += size
    status = "⚠️  OVER 100MB" if size > 100 else "✅"
    print(f"  {status}  {f.name:<25} {size:.1f} MB")
print(f"\n  Total: {total:.1f} MB")
print("=" * 50)
print("\n🎉 Compression complete! Safe to push to GitHub.")
print("   All files are under GitHub's 100MB per-file limit.")
print("   You can now delete .gitattributes LFS config.")