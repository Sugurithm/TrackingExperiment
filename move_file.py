from pathlib import Path
import shutil

src = Path(r"C:\project\experiment\recordings")
dst = Path(r"C:\project\experiment\joints_data\recordings")

for f in src.rglob("rs_joints.npz"):
    if f.parent.name != "joints":
        continue
    target = dst / f.relative_to(src)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(f, target)
    print("copied:", f.relative_to(src))