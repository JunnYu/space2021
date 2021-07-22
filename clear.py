import shutil
from pathlib import Path

for p in Path(".").glob("**/__pycache__"):
    print(p)
    shutil.rmtree(p)
