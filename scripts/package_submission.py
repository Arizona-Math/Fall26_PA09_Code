"""Package only permitted student sources and provenance."""
from pathlib import Path
import zipfile
root=Path(__file__).resolve().parents[1]
names=['bfgs.py', 'AUTHORS.md']
with zipfile.ZipFile(root/'submission.zip','w',zipfile.ZIP_DEFLATED) as z:
    for name in names: z.write(root/name,name)
print('Created submission.zip; upload this ZIP alone to Gradescope.')
