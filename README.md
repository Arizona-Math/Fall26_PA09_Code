# PA09 starter

Read [problem.md](problem.md) for deadlines, mathematical contracts, and grading.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -c "import numpy, scipy; print(numpy.__version__, scipy.__version__)"
make check
make experiments
make submission
```

Public checks intentionally fail until the TODO functions are implemented.
Upload `submission.zip` alone to the Code assignment. The written/report PDF
goes to its separate Gradescope assignment. Do not upload instructor material.
