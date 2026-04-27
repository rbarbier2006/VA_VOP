# VA_VOP

Streamlit web app that preserves the original VOP signal-analysis workflow:
- Load a text signal file (one sample per line).
- Compute centered rolling mean and derivative.
- Detect inflation starts from derivative peaks.
- Run multiple rolling regressions for each compression.
- Plot the original signal, rolling mean, slope curve, detected peaks, regression lines, labels, and endpoints.
- Show/export regression slopes.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL printed by Streamlit (typically `http://localhost:8501`).
