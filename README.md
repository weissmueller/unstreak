**Unstreak** is a specialized tool for astrophotographers to remove star trailing artifacts caused by tracking errors or wind movement. It uses **Richardson-Lucy Deconvolution** with a smart auto-optimization engine to restore round stars.

> **Note**: Currently, Unstreak is optimized for **linear streaks** (straight lines). Curved trails (e.g., from field rotation) may not be fully corrected.

## Features

- **✨ Auto-Optimization**: Automatically detects the streak angle and length with high precision (0.2°). 
- **🚀 High-Speed Processing**: Uses smart downscaling and adaptive contrast stretching to find optimal settings in seconds.
- **👁️ Visual Debugging**: Double-check the algorithm with a "High Contrast" preview that shows exactly what the computer sees.
- **📊 Real-time Metrics**: Track "Star Roundness" and "Star Count" to objectively verify improvements.
- **🖼️ Full Resolution Export**: Process your full 24MP+ images once you are happy with the preview.

## Installation

1. Clone this repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the web interface:

```bash
streamlit run unstreak_app.py
```

1. **Upload** your image (JPG, PNG, TIF).
2. Click **"✨ Auto-Optimize Settings"** to find the streak direction.
3. Review the **Red Line** indicator on the "High Contrast" preview to confirm it matches the streaks.
4. Click **"🚀 Process Full Image"** to apply the correction to the entire photo.
5. **Download** the restored result.
