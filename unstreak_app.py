import streamlit as st
import numpy as np
import cv2
from PIL import Image
import unstreak
import io

st.set_page_config(page_title="Unstreak - Star Streak Removal", layout="wide")

st.title("Unstreak: Star Streak Removal")
st.markdown("""
Upload an astrophotography image to remove star streaks. 
**Pro Tip:** Use the "Preview" mode to tune settings on a crop, then apply to the full image.
""")

# Sidebar for controls
st.sidebar.header("Parameters")

img_file = st.sidebar.file_uploader("Upload Image", type=["jpg", "jpeg", "png", "tif"])

# Session State for Auto-Tuned values
if 'tuned_angle' not in st.session_state: st.session_state.tuned_angle = 0.0
if 'tuned_length' not in st.session_state: st.session_state.tuned_length = 10.0

# Detection Settings
st.sidebar.subheader("Detection")
threshold = st.sidebar.slider("Threshold (0-255)", 0, 255, 50, help="Brightness threshold to detect stars.")

# Correction Settings
st.sidebar.subheader("Correction Model")
# Use session state values if they exist, otherwise default
angle = st.sidebar.number_input("Streak Angle (deg)", value=float(st.session_state.tuned_angle), step=0.1, format="%.1f")
length = st.sidebar.number_input("Streak Length (px)", value=float(st.session_state.tuned_length), step=0.1, format="%.1f")
iterations = st.sidebar.slider("Iterations", 5, 50, 15, help="Higher = sharper but slower.")

def get_image_crop(img):
    """Return a center crop for fast tuning."""
    h, w = img.shape[:2]
    crop_size = 1024
    if h <= crop_size and w <= crop_size:
        return img
    
    cx, cy = w // 2, h // 2
    half = crop_size // 2
    cx, cy = w // 2, h // 2
    half = crop_size // 2
    return img[cy-half:cy+half, cx-half:cx+half]

def draw_streak_indicator(img, angle, length):
    """Draw a red line indicating the streak parameters."""
    vis = img.copy()
    h, w = vis.shape[:2]
    cx, cy = w // 2, h // 2
    
    # Same logic as kernel creation
    rad = np.deg2rad(-angle)
    dx = int((length * 5) * np.sin(rad)) # Scaled up for visibility
    dy = int((length * 5) * np.cos(rad))
    
    # Draw line
    cv2.line(vis, (cx - dx, cy - dy), (cx + dx, cy + dy), (0, 0, 255), 2)
    # Draw arrow head
    cv2.circle(vis, (cx + dx, cy + dy), 3, (0, 255, 0), -1)
    
    return vis

def process(image_bytes):
    # Convert to OpenCV format
    file_bytes = np.asarray(bytearray(image_bytes.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    if img is None:
        st.error("Error decoding image.")
        return

    # Prepare Crop for Preview/Tuning
    img_crop = get_image_crop(img)
    
    # Auto-Tune Logic
    col_tune, col_metrics = st.columns([1, 2])
    with col_tune:
        if st.button("✨ Auto-Optimize Settings"):
            with st.spinner("Optimizing parameters on crop..."):
                # Always use crop for optimization to be fast
                best_angle, best_length = unstreak.optimize_parameters(img_crop, angle, length, threshold)
                best_angle, best_length = unstreak.optimize_parameters(img_crop, angle, length, threshold)
                st.session_state.tuned_angle = float(best_angle)
                st.session_state.tuned_length = float(best_length)
                
                # Automatically generate preview result so metrics show up immediately
                res_bgr = unstreak.process_image_manual(img_crop, best_angle, best_length, iterations)
                st.session_state.preview_result = (img_crop, res_bgr)
                st.session_state.full_result = None
                
                st.success(f"Found: Angle={best_angle:.1f}°, Len={best_length:.1f}px")
                st.rerun()

    # Visualize Direction on Crop
    vis_img = draw_streak_indicator(img_crop, angle, length)
    
    # Generate High Contrast View used for Optimization
    _, scale_factor, contrast_view = unstreak.preprocess_image_for_optimization(img_crop)
    # Scale length for the smaller contrast view
    contrast_length = length * scale_factor
    vis_contrast = draw_streak_indicator(cv2.cvtColor(contrast_view, cv2.COLOR_GRAY2BGR), angle, contrast_length)

    c_vis1, c_vis2 = st.columns(2)
    c_vis1.image(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB), caption="1. Original (Zoomed)", use_container_width=True)
    c_vis2.image(cv2.cvtColor(vis_contrast, cv2.COLOR_BGR2RGB), caption="2. High Contrast (Optimization View)", use_container_width=True)
    
    st.caption("The **High Contrast** view shows strictly what the algorithm sees: only the brightest streaks against a black background.")

    # Action Buttons
    col_btn1, col_btn2 = st.columns(2)
    
    if col_btn1.button("Apply to Preview (Crop)"):
        with st.spinner("Processing Preview..."):
            res_bgr = unstreak.process_image_manual(img_crop, angle, length, iterations)
            st.session_state.preview_result = (img_crop, res_bgr)
            # Clear full result to avoid confusion if settings changed
            st.session_state.full_result = None

    if col_btn2.button("🚀 Process Full Image"):
        with st.spinner("Processing Full Image (This may take a while)..."):
            res_bgr = unstreak.process_image_manual(img, angle, length, iterations)
            st.session_state.full_result = (img, res_bgr)

    # Display Preview
    if 'preview_result' in st.session_state and st.session_state.preview_result:
        p_orig, p_res = st.session_state.preview_result
        st.subheader("Preview Result (Center Crop)")
        
        # Metrics for Preview
        orig_circ, orig_count = unstreak.calculate_star_metrics(p_orig, threshold)
        res_circ, res_count = unstreak.calculate_star_metrics(p_res, threshold)
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Orig. Roundness", f"{orig_circ:.2f}")
        m2.metric("Result Roundness", f"{res_circ:.2f}", f"{res_circ-orig_circ:.2f}")
        m3.metric("Orig. Stars", orig_count)
        m4.metric("Result Stars", res_count)
        
        c1, c2 = st.columns(2)
        c1.image(cv2.cvtColor(p_orig, cv2.COLOR_BGR2RGB), caption="Original Crop", use_container_width=True)
        c2.image(cv2.cvtColor(p_res, cv2.COLOR_BGR2RGB), caption="Restored Crop", use_container_width=True)
        
        st.divider()

    # Display Full Result
    if 'full_result' in st.session_state and st.session_state.full_result:
        f_orig, f_res = st.session_state.full_result
        st.subheader("Full Resolution Result")
        
        # Download
        is_success, buffer = cv2.imencode(".jpg", f_res)
        if is_success:
            st.download_button(
                label="⬇️ Download Restored Image",
                data=buffer.tobytes(),
                file_name="restored_full.jpg",
                mime="image/jpeg",
                key="dl_btn_full"
            )
            
        c1, c2 = st.columns(2)
        c1.image(cv2.cvtColor(f_orig, cv2.COLOR_BGR2RGB), caption="Original Full", use_container_width=True)
        c2.image(cv2.cvtColor(f_res, cv2.COLOR_BGR2RGB), caption="Restored Full", use_container_width=True)

if img_file is not None:
    # Reset file pointer
    img_file.seek(0)
    process(img_file)

