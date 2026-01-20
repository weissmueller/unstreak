#!/usr/bin/env python3
import cv2
import numpy as np
import argparse
import sys
from skimage import restoration, util, img_as_float
from scipy.signal import convolve2d

def parse_args():
    parser = argparse.ArgumentParser(description="Remove star streaks from astrophotography images.")
    parser.add_argument("input", help="Path to input image")
    parser.add_argument("output", help="Path to save output image")
    parser.add_argument("--threshold", type=int, default=200, help="Brightness threshold for star detection (0-255). Default: 200")
    parser.add_argument("--min-length", type=int, default=10, help="Minimum streak length to consider. Default: 10")
    parser.add_argument("--iterations", type=int, default=30, help="Number of iterations for Richardson-Lucy deconvolution. Default: 30")
    parser.add_argument("--reduce-length", type=int, default=5, help="Pixels to subtract from detected length to account for star width. Default: 5")
    return parser.parse_args()

def estimate_psf_parameters(image, threshold=200, min_length=10):
    """
    Estimate the angle and length of star streaks.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Threshold to find bright stars
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    angles = []
    lengths = []
    
    for contour in contours:
        if len(contour) < 5: # FitEllipse needs at least 5 points
            continue
            
        # Fit ellipse
        (x, y), (MA, ma), angle = cv2.fitEllipse(contour)
        
        # Check if it looks like a streak (elongated)
        if ma > min_length and ma > 2 * MA: # At least twice as long as wide
            # Angle from fitEllipse is 0-180. 
            # We align it to be the direction of motion.
            angles.append(angle)
            lengths.append(ma)
            
    if not angles:
        return None, None
        
    # Use median to be robust against outliers
    median_angle = np.median(angles)
    median_length = np.median(lengths)
    
    print(f"Detected {len(angles)} streaks.")
    print(f"Estimated Angle: {median_angle:.2f} degrees")
    print(f"Estimated Length: {median_length:.2f} pixels")
    
    return median_angle, median_length

def create_motion_kernel(length, angle):
    """
    Create a linear motion blur kernel.
    """
    # Convert angle to radians
    # Note: We negate the angle because cv2.fitEllipse is clockwise(?), 
    # but our sin/cos logic combined with image coords (y down) needs adjustment.
    # Testing showed that 135 deg detected needs to result in a '\' kernel, which -135 achieves.
    angle = np.deg2rad(-angle)
    
    # Determine size of the kernel
    size = int(length)
    if size % 2 == 0:
        size += 1
        
    kernel = np.zeros((size, size))
    center = size // 2
    
    # Draw the line
    # Calculate start and end points relative to center
    dx = int((length / 2) * np.sin(angle))
    dy = int((length / 2) * np.cos(angle))
    
    # Check signs to ensure we draw across the center
    # OpenCV line drawing
    cv2.line(kernel, (center - dx, center - dy), (center + dx, center + dy), 1, thickness=1)
    
    # Normalize
    kernel /= np.sum(kernel)
    
    return kernel

def process_image(img, threshold=200, min_length=10, reduce_length=5, iterations=30):
    """
    Process the image to remove star streaks.
    Returns the restored image (BGR uint8) or raises an error/returns None if failed.
    """
    print("Estimating Streak Parameters...")
    angle, length = estimate_psf_parameters(img, threshold, min_length)
    
    if angle is None:
        raise ValueError("Could not detect any valid streaks. Try adjusting threshold or min-length.")
        
    # Adjust length for star width
    adjusted_length = max(1.0, length - reduce_length)
    print(f"Adjusting length from {length:.2f} to {adjusted_length:.2f} (Subtracting {reduce_length})")
        
    print("Creating PSF Kernel...")
    psf = create_motion_kernel(adjusted_length, angle)
    
    print("Deconvolving...")
    # Convert to float for processing
    img_float = img_as_float(img)
    
    # Process each channel independently
    restored_channels = []
    for i in range(3): # BGR channels
        channel = img_float[:, :, i]
        # Richardson-Lucy deconvolution is generally better for stars as it enforces non-negativity
        deconvolved = restoration.richardson_lucy(channel, psf, num_iter=iterations, clip=False)
        restored_channels.append(deconvolved)
        
    restored_img = np.dstack(restored_channels)
    
    # Clip to valid range
    restored_img = np.clip(restored_img, 0, 1)
    
    # Convert back to uint8
    output_img = (restored_img * 255).astype(np.uint8)
    return output_img

def process_image_manual(img, angle, length, iterations=30):
    """
    Process image with EXPLICIT angle and length (no auto-estimation).
    """
    print(f"Deconvolving with Angle={angle:.2f}, Length={length:.2f}")
    psf = create_motion_kernel(length, angle)
    
    img_float = img_as_float(img)
    restored_channels = []
    
    # Handle both grayscale and color
    if len(img.shape) == 3:
        for i in range(3):
            channel = img_float[:, :, i]
            deconvolved = restoration.richardson_lucy(channel, psf, num_iter=iterations, clip=False)
            restored_channels.append(deconvolved)
        restored_img = np.dstack(restored_channels)
    else:
        restored_img = restoration.richardson_lucy(img_float, psf, num_iter=iterations, clip=False)
        
    restored_img = np.clip(restored_img, 0, 1)
    return (restored_img * 255).astype(np.uint8)

def calculate_star_metrics(image, threshold=50):
    """
    Calculate metrics to evaluate star roundness and detection count.
    Returns:
        avg_circularity (float): 0.0 to 1.0 (1.0 is perfect circle)
        num_stars (int): Number of stars detected
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    circularities = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 5: # Ignore noise
            continue
            
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
            
        # Circularity = 4 * pi * Area / (Perimeter^2)
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        circularities.append(circularity)
        
    if not circularities:
        return 0.0, 0
        
    avg_circularity = np.mean(circularities)
    return avg_circularity, len(circularities)

def optimize_parameters(image, initial_angle=None, initial_length=None, threshold=50):
    """
    Optimize angle and length to maximize star energy concentration (brightness).
    Uses downscaling for speed and contrast stretching for robustness.
    """
def preprocess_image_for_optimization(image, target_dim=400):
    """
    Prepares an image for optimization: downscales and stretches contrast.
    Returns:
    - channel: float32 image (0-1) used for processing (green channel or gray)
    - scale: float scaling factor applied (original / new)
    - vis_img: uint8 image (0-255) for visualization
    """
    h, w = image.shape[:2]
    scale = 1.0
    working_img = image
    
    if max(h, w) > target_dim:
        scale = target_dim / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        working_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Contrast Stretching & Thresholding
    if len(working_img.shape) == 3:
        img_float = img_as_float(working_img)
        channel = img_float[:, :, 1] # Green
    else:
        img_float = img_as_float(working_img)
        channel = img_float
        
    # Adaptive Thresholding (Aggressive Contrast)
    # User requested "binary image" or maximized contrast.
    # We clip everything below Mean + 1.5*StdDev to black (0).
    # This removes background noise and faint nebulosity, isolating stars/streaks.
    mean_val = np.mean(channel)
    std_val = np.std(channel)
    thresh_val = mean_val + 1.5 * std_val
    
    # Apply Threshold to Zero (keep values above threshold, set others to 0)
    # We implement this manually for float array or use numpy
    channel_thresh = np.where(channel > thresh_val, channel, 0.0)
    
    # Normalize the result to 0-1 (0-255 equivalent)
    # If the image is all black, handle safely
    c_max = channel_thresh.max()
    if c_max > 0:
        channel = channel_thresh / c_max
    else:
        channel = channel_thresh # All zero
        
    # Create visualization image (uint8)
    vis_img = (np.clip(channel, 0, 1) * 255).astype(np.uint8)
    # Make it 3-channel for consistent handling if needed, or keep gray. 
    # Let's return gray, cv2 handles it.
    
    return channel, scale, vis_img

def optimize_parameters(image, initial_angle=None, initial_length=None, threshold=50):
    """
    Optimize angle and length to maximize star energy concentration (brightness).
    Uses downscaling for speed and contrast stretching for robustness.
    """
    print(f"Starting optimization...")
    
    # 1. Preprocess
    channel, scale, _ = preprocess_image_for_optimization(image, target_dim=400)
    
    print(f"Preprocessed image. Scale factor: {scale:.3f}")

    # 3. Search Configuration
    
    # Coarse Search
    coarse_angles = np.arange(0, 180, 15)
    # Lengths need to be scaled to working resolution
    # Standard lengths to test: 5, 10, 20, 30 px in working res
    coarse_lengths = [5, 10, 20, 30] 
    
    best_score = -1.0
    best_params = (0, 10)
        
    def evaluate(a, l):
        try:
            # Construct PSF
            psf = create_motion_kernel(l, a)
            
            # Deconvolve
            deconvolved = restoration.richardson_lucy(channel, psf, num_iter=10, clip=False)
            
            # Metric: Energy Concentration (Top 1% Brightness)
            sorted_pixels = np.sort(deconvolved.ravel())
            top_1_percent_idx = int(deconvolved.size * 0.99)
            score = np.mean(sorted_pixels[top_1_percent_idx:])
            
            return score
        except Exception:
            return 0.0

    print("Running Global Coarse Search...")
    for a in coarse_angles:
        for l in coarse_lengths:
            score = evaluate(a, l)
            if score > best_score:
                best_score = score
                best_params = (a, l)
                
    print(f"Coarse Best: Angle={best_params[0]}, Length={best_params[1]}, Score={best_score:.4f}")
    
    # Fine Search (Range +/- 30, Step 3)
    center_a, center_l = best_params
    
    fine_angles = np.arange(center_a - 30, center_a + 31, 3) 
    fine_lengths = np.linspace(max(3, center_l - 10), center_l + 10, 7)
    
    print("Running Local Fine Search...")
    for a in fine_angles:
        for l in fine_lengths:
            score = evaluate(a, l)
            if score > best_score:
                best_score = score
                best_params = (a, l)

    print(f"Fine Best: Angle={best_params[0]:.2f}, Length={best_params[1]:.2f}, Score={best_score:.4f}")

    # Super Fine Search (Range +/- 2, Step 0.2)
    center_a, center_l = best_params
    
    sf_angles = np.arange(center_a - 2.0, center_a + 2.1, 0.2)
    sf_lengths = np.linspace(max(3, center_l - 3), center_l + 3, 5)
    
    print("Running Super-Fine Search...")
    for a in sf_angles:
        for l in sf_lengths:
            score = evaluate(a, l)
            if score > best_score:
                best_score = score
                best_params = (a, l)
    
    # 4. Scale Result back to Original Resolution
    final_angle = best_params[0]
    final_length = best_params[1] / scale
    
    print(f"Optimization finished. Working Best: {best_params}. Final: Angle={final_angle:.2f}, Length={final_length:.2f}")
    return final_angle, final_length

def main():
    args = parse_args()
    
    img = cv2.imread(args.input)
    if img is None:
        print(f"Error: Could not read image {args.input}")
        sys.exit(1)
        
    try:
        output_img = process_image(img, args.threshold, args.min_length, args.reduce_length, args.iterations)
        cv2.imwrite(args.output, output_img)
        print(f"Saved result to {args.output}")
    except ValueError as e:
        print(e)
        sys.exit(1)

if __name__ == "__main__":
    main()
