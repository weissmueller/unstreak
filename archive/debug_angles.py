import cv2
import numpy as np
from archive.unstreak import estimate_psf_parameters, create_motion_kernel

def test_angle_consistency():
    # Create an image with a single horizontal line (Angle 90 degrees? Or 0?)
    # In image coords (y, x):
    # Horizontal line: y is constant.
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    # Draw line from (20, 50) to (80, 50) -> Horizontal
    cv2.line(img, (20, 50), (80, 50), (255, 255, 255), 2)
    
    print("Testing Horizontal Line (Detected as 90 or 0/180?)")
    angle, length = estimate_psf_parameters(img, threshold=50, min_length=10)
    print(f"Detected: Angle={angle}, Length={length}")
    
    # Create kernel from this detected angle
    kernel = create_motion_kernel(length, angle)
    
    # Visualize kernel properties
    print(f"Kernel center value: {kernel[kernel.shape[0]//2, kernel.shape[1]//2]}")
    # Check if kernel spreads horizontally
    # Sum along columns (projection to X) vs Sum along rows (projection to Y)
    proj_x = np.sum(kernel, axis=0)
    proj_y = np.sum(kernel, axis=1)
    
    print(f"Kernel spread X: {np.count_nonzero(proj_x > 0.01)}")
    print(f"Kernel spread Y: {np.count_nonzero(proj_y > 0.01)}")
    
    # Test Vertical Line
    img_v = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.line(img_v, (50, 20), (50, 80), (255, 255, 255), 2)
    print("\nTesting Vertical Line")
    angle_v, length_v = estimate_psf_parameters(img_v, threshold=50, min_length=10)
    print(f"Detected: Angle={angle_v}, Length={length_v}")
    
    kernel_v = create_motion_kernel(length_v, angle_v)
    proj_x_v = np.sum(kernel_v, axis=0)
    proj_y_v = np.sum(kernel_v, axis=1)
    print(f"Kernel spread X: {np.count_nonzero(proj_x_v > 0.01)}")
    print(f"Kernel spread Y: {np.count_nonzero(proj_y_v > 0.01)}")

    # Test 45 degrees
    img_45 = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.line(img_45, (20, 20), (80, 80), (255, 255, 255), 2)
    print("\nTesting 45 Degree Line (Top-Left to Bottom-Right)")
    angle_45, length_45 = estimate_psf_parameters(img_45, threshold=50, min_length=10)
    print(f"Detected: Angle={angle_45}, Length={length_45}")
    
    kernel_45 = create_motion_kernel(length_45, angle_45)
    # Just check if it's diagonal

if __name__ == "__main__":
    test_angle_consistency()
