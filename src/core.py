import cv2
import numpy as np

def read_image(image_path):
    """Read image supporting Unicode paths on Windows."""
    try:
        with open(image_path, "rb") as f:
            chunk = f.read()
        chunk_arr = np.frombuffer(chunk, dtype=np.uint8)
        img = cv2.imdecode(chunk_arr, cv2.IMREAD_GRAYSCALE)
    except Exception as e:
        raise FileNotFoundError(f"Failed to read image: {e}")
        
    if img is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    return img

def localize_iris(image):
    """Localize pupil and iris using dynamic scaling and radial gradients."""
    blurred = cv2.GaussianBlur(image, (9, 9), 0)
    h, w = image.shape
    min_dim = min(h, w)
    
    # 1. Dynamic Pupil Detection
    thresh_val = np.percentile(blurred, 2)
    _, thresh = cv2.threshold(blurred, thresh_val + 10, 255, cv2.THRESH_BINARY_INV)
    
    kernel = np.ones((5,5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [c for c in contours if cv2.contourArea(c) > min_dim]
    if not valid_contours:
        valid_contours = contours
        
    if not valid_contours:
        raise ValueError("Pupil not found in the image.")
        
    best_contour = max(valid_contours, key=cv2.contourArea)
    (px, py), pr = cv2.minEnclosingCircle(best_contour)
    px, py, pr = int(px), int(py), int(pr)
    
    # 2. Dynamic Iris Detection using Radial Gradient
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)
    
    max_grad = -1
    best_r = pr
    
    r_min = pr + int(min_dim * 0.05)
    r_max = int(min_dim * 0.45)
    
    theta = np.linspace(0, 2 * np.pi, 360)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    
    for r in range(r_min, r_max, 2):
        x_pts = np.clip(np.int32(px + r * cos_t), 0, w - 1)
        y_pts = np.clip(np.int32(py + r * sin_t), 0, h - 1)
        
        grad_sum = 0
        valid_pts = 0
        for i in range(360):
            # Focus on left and right regions, ignore eyelids
            angle = theta[i]
            if (angle < np.pi/4 or angle > 7*np.pi/4) or (angle > 3*np.pi/4 and angle < 5*np.pi/4):
                gx = sobelx[y_pts[i], x_pts[i]]
                gy = sobely[y_pts[i], x_pts[i]]
                radial_grad = gx * cos_t[i] + gy * sin_t[i]
                grad_sum += radial_grad
                valid_pts += 1
                
        avg_grad = grad_sum / valid_pts if valid_pts > 0 else 0
        if avg_grad > max_grad:
            max_grad = avg_grad
            best_r = r
            
    pupil_circle = (px, py, pr)
    iris_circle = (px, py, best_r)
    
    return pupil_circle, iris_circle

def normalize_iris(image, pupil_circle, iris_circle, width=512, height=64):
    """Normalize iris using rubber-sheet model and create noise mask."""
    pupil_x, pupil_y, pupil_r = pupil_circle
    iris_x, iris_y, iris_r = iris_circle
    
    normalized_img = np.zeros((height, width), dtype=np.uint8)
    mask = np.ones((height, width), dtype=np.uint8) # 1 means valid data
    
    theta = np.linspace(0, 2 * np.pi, width)
    
    for i in range(width):
        t = theta[i]
        x_pupil = pupil_x + pupil_r * np.cos(t)
        y_pupil = pupil_y + pupil_r * np.sin(t)
        
        x_iris = iris_x + iris_r * np.cos(t)
        y_iris = iris_y + iris_r * np.sin(t)
        
        for j in range(height):
            r_ratio = j / float(height - 1)
            x = int(x_pupil + r_ratio * (x_iris - x_pupil))
            y = int(y_pupil + r_ratio * (y_iris - y_pupil))
            
            if 0 <= y < image.shape[0] and 0 <= x < image.shape[1]:
                pixel_val = image[y, x]
                normalized_img[j, i] = pixel_val
                
                # Exclude bright reflections and dark eyelashes
                if pixel_val > 240 or pixel_val < 15:
                    mask[j, i] = 0
            else:
                mask[j, i] = 0
                
    return normalized_img, mask

def extract_features(normalized_image, mask):
    """Extract features using 2D Gabor filters to generate IrisCode."""
    ksize = 21
    sigma = 5.0
    theta = np.pi / 4
    lamda = 10.0
    gamma = 0.5
    psi = 0.0
    
    gabor_kernel_real = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, psi, ktype=cv2.CV_32F)
    gabor_kernel_imag = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, psi + np.pi/2, ktype=cv2.CV_32F)
    
    filtered_real = cv2.filter2D(normalized_image, cv2.CV_32F, gabor_kernel_real)
    filtered_imag = cv2.filter2D(normalized_image, cv2.CV_32F, gabor_kernel_imag)
    
    code_real = (filtered_real > 0).astype(np.uint8)
    code_imag = (filtered_imag > 0).astype(np.uint8)
    
    iris_code = np.concatenate((code_real, code_imag), axis=0)
    expanded_mask = np.concatenate((mask, mask), axis=0)
    
    return iris_code, expanded_mask

def calculate_hamming_distance(code1, mask1, code2, mask2):
    """Calculate minimum Hamming distance with rotational shifts."""
    if code1.shape != code2.shape:
        raise ValueError("IrisCode dimensions do not match.")
        
    shifts = range(-8, 9)
    min_distance = float('inf')
    
    for shift in shifts:
        shifted_code2 = np.roll(code2, shift, axis=1)
        shifted_mask2 = np.roll(mask2, shift, axis=1)
        
        combined_mask = np.logical_and(mask1, shifted_mask2)
        valid_bits_count = np.sum(combined_mask)
        
        if valid_bits_count == 0:
            continue
            
        xor_result = np.bitwise_xor(code1, shifted_code2)
        masked_xor = np.logical_and(xor_result, combined_mask)
        
        distance = np.sum(masked_xor) / float(valid_bits_count)
        
        if distance < min_distance:
            min_distance = distance
            
    return min_distance if min_distance != float('inf') else 1.0
