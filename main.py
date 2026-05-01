import os
import argparse
import cv2
import matplotlib.pyplot as plt
from src.core import read_image, localize_iris, normalize_iris, extract_features, calculate_hamming_distance
from src.database import IrisDatabase

def plot_results(img, pupil_circle, iris_circle, normalized, output_name):
    """Plot results and save as images."""
    os.makedirs("images", exist_ok=True)
    
    output_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.circle(output_img, (pupil_circle[0], pupil_circle[1]), pupil_circle[2], (0, 0, 255), 2)
    cv2.circle(output_img, (iris_circle[0], iris_circle[1]), iris_circle[2], (0, 255, 0), 2)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Localization")
    plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title("Normalization (with Masking preparation)")
    plt.imshow(normalized, cmap='gray', aspect='auto')
    plt.axis('off')
    
    plt.tight_layout()
    save_path = os.path.join("images", f"{output_name}.png")
    plt.savefig(save_path)
    plt.close()
    return save_path

def process_single_image(image_path, output_name="result"):
    """Complete pipeline returning (iris_code, mask)"""
    print(f"\n[+] Processing image: {image_path}")
    img = read_image(image_path)
    print(" - Image read successfully.")
    
    pupil_circle, iris_circle = localize_iris(img)
    print(f" - Pupil: {pupil_circle}, Iris: {iris_circle}")
    
    normalized_img, mask = normalize_iris(img, pupil_circle, iris_circle)
    print(" - Normalization and noise mask generation completed.")
    
    iris_code, expanded_mask = extract_features(normalized_img, mask)
    print(f" - IrisCode extracted (shape: {iris_code.shape}).")
    
    plot_results(img, pupil_circle, iris_circle, normalized_img, output_name)
    return iris_code, expanded_mask

def main():
    parser = argparse.ArgumentParser(description="Production-Ready Iris Authentication System")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # Enroll Command
    parser_enroll = subparsers.add_parser("enroll", help="Enroll a new user into the database")
    parser_enroll.add_argument("user_id", type=str, help="User ID (e.g., Name)")
    parser_enroll.add_argument("image", type=str, help="Path to the iris image")

    # Authenticate Command
    parser_auth = subparsers.add_parser("authenticate", help="Authenticate an unknown user")
    parser_auth.add_argument("image", type=str, help="Path to the iris image for authentication")
    parser_auth.add_argument("--threshold", type=float, default=0.32, help="Acceptance threshold (default: 0.32)")

    # Process Command
    parser_process = subparsers.add_parser("process", help="Analyze an image and extract features without saving")
    parser_process.add_argument("image", type=str, help="Path to the iris image")

    args = parser.parse_args()
    db = IrisDatabase(db_path="db")

    try:
        if args.command == "enroll":
            print("=======================================")
            print(f"Starting enrollment for user: {args.user_id}")
            print("=======================================")
            code, mask = process_single_image(args.image, output_name=f"enroll_{args.user_id}")
            saved_path = db.enroll_user(args.user_id, code, mask)
            print(f"\n[SUCCESS] Iris footprint enrolled successfully at: {saved_path}")
            
        elif args.command == "authenticate":
            print("=======================================")
            print("Starting Authentication Process")
            print("=======================================")
            code, mask = process_single_image(args.image, output_name="auth_attempt")
            
            print("\n - Searching the database...")
            matched_user, min_dist = db.identify_user(code, mask, args.threshold)
            
            print("\n=======================================")
            print(f"[*] Min Hamming Distance: {min_dist:.4f}")
            print(f"[*] Threshold: {args.threshold}")
            
            if matched_user:
                print(f"\n[SUCCESS] ACCESS GRANTED")
                print(f"    Welcome, {matched_user}!")
            else:
                print(f"\n[FAILED] ACCESS DENIED")
                print("    Identity not registered or image mismatch.")
            print("=======================================")
            
        elif args.command == "process":
            output_name = os.path.splitext(os.path.basename(args.image))[0] + "_processed"
            process_single_image(args.image, output_name)
            
    except Exception as e:
        print(f"\n[!] Execution Error: {e}")

if __name__ == "__main__":
    main()
