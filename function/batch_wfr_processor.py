import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import datetime
import random
import pickle
import glob

# Add project root to sys.path to import modules
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# Import tsv pattern generator
try:
    from function.tsv_memory import process_chip_list
except ImportError:
    # Try relative import if running directly from function folder
    from tsv_memory import process_chip_list

# ==========================================
# Configuration
# ==========================================
DEFAULT_DIRS = [
    r"C:\Users\Admin\Pictures\wafer_img\Center",
    r"C:\Users\Admin\Pictures\wafer_img\Donut",
    r"C:\Users\Admin\Pictures\wafer_img\Edge-Loc",
    r"C:\Users\Admin\Pictures\wafer_img\Edge-Ring",
    r"C:\Users\Admin\Pictures\wafer_img\Loc",
    r"C:\Users\Admin\Pictures\wafer_img\Near-Full",
    r"C:\Users\Admin\Pictures\wafer_img\None",
    r"C:\Users\Admin\Pictures\wafer_img\Random",
    r"C:\Users\Admin\Pictures\wafer_img\Scratch"
]

MODEL_PATH = os.path.join(project_root, 'models', 'wafer_classifier.pth')
SAVE_DIR = os.path.join(project_root, 'models')
LABELS = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Near-full', 'Random', 'Scratch', 'None']
NONE_VARIANTS = ['None']

# ==========================================
# Model Definition (Must match training)
# ==========================================
class WaferClassifier(nn.Module):
    def __init__(self, num_classes=9):
        super(WaferClassifier, self).__init__()
        try:
            self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        except:
            self.model = models.resnet18(pretrained=True)
            
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)

def load_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = WaferClassifier()
    
    if not os.path.exists(model_path):
        print(f"[Error] Model file not found: {model_path}")
        return None

    try:
        loaded_obj = torch.load(model_path, map_location=device)
        if isinstance(loaded_obj, dict):
            state_dict = loaded_obj['state_dict'] if 'state_dict' in loaded_obj else loaded_obj
            # Handle possible prefix issues (e.g. 'model.')
            first_key = next(iter(state_dict.keys()))
            if not first_key.startswith('model.') and hasattr(model, 'model'):
                 model.model.load_state_dict(state_dict, strict=False)
            else:
                model.load_state_dict(state_dict, strict=False)
        else:
            model = loaded_obj
    except Exception as e:
        print(f"[Error] Failed to load model: {e}")
        return None
        
    model.to(device)
    model.eval()
    return model

# ==========================================
# Helper Functions
# ==========================================
def determine_grade(defect_density, confidence):
    if defect_density <= 0.05: base_grade = 'A'
    elif defect_density <= 0.20: base_grade = 'B'
    elif defect_density < 0.70: base_grade = 'C'
    else: return 'F'

    if confidence < 0.6:
        if base_grade == 'A': return 'B'
        if base_grade == 'B': return 'C'
        if base_grade == 'C': return 'F'
    
    return base_grade

def predict_failure(model, img_gray_32x32):
    if model is None:
        return "Unknown", 0.0

    img_float = img_gray_32x32.astype(np.float32) / 255.0
    tensor = torch.tensor(img_float).unsqueeze(0).unsqueeze(0).float()
    
    device = next(model.parameters()).device
    tensor = tensor.to(device)
    
    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1)
        max_prob, predicted_idx = torch.max(probs, 1)
        label_idx = predicted_idx.item()
        confidence = max_prob.item()
        
    predicted_label = "Unknown"
    if 0 <= label_idx < len(LABELS):
        predicted_label = LABELS[label_idx]
        
    if predicted_label in NONE_VARIANTS:
        predicted_label = "None"
        
    return predicted_label, confidence

def process_single_image(file_path, model, batch_id):
    # Read image
    try:
        # Handle korean path just in case, though cv2 has issues with it. 
        # Using numpy fromfile for safer read
        img_array = np.fromfile(file_path, np.uint8)
        original_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if original_img is None:
            return None
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

    # Generate pseudo Lot Name from filename
    filename = os.path.splitext(os.path.basename(file_path))[0]
    # Sanitizing filename for compatibility if needed, using as lot_name
    lot_name = filename 

    # Preprocessing
    target_size = (32, 32)
    gray_full = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY) if len(original_img.shape) == 3 else original_img
    
    resized_gray = cv2.resize(gray_full.astype('float32'), target_size, interpolation=cv2.INTER_NEAREST)
    resized_gray = resized_gray.astype(int)

    # Quantization (0:BG, 1:Good, 2:Defect)
    quantized_map = np.zeros_like(resized_gray, dtype=np.uint8)
    quantized_map[resized_gray > 60] = 1   # Good
    quantized_map[resized_gray > 170] = 2  # Defect

    # Model Inference
    failure_type, confidence = predict_failure(model, resized_gray)

    # Chip Extraction
    rows, cols = quantized_map.shape
    chips_status_1 = []
    chips_status_2 = []
    now = datetime.datetime.now()

    for y in range(rows):
        for x in range(cols):
            status_val = int(quantized_map[y, x])
            if status_val == 0: continue
            
            chip_raw = {
                "chip_uid": f"{lot_name}X{x}Y{y}D{status_val}",
                "lot_name": lot_name,
                "tsv_coordinate": f"{x},{y}",
                "coor_x": str(x),
                "coor_y": str(y),
                "tsv_status": status_val,
                "die_status": status_val,
                "failure_type": failure_type,
                "created_at": now
            }
            
            if status_val == 1:
                chips_status_1.append(chip_raw)
            elif status_val == 2:
                chips_status_2.append(chip_raw)

    # Sampling (Max 25 each)
    sample_size = 25
    selected_chips_1 = random.sample(chips_status_1, min(len(chips_status_1), sample_size))
    selected_chips_2 = random.sample(chips_status_2, min(len(chips_status_2), sample_size))
    
    target_chips = selected_chips_1 + selected_chips_2

    # Add TSV Patterns
    processed_chips = process_chip_list(target_chips)
    
    # Calculate stats for display
    die_count = int(np.sum(quantized_map > 0))
    defect_count = int(np.sum(quantized_map == 2))
    defect_density = defect_count / die_count if die_count > 0 else 0.0
    grade = determine_grade(defect_density, confidence)
    
    print(f"   -> [Processed] {filename}: Type={failure_type}, Grade={grade}, Chips={len(processed_chips)}")
    
    return processed_chips

# ==========================================
# Main Execution
# ==========================================
def main():
    print("=== Wafer Batch Processor (Local) ===")
    
    # Check Model
    print(f"Loading model from: {MODEL_PATH}")
    model = load_model(MODEL_PATH)
    if model is None:
        print("CRITICAL: Model load failed. Exiting.")
        return

    # Input Directories
    input_dirs = DEFAULT_DIRS
    print(f"Processing default directories ({len(input_dirs)} locations)...")

    # Processing
    all_chip_data = []
    total_wafers = 0
    
    batch_timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    batch_id = f"BATCH_{batch_timestamp}"
    
    print(f"\nStarting Batch Analysis [{batch_id}]...")
    
    for folder in input_dirs:
        print(f"\nScanning folder: {folder}")
        # Support common image formats
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
            image_files.extend(glob.glob(os.path.join(folder, ext)))
        
        print(f"Found {len(image_files)} images.")
        
        for idx, file_path in enumerate(image_files):
            # process
            chips = process_single_image(file_path, model, batch_id)
            if chips:
                all_chip_data.extend(chips)
                total_wafers += 1

    # Save Result
    if all_chip_data:
        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)
            
        pkl_filename = f"{batch_id}.pkl"
        pkl_path = os.path.join(SAVE_DIR, pkl_filename)
        
        try:
            with open(pkl_path, 'wb') as f:
                pickle.dump(all_chip_data, f)
            print(f"\n[SUCCESS] Batch processing complete!")
            print(f"Total Wafers: {total_wafers}")
            print(f"Total Chips Collected: {len(all_chip_data)}")
            print(f"Saved to: {pkl_path}")
        except Exception as e:
            print(f"\n[ERROR] Failed to save pickle file: {e}")
    else:
        print("\n[WARNING] No chips were processed.")

if __name__ == "__main__":
    main()
