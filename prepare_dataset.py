import os
import shutil
from pathlib import Path
import glob

def prepare_data():
    base_dir = Path(r"c:\Users\muham\.gemini\antigravity\scratch\xai-gyn")
    veri_dir = base_dir / "veri setleri"
    
    # Text file
    txt_file = veri_dir / "celler_150200-1.txt"
    if not txt_file.exists():
        print(f"Error: {txt_file} does not exist.")
        return
        
    repo_dir = veri_dir / "celler_150200"
    if not repo_dir.exists():
        print("Error: repo directory not found.")
        return
    print(f"Sourcing images from: {repo_dir}")
    
    # Target directories
    processed_dir = base_dir / "data" / "processed"
    benign_dir = processed_dir / "benign"
    malign_dir = processed_dir / "malign"
    
    benign_dir.mkdir(parents=True, exist_ok=True)
    malign_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse txt file
    with open(txt_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    found_count = 0
    missing_count = 0
    
    # Format appears to be: 1,"AIS01001","M33010","TRUE",40.0000... 
    for line in lines:
        parts = line.strip().split(',')
        if len(parts) > 3:
            img_id = parts[1].strip('"')
            label_str = parts[3].strip('"').upper()
            
            # TRUE -> malign (assuming TRUE means abnormality), FALSE -> benign
            label = "malign" if label_str == "TRUE" else "benign"
            
            # Find the image
            possible_images = list(repo_dir.glob(f"{img_id}.*"))
            if possible_images:
                img_path = possible_images[0]
                target_path = processed_dir / label / img_path.name
                shutil.copy2(img_path, target_path)
                found_count += 1
            else:
                missing_count += 1
                
    print(f"Processing complete. Found and copied {found_count} images. Missing {missing_count} images.")
    print(f"Benign count: {len(list(benign_dir.glob('*')))}")
    print(f"Malign count: {len(list(malign_dir.glob('*')))}")

if __name__ == "__main__":
    prepare_data()
