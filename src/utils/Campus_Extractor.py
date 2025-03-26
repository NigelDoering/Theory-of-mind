import fitz  # PyMuPDF
import pandas as pd
import re
from collections import defaultdict

def extract_building_data(pdf_path, img_width=5760, img_height=3240):
    doc = fitz.open(pdf_path)
    
    # ===== Page 1: Coordinate Extraction =====
    page1 = doc[0]
    pdf_width = page1.rect.width
    pdf_height = page1.rect.height
    
    # Extract text with precise bounding boxes
    text_blocks = page1.get_text("blocks")
    
    # Filter and process numeric building IDs
    building_coords = defaultdict(dict)
    pattern = re.compile(r'^\d{1,3}$')  # Match 1-3 digit numbers
    
    for block in text_blocks:
        text = block[4].strip()
        if pattern.match(text):
            try:
                building_id = int(text)
                if 1 <= building_id <= 249:
                    # Convert PDF coordinates to image pixels
                    x_center = (block[0] + block[2]) / 2
                    y_center = (block[1] + block[3]) / 2
                    
                    x_img = int((x_center / pdf_width) * img_width)
                    y_img = int((y_center / pdf_height) * img_height)
                    
                    # Store coordinates with text bounds for validation
                    building_coords[building_id] = {
                        'x': x_img,
                        'y': y_img,
                        'bounds': (block[0], block[1], block[2], block[3])
                    }
            except ValueError:
                continue

    # ===== Page 2: Name Extraction =====
    page2 = doc[1]
    raw_text = page2.get_text()
    
    # Advanced pattern matching for building names
    name_pattern = re.compile(
        r'(\d{1,3})[\s\W]+(.*?)(?=\n\d{1,3}\b|\Z)', 
        re.DOTALL | re.MULTILINE
    )
    
    building_names = {}
    for match in name_pattern.finditer(raw_text):
        building_id = int(match.group(1))
        building_name = ' '.join(match.group(2).split()).strip()
        building_names[building_id] = building_name

    # ===== Data Integration =====
    df = pd.DataFrame.from_dict(building_coords, orient='index')
    df.index.name = 'building_id'
    df.reset_index(inplace=True)
    
    # Merge names with coordinates
    df['name'] = df['building_id'].map(building_names)
    
    # Post-processing
    df = df[['building_id', 'x', 'y', 'name']]
    df.dropna(subset=['x', 'y'], inplace=True)
    df['name'].fillna('Unknown Building', inplace=True)
    df.sort_values('building_id', inplace=True)
    
    # Coordinate validation
    df = df[
        (df['x'].between(0, img_width)) & 
        (df['y'].between(0, img_height))
    ]
    
    return df.reset_index(drop=True)

# Execution and output
if __name__ == "__main__":
    df = extract_building_data("/home/rahm/Theory-of-mind/resources/UCSD_Campus_Detailed.pdf")
    
    # Save results
    df.to_csv("ucsd_building_dataset.csv", index=False)
    
    print(f"Successfully extracted {len(df)} buildings")
    print("Sample output:")
    print(df.head(10))