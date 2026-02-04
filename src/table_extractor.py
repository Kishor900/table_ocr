import cv2
import os
import pandas as pd
import numpy as np
import time
import ollama
import base64
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from google.cloud import vision

class TableExtractor:
    def __init__(self, key_path=r"D:\ROBOTICS\allocr\key.json", ocr_model='qwen3-vl:8b-instruct'):
        self.ocr_model = ocr_model
        # Initialize Google Cloud Vision Client
        if os.path.exists(key_path):
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = key_path
            self.vision_client = vision.ImageAnnotatorClient()
        else:
            print(f"Warning: Vision Key not found at {key_path}")
            self.vision_client = None
        
    def get_words(self, image_path):
        """
        Extract words and their bounding boxes using Google Cloud Vision.
        Returns a list of dicts: [{'text': '...', 'x': , 'y': , 'w': , 'h': }, ...]
        """
        if not self.vision_client:
            raise ValueError("Vision client not initialized. Check API Key.")

        with open(image_path, "rb") as image_file:
            content = image_file.read()

        image = vision.Image(content=content)
        response = self.vision_client.text_detection(image=image)
        annotations = response.text_annotations

        if response.error.message:
            raise Exception(f"Vision API Error: {response.error.message}")

        words = []
        # The first annotation is the entire text, the rest are individual words
        for i in range(1, len(annotations)):
            anno = annotations[i]
            # Get bounding box
            vertices = anno.bounding_poly.vertices
            # Vertices are: top-left, top-right, bottom-right, bottom-left
            x = min(v.x for v in vertices)
            y = min(v.y for v in vertices)
            w = max(v.x for v in vertices) - x
            h = max(v.y for v in vertices) - y
            
            words.append({
                'text': anno.description.strip(),
                'x': x,
                'y': y,
                'w': w,
                'h': h,
                'conf': 100  # Vision doesn't provide confidence per word in text_detection easily
            })

        # Get image dimensions
        img = cv2.imread(image_path)
        return words, img.shape

    def find_anchor(self, words, anchor_text="Date"):
        """
        Find the 'Date' word which marks the start of the table.
        """
        for w in words:
            if anchor_text.lower() in w['text'].lower():
                return w
        return None

    def get_table_dimensions(self, words, anchor):
        """
        Determine columns based on words in the same row as the anchor.
        """
        # Threshold for 'same row' (vertical overlap or proximity)
        row_threshold = anchor['h'] // 2
        
        header_row = [w for w in words if abs(w['y'] - anchor['y']) < row_threshold]
        # Sort by x coordinate
        header_row.sort(key=lambda x: x['x'])
        
        return header_row

    def merge_blocks(self, words, anchor, threshold_px=5):
        """
        Merge words horizontally if spacing < threshold_px.
        Focus on words below and to the right of the anchor (including anchor row headers).
        """
        # Filter words that are part of the table (y >= anchor['y'] - margin)
        # We include headers to define column counts.
        margin = anchor['h'] // 2
        table_words = [w for w in words if w['y'] >= anchor['y'] - margin and w['x'] >= anchor['x'] - margin]
        
        # Group by row first
        rows = {}
        for w in table_words:
            # Use a simple bucket for rows based on y coordinate
            row_y = (w['y'] // (anchor['h'] or 10)) * (anchor['h'] or 10)
            if row_y not in rows:
                rows[row_y] = []
            rows[row_y].append(w)
            
        merged_rows = []
        for row_y in sorted(rows.keys()):
            row_words = sorted(rows[row_y], key=lambda x: x['x'])
            if not row_words:
                continue
                
            merged_row = []
            current_block = row_words[0].copy()
            
            for i in range(1, len(row_words)):
                next_word = row_words[i]
                # Check horizontal distance
                dist = next_word['x'] - (current_block['x'] + current_block['w'])
                
                if dist <= threshold_px:
                    # Merge
                    new_x = min(current_block['x'], next_word['x'])
                    new_y = min(current_block['y'], next_word['y'])
                    new_w = max(current_block['x'] + current_block['w'], next_word['x'] + next_word['w']) - new_x
                    new_h = max(current_block['y'] + current_block['h'], next_word['y'] + next_word['h']) - new_y
                    current_block['text'] += " " + next_word['text']
                    current_block['x'], current_block['y'], current_block['w'], current_block['h'] = new_x, new_y, new_w, new_h
                else:
                    merged_row.append(current_block)
                    current_block = next_word.copy()
            
            merged_row.append(current_block)
            merged_rows.append(merged_row)
            
        return merged_rows

        return col_bounds

    def get_robust_grid(self, words, anchor, end_anchor=None, threshold_px=5):
        """
        Estimate rows and columns by averaging block positions.
        """
        margin = anchor['h'] // 2
        
        if end_anchor:
            # Filter words strictly between start and end anchors
            # Use generous margins on BOTH sides to catch centered/shifted headers
            table_words = [w for w in words if 
                           w['y'] >= anchor['y'] - margin and 
                           (w['y'] + w['h']) <= (end_anchor['y'] + end_anchor['h']) + margin and
                           w['x'] >= anchor['x'] - 100 and
                           (w['x'] + w['w']) <= (end_anchor['x'] + end_anchor['w']) + 100]
        else:
            # Default behavior: be generous to the left to catch data under centered headers
            table_words = [w for w in words if w['y'] >= anchor['y'] - margin and w['x'] >= anchor['x'] - 100]
        
        if not table_words:
            return [], []

        # 1. Row Detection (Grouping by vertical proximity)
        table_words.sort(key=lambda w: w['y'])
        rows = []
        if table_words:
            current_row = [table_words[0]]
            for i in range(1, len(table_words)):
                w = table_words[i]
                # If word overlaps vertically with current row or is very close
                # We use a threshold based on average height
                if w['y'] < current_row[-1]['y'] + current_row[-1]['h'] * 0.7:
                    current_row.append(w)
                else:
                    rows.append(current_row)
                    current_row = [w]
            rows.append(current_row)

        # 2. Merging blocks within rows (as before but more aware)
        final_merged_rows = []
        for row in rows:
            row.sort(key=lambda w: w['x'])
            merged = []
            if row:
                curr = row[0].copy()
                for i in range(1, len(row)):
                    nxt = row[i]
                    if nxt['x'] - (curr['x'] + curr['w']) <= threshold_px:
                        curr['text'] += " " + nxt['text']
                        curr['w'] = max(curr['x'] + curr['w'], nxt['x'] + nxt['w']) - curr['x']
                        curr['h'] = max(curr['h'], nxt['h'])
                    else:
                        merged.append(curr)
                        curr = nxt.copy()
                merged.append(curr)
                final_merged_rows.append(merged)

        # 3. Column Estimation (Transitive Merging)
        all_blocks = [b for row in final_merged_rows for b in row]
        if not all_blocks:
            return [], [], [], anchor['x']

        # Start with each block in its own column
        columns = [[b] for b in all_blocks]
        
        # Merge columns that share blocks (transitive closure)
        # Actually, merge columns if any block in A overlaps with the bounds of column B
        merged = True
        while merged:
            merged = False
            new_columns = []
            skip_indices = set()
            
            for i in range(len(columns)):
                if i in skip_indices: continue
                
                current_col = columns[i]
                current_x_start = min(c['x'] for c in current_col)
                current_x_end = max(c['x'] + c['w'] for c in current_col)
                
                # Check against all other columns
                merge_target = i
                for j in range(i + 1, len(columns)):
                    if j in skip_indices: continue
                    
                    target_col = columns[j]
                    target_x_start = min(c['x'] for c in target_col)
                    target_x_end = max(c['x'] + c['w'] for c in target_col)
                    
                    # Overlap or proximity check
                    overlap = min(current_x_end, target_x_end) - max(current_x_start, target_x_start)
                    
                    # We merge if there is ANY overlap OR if they are within 4px (Tightened from 10px)
                    if overlap > 0 or abs(current_x_end - target_x_start) < 4 or abs(target_x_end - current_x_start) < 4:
                        # Before merging, check if this merge is "transitive" via any individual block
                        # i.e., Does any block in the entire table bridge these two columns?
                        # For simplicity, we merge if their bounds overlap/touch.
                        current_col.extend(target_col)
                        skip_indices.add(j)
                        merged = True
                        # Update bounds after extension
                        current_x_start = min(c['x'] for c in current_col)
                        current_x_end = max(c['x'] + c['w'] for c in current_col)
                
                new_columns.append(current_col)
            columns = new_columns

        # Filter out noisy columns and sort
        max_blocks = max(len(col) for col in columns) if columns else 1
        filtered_columns = []
        for col in columns:
            has_anchor = any(b['x'] == anchor['x'] and b['y'] == anchor['y'] for b in col)
            if end_anchor:
                has_anchor = has_anchor or any(b['x'] == end_anchor['x'] and b['y'] == end_anchor['y'] for b in col)
            
            col_width = max(c['x'] + c['w'] for c in col) - min(c['x'] for c in col)
            # Keep if it has an anchor OR (meets density threshold AND is not abnormally thin noise)
            if has_anchor or (len(col) >= max_blocks * 0.1 and col_width >= 8):
                filtered_columns.append(col)
        
        columns = filtered_columns
        columns.sort(key=lambda col: min(c['x'] for c in col))

        # Calculate column boundaries with proximity merging
        col_lines = []
        table_left = anchor['x']
        
        if columns:
            table_left = min(c['x'] for c in columns[0])
            for col in columns:
                max_right = max(c['x'] + c['w'] for c in col)
                if col_lines and (max_right - col_lines[-1]) < 5: # Tightened from 10px to 5px
                    col_lines[-1] = max_right
                else:
                    col_lines.append(max_right)

        # 4. Row Estimation (Averaging Y-bounds)
        row_lines = []
        for row in final_merged_rows:
            avg_bottom = sum(b['y'] + b['h'] for b in row) / len(row)
            row_lines.append(avg_bottom)

        return col_lines, row_lines, final_merged_rows, table_left

    def map_words_to_grid(self, words, col_lines, row_lines, table_left, table_top):
        """
        Pure geometric mapping of Vision OCR words to grid cells.
        """
        num_rows = len(row_lines)
        num_cols = len(col_lines)
        grid = [["" for _ in range(num_cols)] for _ in range(num_rows)]
        
        # Sort words by y then x for natural reading order
        sorted_words = sorted(words, key=lambda w: (w['y'], w['x']))
        
        for r_idx in range(num_rows):
            y_start = row_lines[r_idx-1] if r_idx > 0 else table_top
            y_end = row_lines[r_idx]
            
            for c_idx in range(num_cols):
                x_start = col_lines[c_idx-1] if c_idx > 0 else table_left
                x_end = col_lines[c_idx]
                
                cell_words = []
                for w in sorted_words:
                    # Centroid-based assignment
                    cx = w['x'] + w['w'] / 2
                    cy = w['y'] + w['h'] / 2
                    if x_start <= cx <= x_end and y_start <= cy <= y_end:
                        cell_words.append(w['text'])
                
                grid[r_idx][c_idx] = " ".join(cell_words)
        
        return grid

    def extract_cell_data(self, image_path, anchor, col_lines, row_lines, table_left, headers=None):
        """
        Perform OCR by processing the entire table area in one shot using Qwen,
        then mapping the results back to the detected grid.
        Returns a generator of (current_progress, total_steps, status_msg, final_data_rows)
        """
        img = cv2.imread(image_path)
        if img is None:
            yield 0, 1, "Error", {"error": f"Could not read image at {image_path}"}
            return

        x_coords = [table_left] + col_lines
        y_coords = [anchor['y']] + row_lines
        
        num_rows = len(y_coords) - 1
        num_cols = len(x_coords) - 1
        
        # 1. Crop the entire table
        y_start = int(anchor['y'])
        y_end = int(row_lines[-1])
        x_start = int(table_left)
        x_end = int(col_lines[-1])
        
        pad = 5
        roi = img[max(0, y_start-pad):min(img.shape[0], y_end+pad), 
                  max(0, x_start-pad):min(img.shape[1], x_end+pad)]
        
        # Results storage
        final_grid = [["" for _ in range(num_cols)] for _ in range(num_rows)]
        
        TOTAL_STEPS = 100
        yield 10, TOTAL_STEPS, "Preparing Image", None
        
        if not self.ocr_model:
            yield 0, 1, "Error", {"error": "OCR model not specified"}
            return

        try:
            _, buffer = cv2.imencode('.png', roi)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            if headers:
                header_str = ", ".join([f'"{h}"' for h in headers])
                prompt = (
                    f"You are an expert OCR assistant. Extract the tabular data from the image. "
                    f"The table has these columns: {header_str}. "
                    f"Return ONLY a JSON list of objects, where each object represents a row. "
                    f"Use the exact column names as keys: {header_str}. "
                    f"Ensure every row is captured. Use empty strings for missing values. "
                    f"Output ONLY the JSON and nothing else."
                )
            else:
                prompt = (
                    f"You are an expert OCR assistant. Extract the tabular data from the image. "
                    f"Return ONLY a JSON 2D array (list of lists) where each sub-list is a row. "
                    f"The table has approximately {num_rows} rows and {num_cols} columns. "
                    f"Ensure every row has exactly {num_cols} elements. "
                    f"Use empty strings for empty cells. "
                    f"Output ONLY the JSON and nothing else."
                )
            
            yield 20, TOTAL_STEPS, "Hiring Qwen for extraction", None
            yield 40, TOTAL_STEPS, "Qwen is reading... (This may take 30-90s)", None
            
            response = ollama.chat(
                model=self.ocr_model,
                messages=[{'role': 'user', 'content': prompt, 'images': [img_base64]}],
                options={'temperature': 0}
            )
            
            yield 80, TOTAL_STEPS, "Organizing extracted data", None
            
            raw_content = response['message']['content'].strip()
            import re
            import json
            
            # Robust JSON extraction
            json_match = re.search(r'\[\s*\{.*\}\s*\]|\[\s*\[.*\]\s*\]', raw_content, re.DOTALL)
            if json_match:
                extracted_data = json.loads(json_match.group(0))
                
                yield 90, TOTAL_STEPS, "Mapping results to grid", None
                
                if headers and isinstance(extracted_data, list) and len(extracted_data) > 0 and isinstance(extracted_data[0], dict):
                    # Smart Mapping: Map objects to grid using headers
                    for r_idx, row_obj in enumerate(extracted_data):
                        if r_idx < num_rows:
                            for c_idx, h_name in enumerate(headers):
                                if h_name in row_obj:
                                    final_grid[r_idx][c_idx] = str(row_obj[h_name]) if row_obj[h_name] is not None else ""
                else:
                    # Fallback or List-of-Lists Mapping
                    for r_idx, row in enumerate(extracted_data):
                        if r_idx < num_rows:
                            if isinstance(row, list):
                                for c_idx, val in enumerate(row):
                                    if c_idx < num_cols:
                                        final_grid[r_idx][c_idx] = str(val) if val is not None else ""
                            elif isinstance(row, dict):
                                # Unusual case: list of objects but no headers provided? Try matching keys
                                for c_idx, val in enumerate(row.values()):
                                    if c_idx < num_cols:
                                        final_grid[r_idx][c_idx] = str(val) if val is not None else ""
                
                
                # Vision-Mapped Export (Geometric only)
                # Get words for spatial mapping
                words, _ = self.get_words(image_path)
                vision_grid = self.map_words_to_grid(words, col_lines, row_lines, table_left, anchor['y'])
                
                yield 100, TOTAL_STEPS, "Complete", {
                    "mapped": final_grid, 
                    "raw": extracted_data,
                    "vision": vision_grid
                }
                return
            else:
                raise ValueError(f"Could not parse Qwen response as JSON. Raw response: {raw_content[:200]}...")

        except Exception as e:
            print(f"One-shot extraction failed: {e}")
            yield 0, 100, "Error", {"error": f"Qwen Extraction failed: {str(e)}"}
            return

if __name__ == "__main__":
    # Test script
    extractor = TableExtractor()
    try:
        words, shape = extractor.get_words("data/input/BankStatementChequing.png")
        print(f"Extracted {len(words)} words.")
        anchor = extractor.find_anchor(words)
        if anchor:
            print(f"Found anchor: {anchor}")
            merged = extractor.merge_blocks(words, anchor, threshold_px=10)
            print(f"Detected {len(merged)} rows.")
            lines = extractor.get_column_lines(merged)
            print(f"Column boundaries: {lines}")
        else:
            print("Anchor 'Date' not found.")
    except Exception as e:
        print(f"Error: {e}")
