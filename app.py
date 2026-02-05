import os
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_from_directory, Response
from werkzeug.utils import secure_filename
from src.table_extractor import TableExtractor
import cv2
import base64
import json
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'data/input'
app.config['OUTPUT_FOLDER'] = 'data/output'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
VISION_KEY_PATH = r"d:\table_ocr\key.json"
extractor = TableExtractor(key_path=VISION_KEY_PATH)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/image/<filename>')
def get_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return jsonify({'filename': filename})

@app.route('/process', methods=['POST'])
def process_image():
    data = request.json
    filename = data.get('filename', 'BankStatementChequing.png')
    threshold = int(data.get('threshold', 5))
    manual_anchor = data.get('manual_anchor') # {x: , y: }
    end_anchor_input = data.get('end_anchor') # {x: , y: }
    
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    try:
        # Using PSM 11 for sparse text/word selection as requested
        words, shape = extractor.get_words(image_path)
        
        anchor = None
        if manual_anchor:
            # Find the word closest to the manual coordinates
            mx, my = manual_anchor['x'], manual_anchor['y']
            best_dist = float('inf')
            for w in words:
                cx, cy = w['x'] + w['w']//2, w['y'] + w['h']//2
                dist = (cx - mx)**2 + (cy - my)**2
                if dist < best_dist:
                    best_dist = dist
                    anchor = w
        else:
            # Fully automatic detection
            auto_anchor, auto_end = extractor.auto_detect_table(words, threshold_px=threshold)
            if auto_anchor:
                anchor = auto_anchor
                # If we found an auto_anchor, we also potentially have an auto_end
                # This will be passed to get_robust_grid below
                if not end_anchor_input:
                    end_anchor = auto_end
        
        end_anchor = None
        if end_anchor_input:
            emx, emy = end_anchor_input['x'], end_anchor_input['y']
            best_dist = float('inf')
            for w in words:
                cx, cy = w['x'] + w['w']//2, w['y'] + w['h']//2
                dist = (cx - emx)**2 + (cy - emy)**2
                if dist < best_dist:
                    best_dist = dist
                    end_anchor = w

        if not anchor:
            return jsonify({
                'words': words,
                'image_height': shape[0],
                'image_width': shape[1],
                'status': 'awaiting_start_anchor'
            })
            
        # Use the new robust grid estimation logic with optional end anchor
        col_lines, row_lines, merged_rows, table_left, auto_end_anchor = extractor.get_robust_grid(words, anchor, end_anchor=end_anchor, threshold_px=threshold)
        
        return jsonify({
            'words': words,
            'anchor': anchor,
            'end_anchor': end_anchor or auto_end_anchor,
            'is_auto_end': end_anchor is None and auto_end_anchor is not None,
            'merged_rows': merged_rows,
            'col_lines': col_lines,
            'row_lines': row_lines,
            'table_left': table_left,
            'image_height': shape[0],
            'image_width': shape[1],
            'status': 'processed'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/export', methods=['POST'])
def export_table():
    data = request.json
    filename = data.get('filename')
    manual_anchor = data.get('manual_anchor')
    end_anchor = data.get('end_anchor')
    threshold = int(data.get('threshold', 10))
    col_lines = data.get('col_lines', [])
    row_lines = data.get('row_lines', [])
    table_left = data.get('table_left')
    include_mapped = data.get('include_mapped', True)
    include_raw = data.get('include_raw', True)
    include_vision = data.get('include_vision', True)
    extra_cells = data.get('extra_cells', [])
    
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    def generate():
        try:
            # Re-fetch words if needed or just use the lines provided
            # We need the anchor object
            words, _ = extractor.get_words(image_path)
            
            # Find start and end anchor objects from the words list
            anchor_obj = None
            end_anchor_obj = None
            
            for w in words:
                cx, cy = w['x'] + w['w']//2, w['y'] + w['h']//2
                
                # Check start anchor
                if manual_anchor and anchor_obj is None:
                    if abs(cx - manual_anchor['x']) < 10 and abs(cy - manual_anchor['y']) < 10:
                        anchor_obj = w
                
                # Check end anchor
                if end_anchor and end_anchor_obj is None:
                    if abs(cx - end_anchor['x']) < 10 and abs(cy - end_anchor['y']) < 10:
                        end_anchor_obj = w
                
                if (not manual_anchor or anchor_obj) and (not end_anchor or end_anchor_obj):
                    break
            
            if not anchor_obj:
                yield f"data: {json.dumps({'error': 'Start anchor not found during export'})}\n\n"
                return

            # Use the full objects for grid detection
            _, _, final_merged_rows, _ = extractor.get_robust_grid(words, anchor_obj, end_anchor=end_anchor_obj, threshold_px=threshold)
            
            headers = []
            if final_merged_rows:
                headers = [w['text'] for w in final_merged_rows[0]]

            # Use the generator for real-time progress
            generator = extractor.extract_cell_data(
                image_path, anchor_obj, col_lines, row_lines, table_left, headers=headers,
                include_mapped=include_mapped, include_raw=include_raw, include_vision=include_vision,
                extra_cells=extra_cells
            )
            
            data_rows = None
            for curr, total, status_msg, result in generator:
                if result is None:
                    # Technical progress update
                    progress = (curr / total) * 100 if total > 0 else 0
                    yield f"data: {json.dumps({'progress': progress, 'status_msg': status_msg})}\n\n"
                elif isinstance(result, dict) and 'error' in result:
                    # Error from generator
                    yield f"data: {json.dumps({'error': result['error']})}\n\n"
                    return
                else:
                    # Final result
                    data_rows = result
            
            if result and isinstance(result, dict) and 'mapped' in result:
                ts = int(time.time())
                
                mapped_filename = None
                if include_mapped and 'mapped' in result and result['mapped'] is not None:
                    df_mapped = pd.DataFrame(result['mapped'])
                    if headers and len(headers) == df_mapped.shape[1]:
                        df_mapped.columns = headers
                        header_option = True
                    else:
                        header_option = False
                    mapped_filename = f"mapped_export_{ts}.xlsx"
                    mapped_path = os.path.join(app.config['OUTPUT_FOLDER'], mapped_filename)
                    df_mapped.to_excel(mapped_path, index=False, header=header_option)

                raw_filename = None
                if include_raw and 'raw' in result and result['raw'] is not None:
                    df_raw = pd.DataFrame(result['raw'])
                    raw_filename = f"raw_ai_export_{ts}.xlsx"
                    raw_path = os.path.join(app.config['OUTPUT_FOLDER'], raw_filename)
                    df_raw.to_excel(raw_path, index=False)
                
                vision_filename = None
                if include_vision and 'vision' in result and result['vision'] is not None:
                    df_vision = pd.DataFrame(result['vision'])
                    if headers and len(headers) == df_vision.shape[1]:
                        df_vision.columns = headers
                        header_option = True
                    else:
                        header_option = False
                    vision_filename = f"vision_mapped_export_{ts}.xlsx"
                    vision_path = os.path.join(app.config['OUTPUT_FOLDER'], vision_filename)
                    df_vision.to_excel(vision_path, index=False, header=header_option)
                
                yield f"data: {json.dumps({'progress': 100, 'complete': True, 'mapped_url': f'/download/{mapped_filename}' if mapped_filename else None, 'raw_url': f'/download/{raw_filename}' if raw_filename else None, 'vision_url': f'/download/{vision_filename}' if vision_filename else None})}\n\n"
            else:
                yield f"data: {json.dumps({'error': 'Extraction failed to produce data'})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    headers = {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'X-Accel-Buffering': 'no',
        'Connection': 'keep-alive'
    }
    return Response(generate(), headers=headers)

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
