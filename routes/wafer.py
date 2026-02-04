from flask import Blueprint, request, jsonify
import sys
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import base64
import datetime
import json
import random
import uuid
import pymysql
from urllib.parse import quote
from torchvision import models
import pickle

# TSV 패턴 생성 모듈 import
from function.tsv_memory import process_chip_list

# ==========================================
# 0. 설정 및 초기화
# ==========================================

# 프로젝트 루트 경로 설정
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from db import get_conn

# Blueprint 정의
wafer_bp = Blueprint("wafer", __name__, url_prefix="/wafer")

# 임시 이미지 저장 폴더 (분리형 구조를 위해 필요)
TEMP_FOLDER = os.path.join(project_root, 'temp_images')
if not os.path.exists(TEMP_FOLDER):
    os.makedirs(TEMP_FOLDER)

# Firebase Storage 설정 (수정본 기능)
import firebase_admin
from firebase_admin import credentials, storage
from dotenv import load_dotenv
load_dotenv()

# Firebase 초기화 (이미 초기화되었는지 확인)
try:
    if not firebase_admin._apps:
        cred = credentials.Certificate('wafer-firebase-key.json')
        firebase_admin.initialize_app(cred, {
            'storageBucket': 'wafer-service-a07b3.firebasestorage.app'})
except Exception as e:
    print(f"[Warning] Firebase init error (might be already initialized): {e}")

# ==========================================
# 1. 등급 판정 로직 (기존: 밀도 + 신뢰도 패널티)
# ==========================================

def determine_grade(defect_density, confidence):
    """
    기존 wafer.py의 로직 유지
    """
    # 1. 밀도 기반 1차 판정
    if defect_density <= 0.05: base_grade = 'A'
    elif defect_density <= 0.20: base_grade = 'B'
    elif defect_density < 0.70: base_grade = 'C'
    else: return 'F'

    # 2. 신뢰도 패널티 (0.6 미만이면 등급 하향)
    if confidence < 0.6:
        if base_grade == 'A': return 'B'
        if base_grade == 'B': return 'C'
        if base_grade == 'C': return 'F'
    
    return base_grade

# ==========================================
# 2. AI 모델 및 프로세서 (수정본 + 기존 등급 로직 통합)
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

LABELS = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Near-full', 'Random', 'Scratch', 'None']
NONE_VARIANTS = ['None']

def load_model_instance(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = WaferClassifier()
    
    if not os.path.exists(model_path):
        print(f"[Warning] Model file not found: {model_path}")
        return None

    try:
        loaded_obj = torch.load(model_path, map_location=device)
        if isinstance(loaded_obj, dict):
            state_dict = loaded_obj['state_dict'] if 'state_dict' in loaded_obj else loaded_obj
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

class WaferProcessor:
    def __init__(self, model_path):
        self.model = load_model_instance(model_path)
        self.COLOR_BLACK = (0, 0, 0)       
        self.COLOR_RED = (0, 0, 255)       
        self.COLOR_YELLOW = (0, 255, 255) 

    def predict_failure(self, img_gray_32x32):
        if self.model is None:
            return "Unknown", 0.0

        img_float = img_gray_32x32.astype(np.float32) / 255.0
        tensor = torch.tensor(img_float).unsqueeze(0).unsqueeze(0).float()
        
        device = next(self.model.parameters()).device
        tensor = tensor.to(device)
        
        with torch.no_grad():
            outputs = self.model(tensor)
            probs = F.softmax(outputs, dim=1)
            max_prob, predicted_idx = torch.max(probs, 1)
            label_idx = predicted_idx.item()
            confidence = max_prob.item() # 0.0 ~ 1.0
            
        predicted_label = "Unknown"
        if 0 <= label_idx < len(LABELS):
            predicted_label = LABELS[label_idx]
            
        if predicted_label in NONE_VARIANTS:
            predicted_label = "None"
            
        return predicted_label, confidence

    def process_wafer(self, file_stream, lot_name_input):
        # 1. 이미지 읽기
        file_bytes = np.frombuffer(file_stream.read(), np.uint8)
        original_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if original_img is None:
            raise ValueError("Invalid image file")

        # 2. 전처리 (32x32, Grayscale, Nearest)
        target_size = (32, 32)
        gray_full = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY) if len(original_img.shape) == 3 else original_img
        
        resized_gray = cv2.resize(gray_full.astype('float32'), target_size, interpolation=cv2.INTER_NEAREST)
        resized_gray = resized_gray.astype(int)

        # 3. 로직 매핑 (0:배경, 1:정상, 2:불량)
        quantized_map = np.zeros_like(resized_gray)
        quantized_map[resized_gray > 60] = 1   # Good
        quantized_map[resized_gray > 170] = 2  # Defect

        # 4. 시각화 (Base64)
        color_map = np.zeros((32, 32, 3), dtype=np.uint8)
        color_map[quantized_map == 0] = self.COLOR_BLACK
        color_map[quantized_map == 1] = self.COLOR_RED
        color_map[quantized_map == 2] = self.COLOR_YELLOW
        
        _, buffer = cv2.imencode('.png', color_map)
        png_bytes = buffer.tobytes()
        wafer_map_b64 = base64.b64encode(png_bytes).decode('utf-8')

        # 5. 모델 분석
        failure_type, confidence = self.predict_failure(resized_gray)
        
        # 6. 통계
        die_count = int(np.sum(quantized_map > 0))
        defect_count = int(np.sum(quantized_map == 2))
        defect_density = defect_count / die_count if die_count > 0 else 0.0
        
        # 7. 등급 판정 (기존 로직 사용 - confidence 전달)
        total_grade = determine_grade(defect_density, confidence)
        
        now = datetime.datetime.now()
        
        # 8. Firebase Storage 업로드 (수정본 기능)
        try:
            bucket = storage.bucket()
            blob = bucket.blob(f"wafer_images/{lot_name_input}.png")

            # Firebase Storage download URL 생성
            download_token = str(uuid.uuid4())
            blob.metadata = {"firebaseStorageDownloadTokens": download_token}
            blob.upload_from_string(png_bytes, content_type="image/png")
            # blob.patch() # metadata update

            wafer_map_url = (
                f"https://firebasestorage.googleapis.com/v0/b/{bucket.name}/o/"
                f"{quote(blob.name, safe='')}?alt=media&token={download_token}"
            )
        except Exception as e:
            print(f"[Error] Firebase Upload Failed: {e}")
            # wafer_map_url = None
            # [수정] 이미지가 없으면 DB에 저장하지 않고 에러 처리
            return jsonify({"error": f"Firebase Upload Failed: {str(e)}"}), 500

        # 데이터 구조 준비
        wafer_data = {
            "lot_name": lot_name_input,
            "wafer_map": wafer_map_b64,
            "failure_type": failure_type,
            "confidence": float(confidence),
            "die_count": die_count,
            "defect_count": defect_count,
            "defect_density": defect_density,
            "total_grade": total_grade,
            "created_at": now,
            "img_url": wafer_map_url
        }

        # ---------------------------------------------------------
        # 9. Chip 데이터 추출 및 샘플링 (User Requested Logic)
        # ---------------------------------------------------------
        rows, cols = quantized_map.shape
        chips_status_1 = []
        chips_status_2 = []

        # 1) 전체 칩 좌표 수집
        for y in range(rows):
            for x in range(cols):
                status_val = int(quantized_map[y, x])
                if status_val == 0: continue
                
                # tsv_memory.process_chip_list에서 필요한 포맷으로 미리 구성
                chip_raw = {
                    "chip_uid": f"{lot_name_input}X{x}Y{y}D{status_val}",
                    "lot_name": lot_name_input,
                    "tsv_coordinate": f"{x},{y}",
                    "coor_x": str(x),
                    "coor_y": str(y),
                    "tsv_status": status_val,
                    "die_status": status_val,
                    "failure_type": failure_type, # 패턴 생성에 필요
                    "created_at": now
                }
                
                if status_val == 1:
                    chips_status_1.append(chip_raw)
                elif status_val == 2:
                    chips_status_2.append(chip_raw)

        # 2) 무작위 샘플링 (각 최대 25개)
        sample_size = 25
        
        selected_chips_1 = random.sample(chips_status_1, min(len(chips_status_1), sample_size))
        selected_chips_2 = random.sample(chips_status_2, min(len(chips_status_2), sample_size))
        
        target_chips = selected_chips_1 + selected_chips_2

        # 3) 패턴 주입 (tsv_memory 활용)
        # process_chip_list는 'tsv_status', 'failure_type' 키를 참고하여 'tsv_matrix'를 채워줍니다.
        # 반환값은 tsv_matrix(numpy array)가 포함된 딕셔너리 리스트입니다.
        processed_chips = process_chip_list(target_chips)

        return wafer_data, processed_chips

_PROCESSOR_INSTANCE = None
def get_processor():
    global _PROCESSOR_INSTANCE
    if _PROCESSOR_INSTANCE is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, 'models', 'wafer_classifier.pth')
        
        if not os.path.exists(model_path):
             print(f"!!! CRITICAL: Model not found at {model_path}")
        
        _PROCESSOR_INSTANCE = WaferProcessor(model_path)
    return _PROCESSOR_INSTANCE

# ==========================================
# 3. 라우트 (분리형 구조: Upload -> Analyze -> Get)
# ==========================================

# 메모리상에서 Lot Name과 Batch ID의 매핑을 유지하기 위한 전역 변수
LOT_BATCH_MAP = {}

@wafer_bp.route("/upload", methods=["POST"])
def upload_wafer():
    """
    [1단계] 웨이퍼 업로드 (다중 파일 지원 수정 + Batch ID 생성)
    """
    if 'wafer_image' not in request.files:
        return jsonify({"error": "No file part"}), 400
        
    files = request.files.getlist('wafer_image')
    if not files or files[0].filename == '':
        return jsonify({"error": "No selected file"}), 400

    uploaded_lots = []
    
    # 1. Batch ID 결정 (프론트엔드 우선 -> 없으면 서버 생성)
    batch_id = request.form.get('batch_id')
    if not batch_id:
        now_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        batch_id = f"BATCH_{now_str}"
    
    for file in files:
        # lot_name 생성 (파일마다 고유하게)
        now = datetime.datetime.now()
        chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        random_serial = ''.join(random.choices(chars, k=6))
        lot_name = f"{now.strftime('%y%m%d')}{random_serial}{now.strftime('%H%M')}"
        
        # 임시 폴더에 이미지 저장
        ext = os.path.splitext(file.filename)[1] or '.png'
        # 중복 방지를 위해 lot_name + uuid 일부 추가 등 고려 가능하나, 현재 랜덤 6자리로 충분
        save_path = os.path.join(TEMP_FOLDER, f"{lot_name}{ext}")
        file.save(save_path)
        
        uploaded_lots.append(lot_name)
        
        # [중요] 매핑 저장: 나중에 analyze 할 때 이 lot_name이 오면 batch_id를 찾을 수 있게 함
        LOT_BATCH_MAP[lot_name] = batch_id

    return jsonify({
        "message": f"{len(uploaded_lots)} Wafers uploaded successfully",
        "lot_names": uploaded_lots,
        "batchId": batch_id,
        "status": "Ready for analysis"
    }), 201


@wafer_bp.route("/<lot_name>/analyze", methods=["POST"])
def analyze_wafer(lot_name):
    """
    [2단계] 웨이퍼 분석
    """
    # 쿼리 파라미터로 batch_id 받기 (프론트엔드 전달 우선)
    batch_id = request.args.get('batch_id')
    
    # 전달받은 게 없으면 서버 메모리에서 조회 (Fallback)
    if not batch_id:
        batch_id = LOT_BATCH_MAP.get(lot_name)
    
    # 1. 이미지 찾기
    target_file = None
    for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
        path = os.path.join(TEMP_FOLDER, f"{lot_name}{ext}")
        if os.path.exists(path):
            target_file = path
            break
            
    if not target_file:
        return jsonify({"error": "Image not found"}), 404

    try:
        # 2. 분석 수행 (수정본 Processor 사용)
        processor = get_processor()
        with open(target_file, 'rb') as f:
            wafer_data, chip_list = processor.process_wafer(f, lot_name)

        # 3. DB 저장 (기존 테이블 wafer_data에 저장 - 호환성 유지)
        conn = get_conn()
        cur = conn.cursor()
        try:
            # wafer_data 테이블 (Snake Case 컬럼 매핑)
            cur.execute("""
                INSERT INTO wafer_data
                (lot_name, wafer_map, failure_type, confidence, die_count, defect_count, defect_density, total_grade, created_at) 
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                wafer_data['lot_name'], 
                wafer_data['img_url'], 
                wafer_data['failure_type'], 
                wafer_data['confidence'],
                wafer_data['die_count'],
                wafer_data['defect_count'],
                wafer_data['defect_density'],
                wafer_data['total_grade'],
                wafer_data['created_at']
            ))
            
            wafer_idx = cur.lastrowid

            if chip_list:
                chip_values = []
                for c in chip_list:
                    matrix_val = c.get('tsv_matrix', [])
                    if hasattr(matrix_val, 'tolist'):
                        matrix_str = json.dumps(matrix_val.tolist())
                    else:
                        matrix_str = json.dumps(matrix_val)
                        
                    chip_values.append((
                        wafer_idx,          
                        c['chip_uid'],      
                        matrix_str,         
                        c['failure_type'],   
                        c['coor_x'],        
                        c['coor_y'],        
                        c['die_status'],    
                        c['created_at']     
                    ))
                
                print("DEBUG: Executing chip_data INSERT...")
                chip_sql = """
                    INSERT INTO chip_data 
                    (wafer_idx, chip_uid, tsv_matrix, failure_type, coor_x, coor_y, die_status, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """
                # print(f"DEBUG SQL: {chip_sql}")
                cur.executemany(chip_sql, chip_values)
            
            conn.commit()

            # -------------------------------------------------------
            # [Added] Save chip data to PKL (Batch Append)
            # -------------------------------------------------------
            try:
                save_dir = os.path.join(project_root, 'models')
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                
                # Batch ID가 있으면 통합 저장, 없으면 개별 저장
                if batch_id:
                    pkl_filename = f"{batch_id}.pkl"
                    pkl_path = os.path.join(save_dir, pkl_filename)
                    
                    # 기존 데이터 로드 (Append 방식)
                    existing_data = []
                    # 파일이 존재하면 읽어서 기존 리스트에 추가
                    if os.path.exists(pkl_path):
                        try:
                            with open(pkl_path, 'rb') as f:
                                existing_data = pickle.load(f)
                        except Exception as load_err:
                            print(f"Warning: Failed to load existing batch pkl: {load_err}")
                            existing_data = []
                    
                    # 데이터 합치기
                    combined_data = existing_data + chip_list
                    
                    with open(pkl_path, 'wb') as f:
                        pickle.dump(combined_data, f)
                    print(f"DEBUG: Appended chip data to batch file: {pkl_path} (Total: {len(combined_data)} chips)")
                else:
                     # Batch ID가 없는 경우 (기존 방식 - 개별 저장)
                    pkl_path = os.path.join(save_dir, f"{lot_name}.pkl")
                    with open(pkl_path, 'wb') as f:
                        pickle.dump(chip_list, f)
                    print(f"DEBUG: Saved chip data to individual file: {pkl_path}")

            except Exception as e:
                print(f"Warning: Failed to save pkl file: {e}")

            # -------------------------------------------------------
            # [Cleanup] 분석 완료된 임시 이미지 삭제
            # -------------------------------------------------------
            try:
                if target_file and os.path.exists(target_file):
                    os.remove(target_file)
                    print(f"DEBUG: Deleted temporary image: {target_file}")
            except Exception as e:
                print(f"Warning: Failed to delete temp image: {e}")
            
            return jsonify({
                "message": "Analysis completed",
                "lot_name": lot_name,
                "batchId": batch_id,
                "result": {
                    "failure_type": wafer_data['failure_type'],
                    "confidence": wafer_data['confidence'],
                    "total_grade": wafer_data['total_grade'],
                    "defect_density": wafer_data['defect_density'],
                    "die_count": wafer_data['die_count'],
                    "defect_count": wafer_data['defect_count']
                },
                "img_url": wafer_data['img_url']
            }), 200

        except Exception as e:
            conn.rollback()
            raise e
        finally:
            cur.close()
            conn.close()

    except Exception as e:
        # print(f"DEBUG ERROR: {e}")
        return jsonify({"error": str(e)}), 500

@wafer_bp.route("/<lot_name>", methods=["GET"])
def get_wafer_result(lot_name):
    """
    [3단계] 분석 결과 조회 (기존 로직)
    """
    conn = get_conn()
    cur = conn.cursor(pymysql.cursors.DictCursor)
    try:
        cur.execute("SELECT * FROM wafer_data WHERE lot_name = %s", (lot_name,))
        result = cur.fetchone()
        
        if not result:
            return jsonify({"error": "Not found"}), 404
            
        return jsonify(result), 200
    finally:
        cur.close()
        conn.close()

@wafer_bp.route("/list", methods=["GET"])
def get_wafer_list():
    """
    [4단계] 웨이퍼 목록 조회 (페이지네이션)
    """
    page = int(request.args.get('page', 1))
    limit = int(request.args.get('limit', 20))
    offset = (page - 1) * limit
    
    conn = get_conn()
    cur = conn.cursor(pymysql.cursors.DictCursor)
    try:
        # 1. 전체 개수 조회
        cur.execute("SELECT COUNT(*) as cnt FROM wafer_data")
        total_res = cur.fetchone()
        total_count = total_res['cnt'] if total_res else 0
        
        # 2. 데이터 조회
        query = """
            SELECT lot_name, failure_type, confidence, die_count, defect_count, defect_density, total_grade, created_at, wafer_map
            FROM wafer_data 
            ORDER BY created_at DESC 
            LIMIT %s OFFSET %s
        """
        cur.execute(query, (limit, offset))
        rows = cur.fetchall()
        
        return jsonify({
            "wafers": rows,
            "total": total_count,
            "page": page,
            "limit": limit
        }), 200
        
    except Exception as e:
        print(f"[Error] /list: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        cur.close()
        conn.close()

@wafer_bp.route("/total_status", methods=["GET"])
def get_total_status():
    """
    [대시보드] 전체 통계 조회 (수정됨)
    - 총 분석 웨이퍼 = wafer_data 행 수
    - 추출 가능한 칩 수 = SUM(die_count) from wafer_data
    - 불량 칩 수 = SUM(defect_count) from wafer_data
    - 결함 밀도 = (불량 칩 수 / 추출 가능한 칩 수) * 100
    """
    conn = get_conn()
    cur = conn.cursor(pymysql.cursors.DictCursor)
    try:
        # 1. 총 웨이퍼 수
        cur.execute("SELECT COUNT(*) as cnt FROM wafer_data")
        res_wafer = cur.fetchone()
        total_wafers = res_wafer['cnt'] if res_wafer else 0

        # 2. 칩 통계 (wafer_data 집계)
        cur.execute("SELECT SUM(die_count) as total_die, SUM(defect_count) as total_defect FROM wafer_data")
        row = cur.fetchone()
        
        # DB 디버깅용 로그
        print(f"[DEBUG] /total_status Query Result: {row}")

        # None 체크 및 타입 변환 (Decimal 호환성)
        # pymysql에서 SUM 결과는 Decimal로 반환될 수 있음 -> float/int 변환 필요
        total_die_val = row['total_die'] if row and row['total_die'] is not None else 0
        total_defect_val = row['total_defect'] if row and row['total_defect'] is not None else 0
        
        total_die = float(total_die_val)
        total_defect = float(total_defect_val)
        
        # 결함 밀도 계산
        defect_density = (total_defect / total_die * 100) if total_die > 0 else 0.0
        
        result = {
            "totalWafers": total_wafers,
            "totalDie": int(total_die),
            "defectCount": int(total_defect),
            "defectDensity": round(defect_density, 2)
        }
        print(f"[DEBUG] /total_status Response: {result}")
        
        return jsonify(result), 200

    except Exception as e:
        print(f"[Error] /total_status: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        cur.close()
        conn.close()

# ==========================================
# CSV 내보내기 API
# ==========================================
@wafer_bp.route("/export", methods=["GET"])
def export_wafer_data():
    """wafer_data 테이블을 CSV 파일로 내보내기"""
    try:
        conn = get_conn()
        cur = conn.cursor(pymysql.cursors.DictCursor)
        
        # wafer_data 테이블 전체 조회
        query = """
            SELECT lot_name, failure_type, confidence, die_count, defect_count, 
                   defect_density, total_grade, created_at
            FROM wafer_data 
            ORDER BY created_at DESC
        """
        cur.execute(query)
        rows = cur.fetchall()
        
        if not rows:
            return jsonify({"error": "No data to export"}), 404
        
        # CSV 생성
        import io
        import csv
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # 헤더 작성
        headers = ["Lot Name", "Failure Type", "Confidence", "Die Count", 
                   "Defect Count", "Defect Density", "Grade", "Created At"]
        writer.writerow(headers)
        
        # 데이터 작성
        for row in rows:
            writer.writerow([
                row['lot_name'],
                row['failure_type'],
                f"{row['confidence']:.4f}" if row['confidence'] else "",
                row['die_count'],
                row['defect_count'],
                f"{row['defect_density']:.6f}" if row['defect_density'] else "",
                row['total_grade'],
                row['created_at'].strftime('%Y-%m-%d %H:%M:%S') if row['created_at'] else ""
            ])
        
        # CSV 데이터 가져오기
        csv_data = output.getvalue()
        output.close()
        
        # 파일명 생성 (현재 날짜시간 포함)
        now = datetime.datetime.now()
        filename = f"wafer_data_{now.strftime('%Y%m%d_%H%M%S')}.csv"
        
        # Response 생성
        from flask import make_response
        response = make_response(csv_data)
        response.headers["Content-Disposition"] = f"attachment; filename={filename}"
        response.headers["Content-Type"] = "text/csv; charset=utf-8-sig"  # UTF-8 BOM for Excel
        
        return response

    except Exception as e:
        print(f"[Error] /export: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        cur.close()
        conn.close()
