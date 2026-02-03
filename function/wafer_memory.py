from flask import Blueprint, request, jsonify
import sys
import os
import cv2
import numpy as np
import torch
import base64
import datetime
import random
import uuid
from urllib.parse import quote
import json

# 프로젝트 루트 경로 보장 (auth.py 스타일)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from db import get_conn

# TSV Processor import (메모리 기반 - function 폴더 내 tsv_memory 사용)
from function.tsv_memory import process_chip_list

# firebase storage (v2처럼 사용하되, 중복 initialize 방지)
import firebase_admin
from firebase_admin import credentials, storage

wafer_v3_bp = Blueprint("wafer_v3", __name__, url_prefix="/wafer_v3")

from torchvision import models
import torch.nn as nn
import torch.nn.functional as F

LABELS = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Near-full", "Random", "Scratch", "None"]
NONE_VARIANTS = ["None"]


def _ensure_firebase_initialized():
    # 이미 초기화돼 있으면 재초기화하지 않음(중복 초기화 에러 방지)
    if firebase_admin._apps:
        return
    # 키 파일은 실행 cwd에 의존하지 않도록 hbm-backend 기준 절대경로 사용
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../hbm-backend
    key_path = os.path.join(base_dir, "wafer-firebase-key.json")
    cred = credentials.Certificate(key_path)
    firebase_admin.initialize_app(cred, {"storageBucket": "wafer-service-a07b3.firebasestorage.app"})


def determine_grade(defect_density):
    if defect_density <= 0.05:
        return "A"
    elif defect_density <= 0.15:
        return "B"
    else:
        return "C"


class WaferClassifier(nn.Module):
    def __init__(self, num_classes=9):
        super(WaferClassifier, self).__init__()
        try:
            self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        except Exception:
            self.model = models.resnet18(pretrained=True)

        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)


def load_model_instance(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WaferClassifier()

    if not os.path.exists(model_path):
        print(f"[Warning] Model file not found: {model_path}")
        return None

    try:
        loaded_obj = torch.load(model_path, map_location=device)
        if isinstance(loaded_obj, dict):
            state_dict = loaded_obj.get("state_dict", loaded_obj)
            first_key = next(iter(state_dict.keys()))
            if not first_key.startswith("model.") and hasattr(model, "model"):
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


_PROCESSOR_INSTANCE = None


def get_processor():
    global _PROCESSOR_INSTANCE
    if _PROCESSOR_INSTANCE is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, "models", "wafer_classifier.pth")
        if not os.path.exists(model_path):
            root_dir = os.path.dirname(base_dir)
            model_path = os.path.join(root_dir, "models", "wafer_classifier.pth")
        _PROCESSOR_INSTANCE = WaferProcessor(model_path)
    return _PROCESSOR_INSTANCE


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
            max_prob, pred_idx = torch.max(probs, 1)

        label_idx = pred_idx.item()
        confidence = max_prob.item() * 100.0
        predicted_label = LABELS[label_idx] if 0 <= label_idx < len(LABELS) else "Unknown"
        if predicted_label in NONE_VARIANTS:
            predicted_label = "None"
        return predicted_label, confidence

    def process_wafer(self, file_stream, lot_name_input=None):
        file_bytes = np.frombuffer(file_stream.read(), np.uint8)
        original_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if original_img is None:
            raise ValueError("Invalid image file")

        target_size = (32, 32)
        gray_full = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY) if len(original_img.shape) == 3 else original_img
        resized_gray = cv2.resize(gray_full.astype("float32"), target_size, interpolation=cv2.INTER_NEAREST)
        resized_gray = resized_gray.astype(int)

        quantized_map = np.zeros_like(resized_gray)
        quantized_map[resized_gray > 80] = 1
        quantized_map[resized_gray > 170] = 2

        # 시각화 PNG
        color_map = np.zeros((32, 32, 3), dtype=np.uint8)
        color_map[quantized_map == 0] = self.COLOR_BLACK
        color_map[quantized_map == 1] = self.COLOR_RED
        color_map[quantized_map == 2] = self.COLOR_YELLOW

        ok, buffer = cv2.imencode(".png", color_map)
        if not ok:
            raise ValueError("Failed to encode wafer map")
        wafer_map_b64 = base64.b64encode(buffer).decode("utf-8")

        # 모델 분석
        failure_type, confidence = self.predict_failure(resized_gray)

        die_count = int(np.sum(quantized_map > 0))
        defect_count = int(np.sum(quantized_map == 2))
        defect_density = defect_count / die_count if die_count > 0 else 0.0
        total_grade = determine_grade(defect_density)

        now = datetime.datetime.now()

        # lotName: 길이 짧게(18자리) + 중복 방지 위해 초 포함
        if not lot_name_input:
            import string

            chars = string.ascii_uppercase + string.digits
            random_serial = "".join(random.choices(chars, k=6))
            lot_name = f"{now.strftime('%y%m%d')}{random_serial}{now.strftime('%H%M%S')}"  # 6+6+6=18
        else:
            lot_name = lot_name_input

        # Firebase 업로드 (옵션)
        _ensure_firebase_initialized()
        bucket = storage.bucket()
        blob = bucket.blob(f"wafer_images/{lot_name}.png")
        download_token = str(uuid.uuid4())
        blob.metadata = {"firebaseStorageDownloadTokens": download_token}
        blob.upload_from_string(buffer.tobytes(), content_type="image/png")
        blob.patch()
        wafer_map_url = (
            f"https://firebasestorage.googleapis.com/v0/b/{bucket.name}/o/"
            f"{quote(blob.name, safe='')}?alt=media&token={download_token}"
        )

        wafer_data = {
            "lotName": lot_name,
            "waferMap": wafer_map_b64,
            "waferSize": "32x32",
            "failureType": failure_type,
            "confidence": float(confidence),
            "dieCount": die_count,
            "defectCount": defect_count,
            "defectDensity": defect_density,
            "totalGrade": total_grade,
            "created_at": now.strftime("%Y-%m-%d %H:%M:%S"),
            "img_url": wafer_map_url,
        }

        # chip 리스트 생성 (파일 저장 없음)
        chip_list = []
        rows, cols = quantized_map.shape
        for y in range(rows):
            for x in range(cols):
                status_val = int(quantized_map[y, x])
                if status_val == 0:
                    continue
                chip_list.append(
                    {
                        "chip_uid": f"{lot_name}X{x}Y{y}D{status_val}",
                        "lotName": lot_name,
                        "tsv_matrix": [],
                        "tsv_coordinate": f"{x},{y}",
                        "tsv_status": status_val,
                        "failureType": failure_type,  # TSV 패턴 생성에 필요
                        "created_at": wafer_data["created_at"],
                    }
                )

        return wafer_data, chip_list


@wafer_v3_bp.route("/upload", methods=["POST"])
def upload_wafer_v3():
    files = request.files.getlist("wafer_image")
    if not files or (len(files) == 1 and files[0].filename == ""):
        return jsonify({"error": "No file part"}), 400

    conn = None
    cur = None
    try:
        processor = get_processor()

        all_wafer_values = []
        all_chip_values = []
        uploaded_lot_names = []

        for f in files:
            f.seek(0)
            wafer_data, chip_list = processor.process_wafer(f)

            # TSV 처리 (메모리 기반)
            tsv_chips = process_chip_list(chip_list)

            all_wafer_values.append(
                (
                    wafer_data["lotName"],
                    wafer_data["waferMap"],
                    wafer_data["failureType"],
                    wafer_data["confidence"],
                    wafer_data["dieCount"],
                    wafer_data["defectCount"],
                    wafer_data["defectDensity"],
                    wafer_data["totalGrade"],
                    wafer_data["created_at"],
                    wafer_data["img_url"],
                )
            )
            uploaded_lot_names.append(wafer_data["lotName"])

            for chip in tsv_chips:
                tsv_matrix_value = chip.get("tsv_matrix", [])
                if isinstance(tsv_matrix_value, np.ndarray):
                    tsv_matrix_str = json.dumps(tsv_matrix_value.tolist())
                elif not tsv_matrix_value:
                    tsv_matrix_str = "[]"
                else:
                    tsv_matrix_str = json.dumps(tsv_matrix_value)

                all_chip_values.append(
                    (
                        chip["chip_uid"],
                        chip["lotName"],
                        tsv_matrix_str,
                        chip["tsv_coordinate"],
                        chip["tsv_status"],
                        chip.get("failureType", "None"),  # failureType 추가
                        chip["created_at"],
                    )
                )

        conn = get_conn()
        if conn is None:
            return jsonify({"error": "DB Connection Failed"}), 500
        cur = conn.cursor()

        # 1) wafer_data 먼저
        sql_wafer = """
            INSERT INTO project.wafer_data
            (`lotName`, `waferMap`, `failureType`, `confidence`, `dieCount`, `defectCount`, `defectDensity`, `totalGrade`, `created_at`, `img_url`)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        cur.executemany(sql_wafer, all_wafer_values)

        # 2) chip_data
        if all_chip_values:
            sql_chip = """
                INSERT INTO project.chip_data
                (`chip_uid`, `lotName`, `tsv_matrix`, `tsv_coordinate`, `tsv_status`, `failureType`, `created_at`)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            cur.executemany(sql_chip, all_chip_values)

        conn.commit()

        return jsonify(
            {
                "message": f"성공!! 총 {len(all_wafer_values)}개 웨이퍼, {len(all_chip_values)}개 칩이 들어갔습니다!",
                "count": len(all_wafer_values),
                "chipCount": len(all_chip_values),
                "lotNames": uploaded_lot_names,
            }
        ), 200

    except Exception as e:
        if conn:
            conn.rollback()
        print(f"DEBUG ERROR: {str(e)}")
        return jsonify({"error": str(e)}), 500
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()


