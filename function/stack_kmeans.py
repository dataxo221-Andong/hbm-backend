from flask import Blueprint, request, jsonify
import sys
import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import Dataset, DataLoader
import joblib
import math
import datetime
import json
import pymysql
import random
import time
from db import get_conn

# Blueprint 정의
stack_bp = Blueprint("stack", __name__, url_prefix="/stack")

# 경로 설정
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

def run_stacking_simulation_logic(batch_id):
    # 1. 파일 찾기 logic (Server specific)
    if not batch_id:
        files = [f for f in os.listdir(MODELS_DIR) if f.startswith('BATCH_') and f.endswith('.pkl')]
        if not files: 
            return {"error": "No batch files", "stacks": []}
        files.sort(key=lambda x: x[6:21] if len(x) >= 21 else "", reverse=True)
        filename = files[0]
        batch_id = os.path.splitext(filename)[0]
    else:
        filename = f"{batch_id}.pkl"
    
    pkl_path = os.path.join(MODELS_DIR, filename)
    model_path = os.path.join(MODELS_DIR, 'wafer_classifier.pth')
    kmeans_path = os.path.join(MODELS_DIR, 'kmeans_model.pkl')
    
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"File not found: {pkl_path}")

    # ==================================================================================
    # [Start] code2.py Logic Exact Copy
    # ==================================================================================

    # GPU 사용 가능 시 활용
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        pass # Server might not allow set_device(0) freely, but keeping logic similar
        # torch.cuda.set_device(0) 
        torch.backends.cudnn.benchmark = True
        print(f"CUDA 사용: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA 사용 불가: CPU로 실행합니다.")

    # ===== 데이터 로딩 =====
    df = pd.read_pickle(pkl_path)
    if isinstance(df, list):
        df = pd.DataFrame(df)

    # 컬럼명 정규화
    if "failure_type" not in df.columns:
        df["failure_type"] = df["failure_type"]

    # failure_type 처리
    sample = df['failure_type'].iloc[0] if len(df) > 0 else None
    if sample is not None:
        if isinstance(sample, list):
            df['failure_type'] = df['failure_type'].apply(
                lambda x: x[0][0] if isinstance(x, list) and len(x) > 0 and isinstance(x[0], (list, tuple)) and len(x[0]) > 0 
                else (x[0] if isinstance(x, list) and len(x) > 0 else '')
            )
            df = df[df['failure_type'].apply(lambda x: len(str(x)) > 0)]
        elif not isinstance(sample, str):
            df['failure_type'] = df['failure_type'].astype(str)

    target_labels = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Near-full', 'Random', 'Scratch', 'None']
    df = df[df['failure_type'].isin(target_labels)].reset_index(drop=True)

    print(f"[Data] 필터링 후 데이터 개수: {len(df)}")
    if len(df) == 0:
         return {"stacks": [], "message": "Not enough chips"}

    # ===== Dataset =====
    class WaferDataset(Dataset):
        def __init__(self, dataframe):
            self.data = dataframe
            self.label_map = {label: i for i, label in enumerate(target_labels)}

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            tsv_matrix = self.data.iloc[idx]['tsv_matrix']
            if not isinstance(tsv_matrix, np.ndarray):
                tsv_matrix = np.array(tsv_matrix)
            wafer_map = cv2.resize(tsv_matrix.astype('float32'), (64, 64), interpolation=cv2.INTER_NEAREST)
            wafer_map = wafer_map / 2.0
            wafer_tensor = torch.tensor(wafer_map, dtype=torch.float32).unsqueeze(0)
            return wafer_tensor

    dataset = WaferDataset(df)
    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        pin_memory=(device.type == 'cuda'),
    )

    # ===== Feature Extractor (EXACT code2.py loading) =====
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 9)
    # [EXACT COPY] No try-except, no strict=True/False checks, just load.
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("[Debug] Model loaded via torch.load")
    except Exception as e:
        print(f"[Debug] Standard loading failed: {e}")
        # 키 불일치 등 문제 발생 시 상세 내용 출력
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            print(f"[Debug] Checkpoint keys: {list(checkpoint['state_dict'].keys())[:5]}")
        else:
            print(f"[Debug] Checkpoint keys: {list(checkpoint.keys())[:5]}")
        print(f"[Debug] Model keys: {list(model.state_dict().keys())[:5]}")
        raise e

    model = model.to(device).eval()
    
    # [검증] 가중치 앞부분 출력 (code2.py와 비교용)
    first_conv_weight = model.conv1.weight.data.flatten()[:5].cpu().numpy()
    print(f"[Verify] Model Conv1 Weights Top-5: {first_conv_weight}")
    
    feature_extractor = nn.Sequential(*list(model.children())[:-1]).to(device).eval()

    # ===== Feature 추출 =====
    features = []
    total_batches = len(loader)
    start_t = time.perf_counter()
    last_log_t = start_t

    if device.type == 'cuda':
        torch.cuda.synchronize()

    with torch.no_grad():
        for batch_idx, x in enumerate(loader, start=1):
            x = x.to(device, non_blocking=True)
            out = feature_extractor(x)
            out = out.view(out.size(0), -1)
            features.append(out.cpu().numpy())

            if batch_idx == 1 or batch_idx % 20 == 0 or batch_idx == total_batches:
                now = time.perf_counter()
                elapsed = now - start_t
                avg_per_batch = elapsed / batch_idx
                remaining = avg_per_batch * (total_batches - batch_idx)
                print(f"[Feature] {batch_idx}/{total_batches} elapsed={elapsed:.1f}s eta={remaining/60:.1f}m")
                sys.stdout.flush()

    if device.type == 'cuda':
        torch.cuda.synchronize()

    if len(features) == 0:
        return {"stacks": [], "message": "Feature extraction failed"}

    X_features = np.concatenate(features, axis=0)
    print(f"[Feature] 최종 feature shape: {X_features.shape}")

    # ===== KMeans =====
    kmeans = joblib.load(kmeans_path)
    cluster_labels = kmeans.predict(X_features)

    # ===== 클러스터 -> 패턴 =====
    cluster_to_pattern = {
        0: "Donut",
        1: "Center",
        2: "Random",
        3: "Edge-Ring",
        4: "Scratch",
        5: "Donut",
        6: "Loc",
        7: "Random",
        8: "Edge-Loc",
    }
    
    # DB Save Support
    df['cluster_label'] = cluster_labels
    df['mapped_type'] = [cluster_to_pattern[lbl] for lbl in cluster_labels]

    # ===== Cost 함수 =====
    def build_mask(wafer_tensor):
        arr = wafer_tensor.detach().cpu().numpy()
        arr = np.squeeze(arr)
        if arr.ndim != 2:
            raise ValueError(f"wafer_map shape 오류: {arr.shape}")
        return (arr >= 1.0).astype(np.float32)

    def spatial_overlap(mask_i, mask_j):
        mask_i = np.asarray(mask_i, dtype=np.float32)
        mask_j = np.asarray(mask_j, dtype=np.float32)
        overlap = np.sum(mask_i * mask_j)
        denom = (np.sum(mask_i) + np.sum(mask_j) + 1e-6)
        return overlap / denom

    def critical_weight_map(size=64, center_weight=2.0):
        yy, xx = np.mgrid[0:size, 0:size]
        cy = cx = (size - 1) / 2.0
        dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        return 1.0 + (center_weight - 1.0) * (1.0 - dist / dist.max())

    W_critical = critical_weight_map().astype(np.float32)

    def critical_overlap(mask_i, mask_j):
        mask_i = np.asarray(mask_i, dtype=np.float32)
        mask_j = np.asarray(mask_j, dtype=np.float32)
        mask_i = np.squeeze(mask_i)
        mask_j = np.squeeze(mask_j)
        return np.sum(W_critical * mask_i * mask_j)

    lambda1, lambda2, lambda3 = 0.35, 0.45, 0.20

    base_type_penalty = {
        ("Center", "Center"): 1.0,
        ("Donut", "Donut"): 0.85,
        ("Edge-Ring", "Edge-Ring"): 0.80,
        ("Donut", "Center"): 0.70,
        ("Center", "Edge-Ring"): 0.55,
        ("Donut", "Edge-Ring"): 0.60,
        ("Center", "Loc"): 0.45,
        ("Center", "Edge-Loc"): 0.45,
        ("Center", "Scratch"): 0.40,
        ("Donut", "Loc"): 0.40,
        ("Donut", "Edge-Loc"): 0.40,
        ("Donut", "Scratch"): 0.35,
        ("Edge-Ring", "Loc"): 0.35,
        ("Edge-Ring", "Edge-Loc"): 0.35,
        ("Edge-Ring", "Scratch"): 0.30,
        ("Loc", "Loc"): 0.30,
        ("Edge-Loc", "Edge-Loc"): 0.30,
        ("Scratch", "Scratch"): 0.25,
        ("Loc", "Edge-Loc"): 0.28,
        ("Loc", "Scratch"): 0.25,
        ("Edge-Loc", "Scratch"): 0.25,
        ("Random", "Random"): 0.15,
        ("Random", "Loc"): 0.18,
        ("Random", "Edge-Loc"): 0.18,
        ("Random", "Scratch"): 0.18,
        ("Random", "Donut"): 0.25,
        ("Random", "Center"): 0.25,
        ("Random", "Edge-Ring"): 0.25,
        ("Near-full", "Near-full"): 0.20,
        ("None", "None"): 0.10,
    }

    forbidden_pairs = {
        ("Center", "Center"),
        ("Donut", "Donut"),
        ("Edge-Ring", "Edge-Ring")
    }

    def type_penalty(t1, t2):
        if (t1, t2) in forbidden_pairs or (t2, t1) in forbidden_pairs:
            return math.inf
        return base_type_penalty.get((t1, t2), base_type_penalty.get((t2, t1), 0.2))

    def cost_fn(i, j, masks):
        # [EXACT COPY] No safeguards
        t1 = cluster_to_pattern[cluster_labels[i]]
        t2 = cluster_to_pattern[cluster_labels[j]]
        p = type_penalty(t1, t2)
        if math.isinf(p):
            return math.inf
        so = spatial_overlap(masks[i], masks[j])
        co = critical_overlap(masks[i], masks[j])
        return lambda1 * so + lambda2 * p + lambda3 * co

    # ===== 마스크 계산 =====
    masks = []
    mask_total = len(dataset)
    mask_start = time.perf_counter()
    with torch.no_grad():
        for i in range(mask_total):
            masks.append(build_mask(dataset[i]))
            if (i + 1) == 1 or (i + 1) % 500 == 0 or (i + 1) == mask_total:
                print(f"[Mask] {i+1}/{mask_total}")
                sys.stdout.flush()

    # ===== 8개 단위 매칭(그룹화) =====
    tau = 5.0
    group_size = 8
    N = len(df)

    pair_total = N * (N - 1) // 2
    pair_start = time.perf_counter()
    print(f"[Group] cost matrix 계산 시작: pairs={pair_total}")
    sys.stdout.flush()

    cost_mat = np.full((N, N), np.inf, dtype=np.float32)
    for i in range(N):
        cost_mat[i, i] = 0.0

    processed_pairs = 0
    last_progress = 0
    progress_interval = 0.01

    for i in range(N):
        for j in range(i + 1, N):
            c = cost_fn(i, j, masks)
            if math.isinf(c):
                pass
            else:
                cost_mat[i, j] = c
                cost_mat[j, i] = c
            
            processed_pairs += 1
            current_progress = processed_pairs / pair_total
            if current_progress - last_progress >= progress_interval:
                elapsed = time.perf_counter() - pair_start
                if processed_pairs > 0:
                    print(f"[Group] cost matrix 진행: {current_progress*100:.1f}%")
                    sys.stdout.flush()
                last_progress = current_progress

    elapsed = time.perf_counter() - pair_start
    print(f"[Group] cost matrix 완료: {elapsed:.1f}s")

    def _pick_next(group, remaining):
        best_j = None
        best_score = float("inf")
        group_types = set(df.iloc[g]['failure_type'] for g in group)
        
        for j in remaining:
            candidate_type = df.iloc[j]['failure_type']
            if candidate_type in group_types:
                continue
            costs = [cost_mat[j, g] for g in group]
            if all(math.isinf(c) for c in costs):
                continue
            score = float(np.mean(costs))
            if score < best_score:
                best_score = score
                best_j = j
        return best_j

    remaining = list(range(N))
    groups = []
    max_attempts = len(remaining) * 2
    attempt_count = 0

    print(f"[Grouping] 그룹화 시작")

    while len(remaining) >= group_size and attempt_count < max_attempts:
        attempt_count += 1
        seed = remaining.pop(0)
        group = [seed]
        failed = False
        
        for _ in range(group_size - 1):
            pick = _pick_next(group, remaining)
            if pick is None:
                failed = True
                break
            remaining.remove(pick)
            group.append(pick)
        
        if failed:
            remaining.append(seed)
            for item in group[1:]:
                if item not in remaining:
                    remaining.append(item)
            continue
        
        groups.append(group)

    print(f"[Grouping] 완료: {len(groups)} groups")

    # ==================================================================================
    # [End] code2.py Logic
    # ==================================================================================

    stacks_result = []
    conn = get_conn()
    cur = conn.cursor()
    
    try:
        current_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        values_to_insert = []
        
        for g_idx, grp in enumerate(groups):
            valid_costs = [
                cost_mat[grp[i], grp[j]] 
                for i in range(len(grp)) 
                for j in range(i+1, len(grp)) 
                if not math.isinf(cost_mat[grp[i], grp[j]])
            ]
            total_c = sum(valid_costs) if valid_costs else 0
            avg_c = total_c / (len(grp)*(len(grp)-1)/2) if len(grp) > 1 else 0
            score_label = f"{'A' if avg_c < 0.5 else 'B' if avg_c < 0.8 else 'C'} ({avg_c:.2f})"
            stack_id_str = f"STACK_{batch_id}_{g_idx+1}"
            
            frontend_layer_list = []
            
            for rank, chip_idx in enumerate(grp):
                row = df.iloc[chip_idx]
                chip_uid = str(row.get('chip_uid', ''))
                lot_name = str(row.get('lot_name', ''))
                raw_failure_type = str(row.get('failure_type', ''))
                mapped_type = str(row.get('mapped_type', ''))
                
                tsv_val = row.get('tsv_matrix', [])
                if isinstance(tsv_val, np.ndarray): 
                    tsv_val = tsv_val.tolist()
                tsv_status_str = json.dumps(tsv_val)
                
                cx = str(row.get('coor_x', '0'))
                cy = str(row.get('coor_y', '0'))
                
                values_to_insert.append((
                    g_idx + 1, 
                    rank + 1, 
                    int(chip_idx), 
                    chip_uid, 
                    lot_name, 
                    raw_failure_type,
                    current_time_str, 
                    tsv_status_str, 
                    str(row.get('die_status', '')), 
                    f"{cx},{cy}", 
                    cx, 
                    cy
                ))
                
                c_lbl = int(row.get('cluster_label', -1))
                frontend_layer_list.append({
                    "layer_idx": rank + 1,
                    "chip_id": chip_uid,
                    "cluster_label": c_lbl,
                    "mapped_type": mapped_type,
                    "failure_type": raw_failure_type
                })
            
            stacks_result.append({
                "stack_id": stack_id_str,
                "score": score_label,
                "layers": frontend_layer_list
            })

        if values_to_insert:
            sql = """
                INSERT INTO grouped_data 
                (group_number, position_in_group, data_index, chip_uid, lot_name, 
                 failure_type, created_at, tsv_status, die_status, tsv_coordinate, coor_x, coor_y)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            cur.executemany(sql, values_to_insert)
            conn.commit()
            print(f"[Debug] DB Insert Success: {len(values_to_insert)} rows.")
            
    except Exception as e:
        conn.rollback()
        print(f"[DB Error] {e}")
        import traceback
        traceback.print_exc()
    finally:
        cur.close()
        conn.close()

    return {
        "batch_id": batch_id,
        "stacks": stacks_result,
        "total_chips": len(df),
        "stacks_formed": len(groups)
    }

@stack_bp.route("/analyze", methods=["POST"])
def analyze_stack():
    try:
        data = request.get_json() or {}
        batch_id = data.get('batch_id')
        
        result = run_stacking_simulation_logic(batch_id)
        return jsonify(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
