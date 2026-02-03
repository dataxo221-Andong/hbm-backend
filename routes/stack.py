from flask import Blueprint, request, jsonify
import sys
import os
import cv2
import numpy as np
import pandas as pd
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
    
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"File not found: {pkl_path}")

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

    # ===== AI/KMeans 제거 -> 원본 Label 사용 =====
    # cluster_label 대신 그냥 -1로 채우거나, 필요한 경우 매핑
    df['cluster_label'] = -1 
    # mapped_type은 원본 failure_type 그대로 사용
    df['mapped_type'] = df['failure_type']

    # ===== Cost 함수 관련 설정 =====
    lambda1, lambda2, lambda3 = 0.35, 0.45, 0.20

    # [수정] Diversity Enforcement: 동일 패턴끼리의 페널티를 대폭 상향 (0.3 -> 2.0 이상)
    # 이렇게 하면 Cost 계산 단계에서부터 같은 패턴끼리 묶이는 것을 '매우 비싼 비용'으로 인식하여 피하게 됨.
    base_type_penalty = {
        ("Center", "Center"): 1.0, # 원래 금지지만 혹시 몰라 높게 유지
        ("Donut", "Donut"): 2.0,
        ("Edge-Ring", "Edge-Ring"): 2.0,
        ("Loc", "Loc"): 2.0,
        ("Edge-Loc", "Edge-Loc"): 2.0,
        ("Scratch", "Scratch"): 2.0,
        ("Random", "Random"): 1.5, # Random은 그나마 좀 허용하되 여전히 높게
        ("Near-full", "Near-full"): 2.0,
        ("None", "None"): 0.15,
        
        # 서로 다른 조합은 여전히 낮게 유지 (권장)
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
        ("Loc", "Edge-Loc"): 0.28,
        ("Loc", "Scratch"): 0.25,
        ("Edge-Loc", "Scratch"): 0.25,
        ("Random", "Loc"): 0.18,
        ("Random", "Edge-Loc"): 0.18,
        ("Random", "Scratch"): 0.18,
        ("Random", "Donut"): 0.25,
        ("Random", "Center"): 0.25,
        ("Random", "Edge-Ring"): 0.25,
    }

    forbidden_pairs = {
        ("Center", "Center"),
        ("Donut", "Donut"),
        ("Edge-Ring", "Edge-Ring")
    }

    def critical_weight_map(size=64, center_weight=2.0):
        yy, xx = np.mgrid[0:size, 0:size]
        cy = cx = (size - 1) / 2.0
        dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        return 1.0 + (center_weight - 1.0) * (1.0 - dist / dist.max())

    W_critical = critical_weight_map().astype(np.float32)

    def type_penalty(t1, t2):
        if (t1, t2) in forbidden_pairs or (t2, t1) in forbidden_pairs:
            return math.inf
        # default penalty 0.2 -> 0.1로 낮춰서 '다른 패턴'을 더 장려
        return base_type_penalty.get((t1, t2), base_type_penalty.get((t2, t1), 0.1))

    def spatial_overlap(mask_i, mask_j):
        overlap = np.sum(mask_i * mask_j)
        denom = (np.sum(mask_i) + np.sum(mask_j) + 1e-6)
        return overlap / denom

    def critical_overlap(mask_i, mask_j):
        return np.sum(W_critical * mask_i * mask_j)

    def cost_fn(i, j, masks):
        # AI 결과 대신 원본 failure_type 사용
        t1 = df.iloc[i]['failure_type']
        t2 = df.iloc[j]['failure_type']
        
        p = type_penalty(t1, t2)
        if math.isinf(p):
            return math.inf
        
        so = spatial_overlap(masks[i], masks[j])
        co = critical_overlap(masks[i], masks[j])
        return lambda1 * so + lambda2 * p + lambda3 * co

    # ===== 마스크 계산 (No DataLoader, Direct Processing) =====
    masks = []
    mask_start = time.perf_counter()
    
    print("[Mask] Start processing masks directly from TSV data...")
    for idx in range(len(df)):
        tsv_matrix = df.iloc[idx]['tsv_matrix']
        if not isinstance(tsv_matrix, np.ndarray):
            tsv_matrix = np.array(tsv_matrix)
        
        # Resize to 64x64
        wafer_map = cv2.resize(tsv_matrix.astype('float32'), (64, 64), interpolation=cv2.INTER_NEAREST)
        wafer_map = wafer_map / 2.0
        mask = (wafer_map >= 1.0).astype(np.float32)
        mask = np.squeeze(mask)
        masks.append(mask)

    print(f"[Mask] Completed {len(masks)} masks.")
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
    np.fill_diagonal(cost_mat, 0.0)

    processed_pairs = 0
    last_progress = 0
    progress_interval = 0.05  # 5% 단위

    for i in range(N):
        for j in range(i + 1, N):
            c = cost_fn(i, j, masks)
            if not math.isinf(c):
                cost_mat[i, j] = c
                cost_mat[j, i] = c
            
            processed_pairs += 1
            current_progress = processed_pairs / pair_total
            if current_progress - last_progress >= progress_interval:
                elapsed = time.perf_counter() - pair_start
                print(f"[Group] cost matrix 진행: {current_progress*100:.1f}%")
                sys.stdout.flush()
                last_progress = current_progress

    elapsed = time.perf_counter() - pair_start
    print(f"[Group] cost matrix 완료: {elapsed:.1f}s")

    def _pick_next(group, remaining):
        # 후보군 수집 (score, j)
        candidates = []
        
        # [수정] 규칙 강화: 연속 적층 절대 금지 & 최대 2개 제한 (Diversity First)
        # failure_type 가져올 때 .strip() 적용하여 문자열 매칭 확실하게 함
        
        group_types_raw = [df.iloc[g]['failure_type'] for g in group]
        # 혹시 모를 공백 제거
        group_types = [str(t).strip() for t in group_types_raw]
        last_type = group_types[-1]
        
        for j in remaining:
            candidate_type_raw = df.iloc[j]['failure_type']
            candidate_type = str(candidate_type_raw).strip()
            
            # None이 아닌 경우에만 제약 적용
            if candidate_type != 'None':
                # Rule 1: 연속 적층 절대 금지 (바로 아래층과 같으면 무조건 Skip)
                if candidate_type == last_type:
                    continue
                
                # Rule 2: 한 스택 내 최대 2개까지만 허용 (더 다양하게 섞이도록 유도)
                # 단, 데이터가 너무 부족해서 2개 제한으로는 그룹을 못 만드는 경우를 대비해 
                # 초기에는 엄격하게 하되, 정 안되면 나중에 완화하는 전략이 필요하지만
                # 일단 사용자 요청대로 엄격하게 2개로 제한.
                if group_types.count(candidate_type) >= 2:
                    continue
            
            # 기존 Cost 및 Forbidden Pairs 체크
            costs = [cost_mat[j, g] for g in group]
            
            # 하나라도 inf면(금지된 조합 등) 평균도 inf가 되어 선택되지 않음
            # 명시적으로 스킵
            if any(math.isinf(c) for c in costs):
                continue
                
            score = float(np.mean(costs))
            # [변경] 1등만 찾는게 아니라 리스트에 추가
            candidates.append((score, j))

        # 후보가 없으면 실패
        if not candidates:
            return None
        
        # 점수 오름차순 정렬 (낮은게 좋음)
        candidates.sort(key=lambda x: x[0])
        
        # [핵심] 상위 3개(Top-3) 중에서 랜덤 선택
        # 점수가 약간 더 나쁘더라도(2등, 3등) 선택될 기회를 주어
        # 매번 똑같은 "쌍둥이 스택"이 만들어지는 고착화를 깸
        top_k = candidates[:3]
        return random.choice(top_k)[1]

    remaining = list(range(N))
    groups = []
    max_attempts = len(remaining) * 2
    attempt_count = 0

    print(f"[Grouping] 그룹화 시작")

    while len(remaining) >= group_size and attempt_count < max_attempts:
        attempt_count += 1
        # [수정] 1층(Base Die) 선택 시 무작위 추출 (기존 pop(0) -> 항상 Center가 나오는 문제 해결)
        rand_idx = random.randrange(len(remaining))
        seed = remaining.pop(rand_idx)
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

    # DB Save
    stacks_result = []
    conn = get_conn()
    cur = conn.cursor()
    
    try:
        # [수정] tsv_num 자동 채번 로직 추가 (DictCursor/TupleCursor 호환성 확보)
        # 현재 DB에서 가장 큰 tsv_num을 찾아서 +1 (없으면 1)
        cur.execute("SELECT MAX(tsv_num) as max_num FROM grouped_data")
        row = cur.fetchone()
        
        current_max = 0
        if row:
            if isinstance(row, dict):
                current_max = row.get('max_num')
            else:
                current_max = row[0]
        
        current_tsv_num = (current_max if current_max is not None else 0) + 1
        print(f"[Debug] New tsv_num for this batch: {current_tsv_num}")

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
                    current_tsv_num,  # [추가] 시뮬레이션 회차 번호
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
                
                c_lbl = -1 # AI Removed
                frontend_layer_list.append({
                    "layer_idx": rank + 1,
                    "chip_id": chip_uid,
                    "cluster_label": c_lbl,
                    "mapped_type": mapped_type,
                    "failure_type": raw_failure_type,
                    "die_status": int(float(row.get('die_status', 2))) if row.get('die_status') else 2
                })
            
            stacks_result.append({
                "stack_id": stack_id_str,
                "score": score_label,
                "layers": frontend_layer_list
            })

        if values_to_insert:
            sql = """
                INSERT INTO grouped_data 
                (tsv_num, group_number, position_in_group, data_index, chip_uid, lot_name, 
                 failure_type, created_at, tsv_status, die_status, tsv_coordinate, coor_x, coor_y)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            cur.executemany(sql, values_to_insert)
            conn.commit()
            print(f"[Debug] DB Insert Success: {len(values_to_insert)} rows (tsv_num={current_tsv_num}).")
            
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
