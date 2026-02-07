from flask import Blueprint, request, jsonify, Response, stream_with_context
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
import queue
import threading
from db import get_conn

# Blueprint 정의
stack_bp = Blueprint("stack", __name__, url_prefix="/stack")

# 경로 설정
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

def run_stacking_simulation_logic(batch_id, progress_callback=None):
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
        ("Random", "Random"): 2.0,
        ("Near-full", "Near-full"): 3.0,
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
        ("Edge-Ring", "Loc"): 0.45,
        ("Edge-Ring", "Edge-Loc"): 0.40,
        ("Edge-Ring", "Scratch"): 0.40,
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
    if progress_callback:
        progress_callback(0.0)

    cost_mat = np.full((N, N), np.inf, dtype=np.float32)
    np.fill_diagonal(cost_mat, 0.0)

    processed_pairs = 0
    last_progress = 0
    progress_interval = 0.0001  # 0.01% 단위 (틱 단위)

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
                print(f"[Group] cost matrix 진행: {current_progress*100:.2f}%")
                sys.stdout.flush()
                last_progress = current_progress
                if progress_callback:
                    overall = current_progress * 50.0  # cost matrix = 0~50%
                    progress_callback(round(overall, 2))

    elapsed = time.perf_counter() - pair_start
    print(f"[Group] cost matrix 완료: {elapsed:.1f}s")
    if progress_callback:
        progress_callback(50.0)

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
                
                # Rule 2: 한 스택 내 최대 3개까지만 허용 (더 다양하게 섞이도록 유도)
                if group_types.count(candidate_type) >= 3:
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
        
        # [핵심] 상위 2개(Top-2) 중에서 랜덤 선택
        # 3개는 너무 너그러우므로 2개로 좁혀서 품질 향상 유도
        top_k = candidates[:2]
        return random.choice(top_k)[1]

    remaining = list(range(N))
    groups = []
    max_attempts = len(remaining) * 2
    attempt_count = 0

    print(f"[Grouping] 그룹화 시작")

    # ==========================================
    # [Helper] 스택 수율 계산 함수 (중복 로직 분리)
    # ==========================================
    def _calculate_stack_yield(group_indices):
        """
        주어진 그룹(칩 인덱스 리스트)으로 스택을 쌓았을 때의 최종 수율을 계산
        """
        vertical_matrices = []
        for idx in group_indices:
            chip_info = df.iloc[idx] # Use df.iloc[idx] to get chip info
            tsv = chip_info.get('tsv_matrix')
            if tsv is not None:
                try:
                    # 리스트인 경우 numpy로 변환
                    mat = np.array(tsv) if isinstance(tsv, list) else tsv
                    if mat.size > 0:
                        vertical_matrices.append(mat)
                except:
                    pass
        
        if len(vertical_matrices) == 0:
            return 0.0

        try:
            # 1. Vertical Stacking (3D Array)
            stack_arr = np.array(vertical_matrices)
            
            # 2. Vertical Profection (하나라도 1이면 1)
            merged_defect_map = np.max(stack_arr, axis=0)
            
            rows, cols = merged_defect_map.shape
            total_pins = rows * cols
            
            # 3. Redundancy Processing (Repair)
            # Pad the map with 1s (defects) to handle edges safely
            padded_map = np.pad(merged_defect_map, pad_width=1, mode='constant', constant_values=1)
            
            real_defect_count = 0
            for r in range(rows):
                for c in range(cols):
                    if merged_defect_map[r, c] == 1:
                        # Check 3x3 neighbors (padded coordinates: r+1, c+1)
                        neighbors = padded_map[r:r+3, c:c+3]
                        
                        # 0(Clean)이 하나라도 있으면 Repair 성공 -> 불량 아님
                        if np.any(neighbors == 0):
                            pass 
                        else:
                            real_defect_count += 1
            
            if total_pins == 0: return 0.0
            
            calc_yield = ((total_pins - real_defect_count) / total_pins) * 100.0
            return calc_yield

        except Exception as e:
            # print(f"Yield Calc Warning: {e}")
            return 0.0

    while len(remaining) >= group_size and attempt_count < max_attempts:
        attempt_count += 1
        
        # [수정] 1층(Base Die) 랜덤 선택
        rand_idx = random.randrange(len(remaining))
        seed_candidate = remaining[rand_idx] 

        # [Rule] "Best of N" 전략 + [NEW] 수율 필터링
        trial_count = 5
        best_trial_group = None
        best_trial_score = float('inf')
        
        for _ in range(trial_count):
            temp_remaining = remaining.copy()
            temp_remaining.pop(rand_idx) 
            
            temp_group = [seed_candidate]
            temp_score_sum = 0
            failed = False

            for _ in range(group_size - 1):
                pick = _pick_next(temp_group, temp_remaining)
                if pick is None:
                    failed = True
                    break
                
                last_chip = temp_group[-1]
                cost = cost_mat[last_chip, pick]
                temp_score_sum += cost
                
                temp_remaining.remove(pick)
                temp_group.append(pick)
            
            if not failed:
                # [NEW] 수율 검증 단계
                simulated_yield = _calculate_stack_yield(temp_group)
                
                # 수율 85% 미만이면 가차없이 탈락 (적층 불가 판정)
                if simulated_yield < 85.0:
                    continue

                avg_score = temp_score_sum / (group_size - 1)
                
                # 수율이 보장된 후보 중에서 Cost가 가장 낮은 것을 선택
                if avg_score < best_trial_score:
                    best_trial_score = avg_score
                    best_trial_group = temp_group

        # 5번 시도 후에도 성공한 스택이 없으면 이 Base Die는 스킵 (다음 기회에)
        if best_trial_group is None:
            # 실패했더라도 seed를 맨 뒤로 보내진 않고(random pick이므로), 그냥 continue하면 됨
            continue

        # [확정] 가장 좋았던 스택을 실제 그룹으로 등록하고 remaining에서 제거
        # seed는 pop 해주어야 함
        seed = remaining.pop(rand_idx)
        
        # 나머지 멤버들도 remaining에서 제거
        # (주의: seed는 이미 위에서 뺐으므로 나머지 7개만 빼면 됨)
        final_group_members = best_trial_group[1:] # 0번(seed) 제외
        for member in final_group_members:
            if member in remaining:
                remaining.remove(member)
        
        groups.append(best_trial_group)
        if progress_callback:
            max_possible = max(1, N // group_size)
            grouping_pct = min(1.0, len(groups) / max_possible)
            overall = 50.0 + 50.0 * grouping_pct  # grouping = 50~100%
            progress_callback(round(overall, 2))

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
                
                # [Modified] 공통 함수 사용하여 최종 수율 계산 (DB 저장을 위해 미리 계산)
                final_yield = _calculate_stack_yield(grp)

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
                    cy,
                    final_yield # [추가] stack_yield
                ))
                
                # Calculate Chip Yield
                chip_yield = 0.0
                if isinstance(tsv_val, list) and len(tsv_val) > 0:
                    try:
                        arr = np.array(tsv_val)
                        if arr.size > 0:
                            count_0 = np.sum(arr == 0)
                            chip_yield = (float(count_0) / float(arr.size)) * 100.0
                    except:
                        pass

                c_lbl = -1 # AI Removed
                frontend_layer_list.append({
                    "layer_idx": rank + 1,
                    "chip_id": chip_uid,
                    "cluster_label": c_lbl,
                    "mapped_type": mapped_type,
                    "failure_type": raw_failure_type,
                    "die_status": int(float(row.get('die_status', 2))) if row.get('die_status') else 2,
                    "tsv_matrix": tsv_val,
                    "chip_yield": chip_yield,
                    "chip_yield": chip_yield,
                    "created_at": str(row.get('created_at', current_time_str))  # [수정] 칩 원본 생성 시간
                })
            
            # --- Vertical Stacking Yield & Grade Calculation (Simulation) ---
            # [Modified] 공통 함수 사용하여 최종 수율 계산
            final_yield = _calculate_stack_yield(grp)
            
            final_grade = "N/A"
            
            if final_yield >= 96.0:
                final_grade = "A"
            elif final_yield >= 92.0:
                final_grade = "B"
            else:
                final_grade = "C"

            stacks_result.append({
                "stack_id": stack_id_str,
                "score": score_label,
                "final_grade": final_grade,
                "final_yield": final_yield,
                "layers": frontend_layer_list
            })

        if values_to_insert:
            sql = """
                INSERT INTO grouped_data 
                (tsv_num, group_number, position_in_group, data_index, chip_uid, lot_name, 
                 failure_type, created_at, tsv_status, die_status, tsv_coordinate, coor_x, coor_y, stack_yield)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            cur.executemany(sql, values_to_insert)
            
            # --- [NEW] Simulation Log Summary Insert ---
            # 시뮬레이션 결과를 요약하여 로그 테이블에 저장 (페이지 3 로딩 속도 최적화)
            total_stk = len(stacks_result)
            if total_stk > 0:
                sum_yield = sum([s['final_yield'] for s in stacks_result])
                avg_val = sum_yield / total_stk
                
                cnt_a = sum([1 for s in stacks_result if s['final_grade'] == 'A'])
                cnt_b = sum([1 for s in stacks_result if s['final_grade'] == 'B'])
                cnt_c = sum([1 for s in stacks_result if s['final_grade'] == 'C'])
                
                log_sql = """
                    INSERT INTO simulation_log 
                    (tsv_num, created_at, total_stacks, avg_yield, grade_a, grade_b, grade_c)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """
                cur.execute(log_sql, (
                    current_tsv_num, 
                    current_time_str, 
                    total_stk, 
                    avg_val, 
                    cnt_a, 
                    cnt_b, 
                    cnt_c
                ))
            # -------------------------------------------

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

    if progress_callback:
        progress_callback(100.0)
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


@stack_bp.route("/analyze/stream", methods=["POST"])
def analyze_stack_stream():
    """SSE 스트리밍으로 진행률 실시간 전송 (cost matrix + grouping 전체)"""
    try:
        data = request.get_json() or {}
        batch_id = data.get('batch_id')

        progress_queue = queue.Queue()
        result_holder = [None]

        def progress_callback(percent):
            progress_queue.put(percent)

        def run_analysis():
            try:
                result_holder[0] = run_stacking_simulation_logic(batch_id, progress_callback=progress_callback)
            except Exception as e:
                import traceback
                traceback.print_exc()
                result_holder[0] = {"error": str(e)}

        thread = threading.Thread(target=run_analysis)
        thread.start()

        def generate():
            last_sent = -1
            while thread.is_alive():
                try:
                    p = progress_queue.get(timeout=0.05)
                    if p > last_sent:
                        last_sent = p
                        yield f"data: {{\"percent\": {p:.2f}}}\n\n"
                except queue.Empty:
                    pass
            thread.join()
            result = result_holder[0]
            try:
                result_str = json.dumps(result, default=str)
            except (TypeError, ValueError):
                result_str = json.dumps({"error": "Serialization failed", "stacks": []})
            yield f"data: {{\"percent\": 100.0, \"result\": {result_str}}}\n\n"

        return Response(
            stream_with_context(generate()),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no',
                'Connection': 'keep-alive',
            }
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@stack_bp.route("/list", methods=["GET"])
def list_history():
    conn = get_conn()
    cur = conn.cursor()
    try:
        # [원복] grouped_data를 직접 조회하여 모든 이력(1번 포함)이 나오도록 수정
        # 시각화 페이지 드롭다운용 (원본 데이터 기준)
        sql = """
            SELECT tsv_num, MAX(created_at) as created_at, COUNT(DISTINCT group_number) as stack_count 
            FROM grouped_data 
            GROUP BY tsv_num 
            ORDER BY tsv_num DESC
        """
        cur.execute(sql)
        rows = cur.fetchall()
        
        history = []
        for r in rows:
            if isinstance(r, dict):
                history.append(r)
            else:
                history.append({
                    "tsv_num": r[0],
                    "created_at": str(r[1]),
                    "stack_count": r[2]
                })
        return jsonify(history)
    except Exception as e:
        print(f"[List Error] {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        cur.close()
        conn.close()

@stack_bp.route("/result/<int:tsv_num>", methods=["GET"])
def get_result(tsv_num):
    conn = get_conn()
    cur = conn.cursor()
    try:
        # [수정] chip_data와 조인하여 원본 칩 생성 시간(created_at) 조회 + stack_yield 조회
        sql = """
            SELECT 
                g.group_number, 
                g.position_in_group, 
                g.chip_uid, 
                g.failure_type, 
                g.die_status, 
                g.tsv_status,
                c.created_at as chip_created_at,
                g.stack_yield
            FROM grouped_data g
            LEFT JOIN chip_data c ON g.chip_uid = c.chip_uid
            WHERE g.tsv_num = %s
            ORDER BY g.group_number, g.position_in_group
        """
        cur.execute(sql, (tsv_num,))
        rows = cur.fetchall()
            
        stacks_map = {}
        # 스택별 저장된 수율을 담아둘 맵
        stack_yield_map = {}
        
        for r in rows:
            if isinstance(r, dict):
                g_num = r['group_number']
                pos = r['position_in_group']
                uid = r['chip_uid']
                ftype = r['failure_type']
                dstatus = r['die_status']
                tstatus_str = r['tsv_status']
                c_created = r.get('chip_created_at')
                s_yield = r.get('stack_yield')
            else:
                g_num = r[0]
                pos = r[1]
                uid = r[2]
                ftype = r[3]
                dstatus = r[4]
                tstatus_str = r[5]
                c_created = r[6]
                s_yield = r[7]
            
            if g_num not in stacks_map:
                stacks_map[g_num] = []
            
            # 저장된 수율이 있으면 맵에 기록 (모든 레이어가 같은 값을 가질 것임)
            if s_yield is not None:
                stack_yield_map[g_num] = float(s_yield)
            
            # die_status 처리
            try:
                ds = int(float(dstatus)) if dstatus else 1 
            except:
                ds = 1

            # tsv_status 파싱 (DB에는 JSON String으로 저장됨)
            tsv_matrix = []
            if tstatus_str:
                try:
                    loaded = json.loads(tstatus_str)
                    if isinstance(loaded, list):
                        tsv_matrix = loaded
                except:
                    tsv_matrix = []

            # Yield 계산 (0: 정상, 1: 불량)
            # 0 개수 / 전체 개수
            chip_yield = 0.0
            if tsv_matrix:
                try:
                    # 2D list flatten or just count
                    arr = np.array(tsv_matrix)
                    total_p = arr.size
                    if total_p > 0:
                        count_0 = np.sum(arr == 0)
                        chip_yield = (float(count_0) / float(total_p)) * 100.0
                except Exception as e:
                    print(f"Yield Calc Error: {e}")
                    chip_yield = 0.0

            layer = {
                "layer_idx": pos,
                "chip_id": uid,
                "cluster_label": -1,
                "mapped_type": ftype,
                "failure_type": ftype,
                "die_status": ds,
                "tsv_matrix": tsv_matrix,
                "chip_yield": chip_yield,
                "created_at": str(c_created) if c_created else "N/A"
            }
            stacks_map[g_num].append(layer)

        stacks_result = []
        for g_num in sorted(stacks_map.keys()):
            layers = stacks_map[g_num]
            
            # --- [NEW] Vertical Stacking Yield & Grade Calculation ---
            
            # 1. DB에 저장된 값이 있으면 우선 사용
            final_yield = 0.0
            if g_num in stack_yield_map:
                 final_yield = stack_yield_map[g_num]
            else:
                # 저장된 값이 없으면 (옛날 데이터 등) 직접 계산
                # 1. Collect all TSV matrices
                matrices = []
                for l in layers:
                    if l['tsv_matrix'] and len(l['tsv_matrix']) > 0:
                        try:
                            matrices.append(np.array(l['tsv_matrix']))
                        except:
                            pass
                
                if len(matrices) > 0:
                    try:
                        # Assumption: All matrices are same size (e.g. 32x32)
                        # Stack them along a new axis: (8, 32, 32)
                        stack_arr = np.array(matrices)
                        
                        # 1. Vertical Connectivity Check (Candidates for defect)
                        merged_defect_map = np.max(stack_arr, axis=0) # 0 or 1
                        
                        rows, cols = merged_defect_map.shape
                        total_pins = rows * cols
                        
                        # 2. Redundancy Logic (Repair Check)
                        padded_map = np.pad(merged_defect_map, pad_width=1, mode='constant', constant_values=1)
                        real_defect_count = 0
                        
                        for r in range(rows):
                            for c in range(cols):
                                if merged_defect_map[r, c] == 1:
                                    # Check 3x3 neighbors
                                    neighbors = padded_map[r:r+3, c:c+3]
                                    if np.any(neighbors == 0):
                                        pass # Repaired
                                    else:
                                        real_defect_count += 1
                                        
                        final_yield = ((total_pins - real_defect_count) / total_pins) * 100.0
                        
                    except Exception as e:
                        print(f"Stack Calc Error: {e}")
                        # Fallback: average of individual yields
                        individual_yields = [l['chip_yield'] for l in layers]
                        final_yield = sum(individual_yields) / len(individual_yields) if individual_yields else 0.0

            final_grade = "N/A"
            
            # [수정] Grading Logic Simplified (Yield Only)
            # A: >= 96.0
            # B: >= 92.0
            # C: < 92.0
            
            if final_yield >= 96.0:
                final_grade = "A"
            elif final_yield >= 92.0:
                final_grade = "B"
            else:
                final_grade = "C"

            stacks_result.append({
                "stack_id": f"STACK_DB_{tsv_num}_{g_num}",
                "score": "Loaded",
                "final_grade": final_grade,
                "final_yield": final_yield,
                "layers": layers
            })
            
        return jsonify({
            "batch_id": f"DB_LOAD_{tsv_num}",
            "stacks": stacks_result
        })

    except Exception as e:
        print(f"[Result Error] {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        cur.close()
        conn.close()
