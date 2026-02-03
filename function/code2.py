import os, math, joblib, time, concurrent.futures as cf, threading
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models
import cv2
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side

# GPU 사용 가능 시 활용
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = True
    print(f"CUDA 사용: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA 사용 불가: CPU로 실행합니다.")

# ===== 경로 =====
# 현재 파일 기준 경로
here = os.path.dirname(os.path.abspath(__file__))

# 하위 폴더
pkl_dir = os.path.join(here, "pkl")
result_dir = os.path.join(here, "result")

# 파일명
pkl_name = "BATCH_20260131_144222.pkl"
model_name = "wafer_classifier.pth"
kmeans_name = "kmeans_model.pkl"  # 루트에 위치

# 경로 지정
pkl_path = os.path.join(pkl_dir, pkl_name)
model_path = os.path.join(here, model_name)
kmeans_path = os.path.join(here, kmeans_name)

for path in [pkl_path, model_path, kmeans_path]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"필수 파일 없음: {path}")

# ===== 데이터 로딩 =====
df = pd.read_pickle(pkl_path)
if isinstance(df, list):
    df = pd.DataFrame(df)

# 컬럼명 정규화
if "failure_type" not in df.columns:
    df["failure_type"] = df["failure_type"]

# failure_type 처리: 이미 문자열인 경우와 리스트인 경우 모두 처리
sample = df['failure_type'].iloc[0] if len(df) > 0 else None
if sample is not None:
    if isinstance(sample, list):
        # 리스트인 경우 첫 번째 요소 추출
        df['failure_type'] = df['failure_type'].apply(
            lambda x: x[0][0] if isinstance(x, list) and len(x) > 0 and isinstance(x[0], (list, tuple)) and len(x[0]) > 0 
            else (x[0] if isinstance(x, list) and len(x) > 0 else '')
        )
        df = df[df['failure_type'].apply(lambda x: len(str(x)) > 0)]
    elif not isinstance(sample, str):
        # 다른 타입인 경우 문자열로 변환
        df['failure_type'] = df['failure_type'].astype(str)

target_labels = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Near-full', 'Random', 'Scratch', 'None']
df = df[df['failure_type'].isin(target_labels)].reset_index(drop=True)

print(f"[Data] 필터링 후 데이터 개수: {len(df)}")
if len(df) == 0:
    raise ValueError("필터링 후 유효한 데이터가 없습니다. failure_type을 확인하세요.")

# ===== Dataset =====
class WaferDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe
        self.label_map = {label: i for i, label in enumerate(target_labels)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # tsv_matrix 사용 (waferMap 대신)
        tsv_matrix = self.data.iloc[idx]['tsv_matrix']
        # 이미 numpy array인지 확인
        if not isinstance(tsv_matrix, np.ndarray):
            tsv_matrix = np.array(tsv_matrix)
        # 32x32를 64x64로 리사이즈
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

# ===== Feature Extractor =====
model = models.resnet18(weights=None)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 9)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device).eval()
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

        # 진행 상황 및 ETA 표시 (과도한 출력 방지)
        if batch_idx == 1 or batch_idx % 20 == 0 or batch_idx == total_batches:
            now = time.perf_counter()
            elapsed = now - start_t
            avg_per_batch = elapsed / batch_idx
            remaining = avg_per_batch * (total_batches - batch_idx)
            print(
                f"[Feature] {batch_idx}/{total_batches} "
                f"elapsed={elapsed:.1f}s eta={remaining/60:.1f}m"
            )
            last_log_t = now

if device.type == 'cuda':
    torch.cuda.synchronize()
    print(
        f"GPU 메모리 사용량(MB): "
        f"current={torch.cuda.memory_allocated()/1024**2:.1f}, "
        f"max={torch.cuda.max_memory_allocated()/1024**2:.1f}"
    )

# features 리스트가 비어있는 경우 에러 처리
if len(features) == 0:
    raise ValueError("Feature 추출 실패: features 리스트가 비어있습니다. 데이터가 올바르게 로드되었는지 확인하세요.")

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
    if mask_i.shape != W_critical.shape or mask_j.shape != W_critical.shape:
        raise ValueError(
            f"mask shape 불일치: {mask_i.shape}, {mask_j.shape}, W={W_critical.shape}"
        )
    return np.sum(W_critical * mask_i * mask_j)

# lambda 합=1
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

# 같은 failure_type 조합 금지 (TSV 적층 시 물리적 불량 방지)
# 엄격한 규칙: 모든 같은 타입 조합 금지
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
            elapsed = time.perf_counter() - mask_start
            avg = elapsed / (i + 1)
            eta = avg * (mask_total - (i + 1))
            print(f"[Mask] {i+1}/{mask_total} elapsed={elapsed:.1f}s eta={eta/60:.1f}m")

# ===== 8개 단위 매칭(그룹화) =====
tau = 5.0
group_size = 8
N = len(df)

pair_total = N * (N - 1) // 2
pair_start = time.perf_counter()
print(f"[Group] cost matrix 계산 시작: pairs={pair_total}")

# 비용 행렬 (작은 데이터 기준, 큰 데이터면 메모리 주의)
cost_mat = np.full((N, N), np.inf, dtype=np.float32)
for i in range(N):
    cost_mat[i, i] = 0.0

# 진행 상황 출력을 위한 변수
processed_pairs = 0
last_progress = 0
progress_interval = 0.04  # 4% 단위

for i in range(N):
    for j in range(i + 1, N):
        c = cost_fn(i, j, masks)
        if math.isinf(c):
            continue
        cost_mat[i, j] = c
        cost_mat[j, i] = c
        
        # 진행 상황 출력 (4% 단위)
        processed_pairs += 1
        current_progress = processed_pairs / pair_total
        if current_progress - last_progress >= progress_interval:
            elapsed = time.perf_counter() - pair_start
            avg_per_pair = elapsed / processed_pairs
            remaining = avg_per_pair * (pair_total - processed_pairs)
            print(
                f"[Group] cost matrix 진행: {current_progress*100:.1f}% "
                f"({processed_pairs}/{pair_total}) "
                f"elapsed={elapsed:.1f}s eta={remaining/60:.1f}m"
            )
            last_progress = current_progress

elapsed = time.perf_counter() - pair_start
print(f"[Group] cost matrix 완료: {elapsed:.1f}s")

def _pick_next(group, remaining):
    # 그룹 내 평균 비용이 최소가 되는 후보 선택
    # 단, 같은 failure_type은 제외 (TSV 적층 시 물리적 불량 방지)
    best_j = None
    best_score = float("inf")
    
    # 현재 그룹의 failure_type 집합 (다이버시티 보장)
    # 이론적 근거: 같은 타입 적층 시 물리적 불량 누적 방지
    group_types = set(df.iloc[g]['failure_type'] for g in group)
    
    for j in remaining:
        # 같은 failure_type이 이미 그룹에 있으면 제외
        candidate_type = df.iloc[j]['failure_type']
        if candidate_type in group_types:
            continue
            
        costs = [cost_mat[j, g] for g in group]
        # 모두 inf면 스킵
        if all(math.isinf(c) for c in costs):
            continue
        score = float(np.mean(costs))
        if score < best_score:
            best_score = score
            best_j = j
    return best_j

remaining = list(range(N))
groups = []
max_attempts = len(remaining) * 2  # 최대 시도 횟수 (무한 루프 방지)
attempt_count = 0

# 진행 상황 출력을 위한 변수 (10초 단위)
grouping_start = time.perf_counter()
last_log_time = grouping_start
log_interval = 10.0  # 10초 단위
max_possible_groups = N // group_size  # 이론상 최대 그룹 수
initial_remaining = N

print(f"[Grouping] 그룹화 시작: 최대 {max_possible_groups}개 그룹 생성 가능")

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
        # 8개를 만들 수 없으면 seed를 다시 remaining에 추가하고 다음 시도
        # 부분 그룹은 절대 허용하지 않음 (무조건 8층 적층)
        remaining.append(seed)
        # 이미 추가된 요소들도 remaining에 다시 추가
        for item in group[1:]:  # seed 제외
            if item not in remaining:
                remaining.append(item)
        continue
    
    # 성공적으로 8개 그룹 생성
    groups.append(group)
    
    # 진행 상황 출력 (10초 단위)
    now = time.perf_counter()
    if now - last_log_time >= log_interval:
        elapsed = now - grouping_start
        processed = initial_remaining - len(remaining)
        progress_pct = (processed / initial_remaining) * 100
        groups_pct = (len(groups) / max_possible_groups) * 100 if max_possible_groups > 0 else 0
        
        # ETA 계산
        if len(groups) > 0:
            avg_time_per_group = elapsed / len(groups)
            remaining_groups = max_possible_groups - len(groups)
            eta = avg_time_per_group * remaining_groups
        else:
            eta = 0
        
        print(
            f"[Grouping] {progress_pct:.1f}% 처리 | "
            f"그룹: {len(groups)}/{max_possible_groups} ({groups_pct:.1f}%) | "
            f"남은 데이터: {len(remaining)}개 | "
            f"elapsed={elapsed:.1f}s eta={eta:.1f}s"
        )
        last_log_time = now

# 최종 완료 메시지
grouping_elapsed = time.perf_counter() - grouping_start
print(f"[Grouping] 완료: {grouping_elapsed:.1f}s")

if attempt_count >= max_attempts:
    print(f"[Grouping] 경고: 최대 시도 횟수({max_attempts})에 도달했습니다.")

# ===== 결과 저장 (TXT + XLSX) =====
if not groups:
    print("유효한 8개 그룹을 만들 수 없습니다. 모두 폐기합니다.")
else:
    print(f"그룹 수: {len(groups)} (모두 {group_size}개씩)")
    
    # 그룹 결과를 unsigned int로 변환
    grouped_array = np.array(groups, dtype=np.uint32)
    created_at = pd.Timestamp.now().isoformat()
    
    # result 폴더 없으면 생성
    os.makedirs(result_dir, exist_ok=True)
    date_str = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    
    # ===== TXT 저장 =====
    grouped_txt_path = os.path.join(result_dir, f"grouped({date_str}).txt")
    with open(grouped_txt_path, "w", encoding="utf-8") as f:
        np.set_printoptions(threshold=np.inf, linewidth=np.inf)
        f.write("array([")
        for i, row in enumerate(grouped_array):
            if i > 0:
                f.write(",\n       ")
            f.write("[")
            f.write(", ".join(map(str, row)))
            f.write("]")
        f.write("],\n      shape=(")
        f.write(f"{grouped_array.shape[0]}, {grouped_array.shape[1]}")
        f.write(f"), dtype={grouped_array.dtype})")
        f.write(f"\ncreated_at={created_at}")
        np.set_printoptions(threshold=1000, linewidth=75)
    print(f"[Export] TXT 저장 완료: {grouped_txt_path}")
    
    # ===== XLSX 저장 =====
    grouped_xlsx_path = os.path.join(result_dir, f"grouped({date_str}).xlsx")
    
    wb = Workbook()
    ws = wb.active
    ws.title = "Grouped Data"
    
    # 스타일 정의
    header_font_white = Font(bold=True, size=11, color="FFFFFF")
    header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
    thin_border = Border(
        left=Side(style='thin'),
        right=Side(style='thin'),
        top=Side(style='thin'),
        bottom=Side(style='thin')
    )
    
    # 헤더 작성
    headers = [
        "grouped_number",
        "position_in_group", 
        "data_index",
        "chip_uid",
        "lot_name",
        "failure_type",
        "created_at",
        "tsv_status",
        "die_status",
        "tsv_coordination"
    ]
    
    for col_idx, header in enumerate(headers, start=1):
        cell = ws.cell(row=1, column=col_idx, value=header)
        cell.font = header_font_white
        cell.fill = header_fill
        cell.border = thin_border
        cell.alignment = Alignment(horizontal='center', vertical='center')
    
    # 데이터 작성
    row_idx = 2
    for gi, group in enumerate(groups):
        for position, wafer_idx in enumerate(group):
            row_data = df.iloc[wafer_idx]
            
            # grouped_number
            ws.cell(row=row_idx, column=1, value=gi).border = thin_border
            # position_in_group (1부터 시작)
            ws.cell(row=row_idx, column=2, value=position + 1).border = thin_border
            # data_index
            ws.cell(row=row_idx, column=3, value=int(wafer_idx)).border = thin_border
            # chip_uid
            chip_uid = row_data.get('chip_uid', '') if 'chip_uid' in row_data else ''
            ws.cell(row=row_idx, column=4, value=chip_uid).border = thin_border
            # lot_name
            lot_name = row_data.get('lot_name', '') if 'lot_name' in row_data else ''
            ws.cell(row=row_idx, column=5, value=lot_name).border = thin_border
            # failure_type
            ws.cell(row=row_idx, column=6, value=row_data['failure_type']).border = thin_border
            # created_at
            ws.cell(row=row_idx, column=7, value=created_at).border = thin_border
            # tsv_status
            tsv_status = row_data.get('tsv_status', '') if 'tsv_status' in row_data else ''
            ws.cell(row=row_idx, column=8, value=tsv_status).border = thin_border
            # die_status
            die_status = row_data.get('die_status', '') if 'die_status' in row_data else ''
            ws.cell(row=row_idx, column=9, value=die_status).border = thin_border
            # tsv_coordination
            tsv_coord = row_data.get('tsv_coordination', '') if 'tsv_coordination' in row_data else ''
            ws.cell(row=row_idx, column=10, value=str(tsv_coord)).border = thin_border
            
            # 중앙 정렬
            for col in range(1, 11):
                ws.cell(row=row_idx, column=col).alignment = Alignment(horizontal='center', vertical='center')
            
            row_idx += 1
    
    # 열 너비 조정
    col_widths = [15, 18, 12, 20, 20, 14, 22, 12, 12, 20]
    for i, width in enumerate(col_widths):
        ws.column_dimensions[chr(ord('A') + i)].width = width
    
    # 저장
    wb.save(grouped_xlsx_path)
    print(f"[Export] XLSX 저장 완료: {grouped_xlsx_path}")

# ===== 시각화: 그룹 그래프 =====
if groups:
    Gg = nx.Graph()
    for gi, group in enumerate(groups):
        # 그룹 내부 완전 연결 (클리크)
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                if cost_mat[group[i], group[j]] < tau:
                    Gg.add_edge(group[i], group[j])

    pos = nx.spring_layout(Gg, seed=42)
    palette = plt.get_cmap("tab20")
    node_colors = []
    for node in Gg.nodes():
        # 노드가 속한 그룹 찾기
        gidx = next((i for i, g in enumerate(groups) if node in g), 0)
        node_colors.append(palette(gidx % 20))

    plt.figure(figsize=(9, 9))
    nx.draw_networkx_nodes(Gg, pos, node_size=30, node_color=node_colors, alpha=0.85)
    nx.draw_networkx_edges(Gg, pos, edge_color="#999999", width=0.8, alpha=0.5)
    plt.title(f"Grouped Matching Graph (size={group_size})")
    plt.axis("off")

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=palette(i % 20),
               markersize=6, label=f"Group {i}") for i in range(len(groups))
    ]
    plt.legend(handles=legend_handles, loc="upper right", ncol=2)
    plt.show()

    # ===== 시각화: Plot2 (8개 그룹 구성 표시) =====
    plt.figure(figsize=(10, 5))
    for gi, group in enumerate(groups):
        x = [gi] * len(group)
        y = list(range(1, len(group) + 1))
        plt.scatter(x, y, color=palette(gi % 20), s=60, alpha=0.9)
        for yi, node in zip(y, group):
            plt.text(gi, yi + 0.1, str(node), ha="center", va="bottom", fontsize=4)

    plt.title("Plot2: Group Composition (8-per group)", fontsize=7)
    plt.xlabel("Group Index", fontsize=7)
    plt.ylabel("Position in Group", fontsize=7)
    plt.xticks(range(len(groups)), fontsize=7)
    plt.yticks(range(1, group_size + 1), fontsize=7)
    plt.grid(True, axis="y", alpha=0.3)
    plt.show()
