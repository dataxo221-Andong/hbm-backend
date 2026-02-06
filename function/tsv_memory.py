import numpy as np
import cv2
import random

# 기본 칩 사이즈
GRID_SIZE = (32, 32)

# ========================
# 패턴 생성 함수
# ========================
def gen_none(shape, noise_range=(0, 3)):
    matrix = np.zeros(shape, dtype=np.uint8)
    h, w = shape
    low, high = noise_range
    num_noise = random.randint(low, high)
    for _ in range(num_noise):
        rx, ry = random.randint(0, w - 1), random.randint(0, h - 1)
        matrix[ry, rx] = 1
    return matrix


def _add_background_noise(matrix, density_range=(0.005, 0.03)):
    """
    패턴 주변에 랜덤 노이즈(점 불량) 추가
    """
    h, w = matrix.shape
    density = random.uniform(*density_range)
    noise = np.random.choice([0, 1], size=(h, w), p=[1 - density, density])
    # 기존 매트릭스와 합치기 (OR 연산)
    matrix = np.maximum(matrix, noise.astype(np.uint8))
    return matrix


def gen_center(shape):
    matrix = np.zeros(shape, dtype=np.uint8)
    h, w = shape
    cx = (w // 2) + random.randint(-2, 2)
    cy = (h // 2) + random.randint(-2, 2)
    radius = random.uniform(6, 10)
    y_idx, x_idx = np.ogrid[:h, :w]
    mask = (x_idx - cx) ** 2 + (y_idx - cy) ** 2 <= radius**2
    
    # [수정] 밀도를 40~60%로 낮춤 (기존 95% 이상) -> 덜 빽빽하게
    density = random.uniform(0.4, 0.6)
    prob_mask = np.random.rand(h, w) < density
    
    matrix[mask & prob_mask] = 1
    return _add_background_noise(matrix)


def gen_donut(shape):
    matrix = np.zeros(shape, dtype=np.uint8)
    h, w = shape
    cx, cy = w // 2, h // 2
    r_in = random.uniform(4, 6)
    r_out = random.uniform(11, 14)
    y_idx, x_idx = np.ogrid[:h, :w]
    dist_sq = (x_idx - cx) ** 2 + (y_idx - cy) ** 2
    mask = (dist_sq >= r_in**2) & (dist_sq <= r_out**2)
    
    # [수정] 밀도를 40~60%로 낮춤
    density = random.uniform(0.4, 0.6)
    prob_mask = np.random.rand(h, w) < density
    
    matrix[mask & prob_mask] = 1
    return _add_background_noise(matrix)


def gen_edge_ring(shape):
    matrix = np.zeros(shape, dtype=np.uint8)
    h, w = shape
    thickness = random.randint(2, 4)
    
    # 테두리 마스크 생성
    border_mask = np.zeros(shape, dtype=bool)
    border_mask[:thickness, :] = True
    border_mask[-thickness:, :] = True
    border_mask[:, :thickness] = True
    border_mask[:, -thickness:] = True
    
    # [수정] 테두리를 꽉 채우지 않고 40~60%만 채움
    density = random.uniform(0.4, 0.6)
    prob_mask = np.random.rand(h, w) < density
    
    matrix[border_mask & prob_mask] = 1
    return _add_background_noise(matrix)


def gen_edge_loc(shape):
    matrix = np.zeros(shape, dtype=np.uint8)
    h, w = shape
    side = random.choice(["top", "bottom", "left", "right"])
    cluster_r = random.uniform(5, 9)
    if side == "top":
        cx, cy = random.randint(5, w - 5), 0
    elif side == "bottom":
        cx, cy = random.randint(5, w - 5), h
    elif side == "left":
        cx, cy = 0, random.randint(5, h - 5)
    else:
        cx, cy = w, random.randint(5, h - 5)
    y_idx, x_idx = np.ogrid[:h, :w]
    mask = (x_idx - cx) ** 2 + (y_idx - cy) ** 2 <= cluster_r**2
    
    # [수정] 밀도 낮춤
    density = random.uniform(0.4, 0.6)
    prob_mask = np.random.rand(h, w) < density
    
    matrix[mask & prob_mask] = 1
    return _add_background_noise(matrix)


def gen_loc(shape):
    matrix = np.zeros(shape, dtype=np.uint8)
    h, w = shape
    cx, cy = random.randint(8, w - 8), random.randint(8, h - 8)
    radius = random.uniform(3, 6)
    y_idx, x_idx = np.ogrid[:h, :w]
    mask = (x_idx - cx) ** 2 + (y_idx - cy) ** 2 <= radius**2
    
    # [수정] 밀도 낮춤
    density = random.uniform(0.4, 0.6)
    prob_mask = np.random.rand(h, w) < density
    
    matrix[mask & prob_mask] = 1
    return _add_background_noise(matrix)


def gen_scratch(shape):
    matrix = np.zeros(shape, dtype=np.uint8)
    h, w = shape
    num_lines = random.randint(1, 2)
    for _ in range(num_lines):
        p1 = (random.randint(0, w), random.randint(0, h))
        p2 = (random.randint(0, w), random.randint(0, h))
        thickness = random.randint(1, 2)
        cv2.line(matrix, p1, p2, 1, thickness)
    noise = np.random.rand(h, w)
    matrix[noise < 0.05] = 0
    return _add_background_noise(matrix)


def gen_random(shape):
    density = random.uniform(0.25, 0.45)
    matrix = np.random.choice([0, 1], size=shape, p=[1 - density, density])
    return matrix.astype(np.uint8)


def gen_near_full(shape):
    density = random.uniform(0.7, 0.9)
    matrix = np.random.choice([0, 1], size=shape, p=[1 - density, density])
    return matrix.astype(np.uint8)


# ========================
# 패턴 매핑
# ========================
PATTERN_MAPPING = {
    "Center": gen_center,
    "Donut": gen_donut,
    "Edge-Ring": gen_edge_ring,
    "Edge-Loc": gen_edge_loc,
    "Loc": gen_loc,
    "Scratch": gen_scratch,
    "Random": gen_random,
    "Near-full": gen_near_full,
    "None": gen_none,
    "none": gen_none,
}


# ========================
# TSV matrix 주입 함수 (메모리 기반 - 파일 없이)
# ========================
def process_chip_list(chip_list):
    """
    메모리 리스트에서 직접 TSV 매트릭스 생성 (파일 저장 없이)
    """
    result_list = []
    for chip_data in chip_list:
        try:
            f_type = str(chip_data.get("failure_type", "None"))
            status = int(chip_data.get("tsv_status", 0) or 0)

            if status == 1:
                # good chip도 약간의 noise를 넣는 기존 로직 유지
                target_matrix = gen_none(GRID_SIZE, noise_range=(2, 5)) if f_type != "None" else gen_none(GRID_SIZE, (0, 3))
            elif status == 2:
                generator = PATTERN_MAPPING.get(f_type, gen_random)
                target_matrix = generator(GRID_SIZE)
            else:
                target_matrix = gen_none(GRID_SIZE)

            chip_data["tsv_matrix"] = target_matrix
            result_list.append(chip_data)
        except Exception as e:
            print(f"Error processing chip {chip_data.get('chip_uid', 'unknown')}: {e}")
            continue

    return result_list



