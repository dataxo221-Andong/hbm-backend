import os
import pandas as pd
import numpy as np
import re

# ===== 파일 경로 =====
here = os.path.dirname(os.path.abspath(__file__))
result_dir = os.path.join(here, "..", "result")

txt_files = [
    os.path.join(result_dir, f)
    for f in os.listdir(result_dir)
    if f.lower().endswith(".txt")
]
if not txt_files:
    raise FileNotFoundError(f"result 폴더에 txt 파일이 없습니다: {result_dir}")

grouped_txt_path = max(txt_files, key=os.path.getmtime)
pkl_path = os.path.join(here, "..", "pkl", "undefined.pkl")  # grouped.txt의 인덱스 범위에 맞는 데이터셋
sql_output_path = os.path.splitext(grouped_txt_path)[0] + ".sql"

print(f"[INFO] 최신 grouped 파일 사용: {grouped_txt_path}")

# ===== grouped.txt 파일 읽기 =====
print("[1/4] grouped.txt 파일 읽는 중...")
with open(grouped_txt_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# shape 정보 추출
shape_match = None
for line in lines:
    if 'shape=' in line:
        shape_match = re.search(r'shape=\((\d+),\s*(\d+)\)', line)
        break

if not shape_match:
    raise ValueError("grouped.txt 파일에서 shape 정보를 찾을 수 없습니다.")

rows = int(shape_match.group(1))
cols = int(shape_match.group(2))

# 각 행 파싱
groups = []
for line in lines:
    line = line.strip()
    if not line or line.startswith('created_at') or 'shape=' in line or 'dtype=' in line:
        continue
    # [숫자, 숫자, ...] 형태 파싱
    if line.startswith('['):
        # 마지막 ] 제거
        if line.endswith(','):
            line = line[:-1]
        if line.endswith(']'):
            line = line[:-1]
        if line.startswith('['):
            line = line[1:]
        # 쉼표로 분리하고 숫자 추출
        try:
            numbers = [int(x.strip()) for x in line.split(',') if x.strip()]
            if len(numbers) == cols:
                groups.append(numbers)
        except ValueError:
            continue

groups_array = np.array(groups, dtype=np.uint32)
print(f"  - 그룹 수: {len(groups_array)}개 (각 그룹당 {cols}개 인덱스)")

# ===== 원본 데이터 로드 =====
print("[2/4] 원본 데이터 로드 중...")
df_original = pd.read_pickle(pkl_path)
if isinstance(df_original, list):
    df_original = pd.DataFrame(df_original)
if "failure_type" not in df_original.columns and "failureType" in df_original.columns:
    df_original["failure_type"] = df_original["failureType"]
print(f"  - 원본 데이터 개수: {len(df_original)}개")

# ===== SQL 데이터 준비 =====
print("[3/4] SQL 데이터 준비 중...")
sql_data = []

for group_idx, group in enumerate(groups_array):
    for pos_in_group, data_idx in enumerate(group):
        if data_idx < len(df_original):
            row_data = {
                'group_number': group_idx + 1,
                'position_in_group': pos_in_group + 1,
                'data_index': int(data_idx),
            }

            # 원본 데이터 정보 추가 
            if 'chip_uid' in df_original.columns:
                row_data['chip_uid'] = df_original.iloc[data_idx]['chip_uid']
            if 'lot_name' in df_original.columns:
                row_data['lot_name'] = df_original.iloc[data_idx]['lot_name']
            if 'failure_type' in df_original.columns:
                row_data['failure_type'] = df_original.iloc[data_idx]['failure_type']
            if 'created_at' in df_original.columns:
                row_data['created_at'] = df_original.iloc[data_idx]['created_at']
            if 'tsv_status' in df_original.columns:
                row_data['tsv_status'] = df_original.iloc[data_idx]['tsv_status']
            if 'die_status' in df_original.columns:
                row_data['die_status'] = df_original.iloc[data_idx]['die_status']
            if 'tsv_coordinate' in df_original.columns:
                row_data['tsv_coordinate'] = df_original.iloc[data_idx]['tsv_coordinate']
            if 'coor_x' in df_original.columns:
                row_data['coor_x'] = df_original.iloc[data_idx]['coor_x']
            if 'coor_y' in df_original.columns:
                row_data['coor_y'] = df_original.iloc[data_idx]['coor_y']

            sql_data.append(row_data)
        else:
            # 범위를 벗어나는 인덱스는 건너뛰기
            pass

df_sql = pd.DataFrame(sql_data)
print(f"  - SQL 행 수: {len(df_sql)}개")

# ===== SQL 파일 저장 =====
print("[4/4] SQL 파일 저장 중...")

def sql_escape(value):
    # 배열/리스트는 문자열(JSON 유사)로 저장
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == 0:
            return "NULL"
        text = str(value).replace("'", "''")
        return f"'{text}'"
    # 스칼라 결측치 처리
    if pd.isna(value):
        return "NULL"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return str(float(value))
    # 문자열 처리
    text = str(value).replace("'", "''")
    return f"'{text}'"

columns = list(df_sql.columns)

with open(sql_output_path, 'w', encoding='utf-8') as f:
    f.write("START TRANSACTION;\n")
    f.write("CREATE TABLE IF NOT EXISTS grouped_data (\n")
    f.write("  group_number INTEGER,\n")
    f.write("  position_in_group INTEGER,\n")
    f.write("  data_index INTEGER")

    optional_columns = [col for col in columns if col not in ['group_number', 'position_in_group', 'data_index']]
    for col in optional_columns:
        # 기본적으로 TEXT로 저장 (숫자는 INSERT 시 자동 변환 가능)
        f.write(f",\n  {col} TEXT")
    f.write("\n);\n")

    col_list = ", ".join(columns)
    for _, row in df_sql.iterrows():
        values = ", ".join(sql_escape(row[col]) for col in columns)
        f.write(f"INSERT INTO grouped_data ({col_list}) VALUES ({values});\n")

    f.write("COMMIT;\n")

print(f"[완료] SQL 파일 저장: {sql_output_path}")
print(f"   - 총 {len(groups_array)}개 그룹")
print(f"   - 총 {len(df_sql)}개 행")
if len(df_sql) < len(groups_array) * cols:
    print(f"   - 경고: 일부 인덱스가 데이터 범위를 벗어나 제외되었습니다.")

