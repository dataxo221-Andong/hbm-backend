import pandas as pd
import numpy as np
import os
import sys

# 설정: 분석할 파일 경로와 보고 싶은 Chip UID
PKL_PATH = "..\models\BATCH_20260130_162314.pkl"

# 여기에 확인하고 싶은 chip_uid를 입력하세요. None으로 두면 첫 번째 칩을 보여줍니다.
# 예: TARGET_CHIP_UID = "260130X15QOJ1608X3Y8D2"
TARGET_CHIP_UID = "260130BTUCR31623X17Y23D2"

def visualize_matrix(mat, title="TSV Matrix"):
    """행렬 데이터를 시각화하여 출력"""
    print(f"\n[{title}]")
    
    if isinstance(mat, list):
        mat = np.array(mat)
    
    if not isinstance(mat, np.ndarray):
        print(" [!] 데이터를 numpy array로 변환할 수 없습니다.")
        return

    if mat.ndim != 2:
        print(f" [!] 2D 행렬이 아닙니다. Shape: {mat.shape}")
        return
        
    h, w = mat.shape
    print(f"  - Shape: {mat.shape}")
    
    # 1의 개수 (패턴 존재 여부 확인용)
    ones_count = np.sum(mat > 0)
    print(f"  - Pattern Pixels (Sum): {ones_count} / {h*w}")
    
    print("  - Visualization (■: Defect, .: Empty)")
    print("   " + "".join([str(i%10) for i in range(w)]))
    print("   " + "-" * w)
    for r in range(h):
        row_str = ""
        for c in range(w):
            val = mat[r, c]
            # 값이 1 이상이면 불량(■), 0이면 정상(.)
            char = "■" if val > 0 else "."
            row_str += char
        print(f"{r:2d}|{row_str}|")
    print("   " + "-" * w)

def inspect_pkl():
    if not os.path.exists(PKL_PATH):
        print(f"[Error] 파일을 찾을 수 없습니다: {PKL_PATH}")
        return

    try:
        df = pd.read_pickle(PKL_PATH)
        if isinstance(df, list):
            df = pd.DataFrame(df)
            
        print(f"[Info] 로드 성공: {len(df)}개의 칩 데이터가 있습니다.")
        
        # 타겟 칩 찾기
        target = None
        
        if TARGET_CHIP_UID:
            print(f"[Search] UID '{TARGET_CHIP_UID}' 검색 중...")
            subset = df[df['chip_uid'] == TARGET_CHIP_UID]
            if len(subset) > 0:
                target = subset.iloc[0]
            else:
                print(f"[Warning] 해당 UID를 찾을 수 없습니다.")
        
        # 타겟이 없으면 (또는 못 찾았으면) 랜덤/첫번째 선택
        if target is None:
            print("\n[List] 확인 가능한 칩 UID 목록 (상위 5개):")
            for idx, uid in enumerate(df['chip_uid'].head(5)):
                print(f"  {idx+1}. {uid} ({df.iloc[idx].get('failure_type', 'Unknown')})")
            
            print("\n>> 가장 첫 번째 칩을 자동으로 선택하여 보여줍니다.")
            target = df.iloc[0]

        # 정보 출력
        print("\n" + "="*50)
        print(f" Chip Info: {target['chip_uid']}")
        print("="*50)
        print(f" - Location: ({target.get('coor_x')}, {target.get('coor_y')})")
        print(f" - Failure Type: {target.get('failure_type')}")
        print(f" - Die Status: {target.get('die_status')}")
        
        # Matrix 시각화
        tsv_data = target.get('tsv_matrix')
        visualize_matrix(tsv_data)

    except Exception as e:
        print(f"[Error] 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    inspect_pkl()
