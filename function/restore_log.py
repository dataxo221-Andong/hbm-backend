import sys
import os
import json
import numpy as np

# 프로젝트 루트 경로를 sys.path에 추가 (db.py 불러오기 위함)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from db import get_conn

def restore_log(target_tsv_num):
    print(f"[{target_tsv_num}번] 시뮬레이션 로그 복구를 시작합니다...")
    
    conn = get_conn()
    cur = conn.cursor()
    
    try:
        # 1. grouped_data와 chip_data를 조인하여 원본 데이터 가져오기
        sql = """
            SELECT 
                g.group_number, 
                g.position_in_group, 
                g.die_status, 
                g.tsv_status,
                g.failure_type,
                g.created_at
            FROM grouped_data g
            WHERE g.tsv_num = %s
            ORDER BY g.group_number, g.position_in_group
        """
        cur.execute(sql, (target_tsv_num,))
        rows = cur.fetchall()
        
        if not rows:
            print(f"Error: tsv_num={target_tsv_num}에 해당하는 데이터가 grouped_data 테이블에 없습니다.")
            return

        print(f"-> 총 {len(rows)}개의 칩 데이터를 가져왔습니다.")

        # 2. 스택별로 데이터 정리
        stacks_map = {}
        last_created_at = None
        
        for r in rows:
            if isinstance(r, dict):
                g_num = r['group_number']
                dstatus = r['die_status']
                tstatus_str = r['tsv_status']
                ftype = r['failure_type']
                created = r['created_at']
            else:
                g_num = r[0]
                dstatus = r[2]
                tstatus_str = r[3]
                ftype = r[4]
                created = r[5]
            
            if last_created_at is None:
                last_created_at = created
                
            if g_num not in stacks_map:
                stacks_map[g_num] = []
            
            # tsv parsing
            tsv_matrix = []
            if tstatus_str:
                try:
                    loaded = json.loads(tstatus_str)
                    if isinstance(loaded, list):
                        tsv_matrix = loaded
                except:
                    pass
            
            # die_status parsing
            try:
                ds = int(float(dstatus)) if dstatus else 1
            except:
                ds = 1

            stacks_map[g_num].append({
                "tsv_matrix": tsv_matrix,
                "die_status": ds,
                "failure_type": ftype
            })

        # 3. 각 스택별 수율/등급 계산
        grade_a_count = 0
        grade_b_count = 0
        grade_c_count = 0
        total_yield_sum = 0.0
        
        stack_count = len(stacks_map)
        print(f"-> 총 {stack_count}개의 스택을 구성했습니다. 계산 시작...")

        for g_num, layers in stacks_map.items():
            # (1) Vertical Stacking & Yield Calculation
            matrices = []
            for l in layers:
                if l['tsv_matrix'] and len(l['tsv_matrix']) > 0:
                    try:
                        matrices.append(np.array(l['tsv_matrix']))
                    except:
                        pass
            
            final_yield = 0.0
            if len(matrices) > 0:
                try:
                    stack_arr = np.array(matrices)
                    merged = np.max(stack_arr, axis=0) # 0 or 1
                    rows, cols = merged.shape
                    total_pins = rows * cols
                    
                    # Redundancy Check
                    padded = np.pad(merged, pad_width=1, mode='constant', constant_values=1)
                    real_defect = 0
                    for r in range(rows):
                        for c in range(cols):
                            if merged[r, c] == 1:
                                neighbors = padded[r:r+3, c:c+3]
                                if not np.any(neighbors == 0):
                                    real_defect += 1
                                    
                    final_yield = ((total_pins - real_defect) / total_pins) * 100.0
                except:
                    # Fallback (개별 칩 수율 평균)
                    pass

            # (2) Grading
            has_critical = any(l['die_status'] == 2 and l['failure_type'] in ['Random', 'Near-full'] for l in layers)
            
            grade = "C"
            if final_yield >= 96.0 and not has_critical:
                grade = "A"
            elif final_yield >= 90.0:
                grade = "B"
            elif final_yield >= 85.0 and not has_critical:
                grade = "B"
            
            # (3) Count
            if grade == "A": grade_a_count += 1
            elif grade == "B": grade_b_count += 1
            else: grade_c_count += 1
            
            total_yield_sum += final_yield

        avg_yield = total_yield_sum / stack_count if stack_count > 0 else 0.0

        # 4. Result Printing & SQL Generation
        print("\n" + "="*40)
        print(f" [계산 완료] tsv_num={target_tsv_num}")
        print(f" - 생성 일시: {last_created_at}")
        print(f" - 총 스택 수: {stack_count}")
        print(f" - 평균 수율: {avg_yield:.4f}%")
        print(f" - 등급 분포: A({grade_a_count}), B({grade_b_count}), C({grade_c_count})")
        print("="*40 + "\n")
        
        # INSERT SQL 생성
        sql_query = f"""
INSERT INTO simulation_log 
(tsv_num, created_at, total_stacks, avg_yield, grade_a, grade_b, grade_c)
VALUES ({target_tsv_num}, '{last_created_at}', {stack_count}, {avg_yield:.4f}, {grade_a_count}, {grade_b_count}, {grade_c_count})
ON DUPLICATE KEY UPDATE 
    created_at=VALUES(created_at), total_stacks=VALUES(total_stacks), 
    avg_yield=VALUES(avg_yield), grade_a=VALUES(grade_a), 
    grade_b=VALUES(grade_b), grade_c=VALUES(grade_c);
"""
        print("아래 SQL을 복사해서 실행하세요:\n")
        print(sql_query)
        
        # 실제 DB에 바로 넣을지 여부 (여기선 자동 실행)
        cur.execute(sql_query)
        conn.commit()
        print("✅ DB에 자동으로 저장되었습니다!")

    except Exception as e:
        print(f"[Error] {e}")
        import traceback
        traceback.print_exc()
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    # tsv_num = 1 복구 실행
    restore_log(1)
