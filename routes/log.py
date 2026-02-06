from flask import Blueprint, jsonify
from db import get_conn

# Blueprint 정의
log_bp = Blueprint("log", __name__, url_prefix="/log")

@log_bp.route("/list", methods=["GET"])
def list_history():
    conn = get_conn()
    cur = conn.cursor()
    try:
        # simulation_log 테이블을 조회하여 시뮬레이션 이력 목록 반환
        sql = """
            SELECT tsv_num, created_at, total_stacks, avg_yield, grade_a, grade_b, grade_c
            FROM simulation_log
            ORDER BY tsv_num DESC
        """
        cur.execute(sql)
        rows = cur.fetchall()
        
        history = []
        for r in rows:
            if isinstance(r, dict):
                # DictCursor를 사용하는 경우
                history.append({
                    "tsv_num": r['tsv_num'],
                    "created_at": str(r['created_at']),
                    "stack_count": r['total_stacks'], 
                    "avg_yield": r['avg_yield'],
                    "grade_a": r['grade_a'],
                    "grade_b": r['grade_b'],
                    "grade_c": r['grade_c']
                })
            else:
                # TupleCursor를 사용하는 경우
                history.append({
                    "tsv_num": r[0],
                    "created_at": str(r[1]),
                    "stack_count": r[2], 
                    "avg_yield": r[3],
                    "grade_a": r[4],
                    "grade_b": r[5],
                    "grade_c": r[6]
                })
        return jsonify(history)
    except Exception as e:
        print(f"[Log List Error] {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        cur.close()
        conn.close()
