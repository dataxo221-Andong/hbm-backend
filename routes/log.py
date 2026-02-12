from flask import Blueprint, jsonify
from db import get_conn

# Blueprint 정의
log_bp = Blueprint("log", __name__, url_prefix="/log")

@log_bp.route("/list", methods=["GET"])
def list_history():
    conn = get_conn()
    if conn is None:
        return jsonify({"error": "데이터베이스 연결에 실패했습니다."}), 503
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

@log_bp.route("/stats/daily", methods=["GET"])
def get_daily_stats():
    conn = get_conn()
    if conn is None:
        return jsonify({"error": "데이터베이스 연결에 실패했습니다."}), 503
    cur = conn.cursor()
    try:
        # 1. DB에 존재하는 가장 최근 날짜 조회 (SQL에게 위임)
        cur.execute("SELECT MAX(DATE(created_at)) FROM simulation_log")
        row = cur.fetchone()
        
        latest_date = None
        if row:
            if isinstance(row, dict):
                latest_date = row.get("MAX(DATE(created_at))") or row.get("max(date(created_at))")
            else:
                latest_date = row[0]
        
        # 데이터가 아예 없으면 0 리턴
        if not latest_date:
            return jsonify({
                "today": {"total_production": 0, "avg_yield": 0.0, "grade_a_ratio": 0.0, "avg_cycle": 0.0},
                "yesterday": {"total_production": 0, "avg_yield": 0.0, "grade_a_ratio": 0.0, "avg_cycle": 0.0}
            })

        # 2. 기준 날짜(Target)와 비교 날짜(Compare) 설정
        import datetime
        target_date = latest_date # 이미 date 객체이거나 'YYYY-MM-DD' 문자열임
        
        # 문자열이면 date 객체로 변환 (timedelta 계산 위해)
        if isinstance(target_date, str):
            target_date = datetime.datetime.strptime(target_date, "%Y-%m-%d").date()
            
        compare_date = target_date - datetime.timedelta(days=1)

        # 3. 해당 날짜 데이터 조회 (SQL WHERE 절 사용)
        sql_daily = """
            SELECT created_at, total_stacks, avg_yield, grade_a 
            FROM simulation_log 
            WHERE DATE(created_at) = %s
            ORDER BY created_at ASC
        """
        
        # Target Data
        cur.execute(sql_daily, (target_date,))
        today_rows = cur.fetchall()
        
        # Compare Data
        cur.execute(sql_daily, (compare_date,))
        yesterday_rows = cur.fetchall()

        # 4. 통계 계산 함수 (단순화)
        def calc_stats_from_rows(rows):
            if not rows:
                return {"total_production": 0, "avg_yield": 0.0, "grade_a_ratio": 0.0, "avg_cycle": 0.0}
            
            # Row Access Helper
            def get_val(r, idx, key):
                if isinstance(r, dict): return r.get(key)
                return r[idx]

            data_list = []
            for r in rows:
                dt = get_val(r, 0, 'created_at')
                tot = get_val(r, 1, 'total_stacks') or 0
                yld = get_val(r, 2, 'avg_yield') or 0.0
                grd = get_val(r, 3, 'grade_a') or 0
                data_list.append({"dt": dt, "total": tot, "yield": yld, "grade_a": grd})

            total_prod = sum(d["total"] for d in data_list)
            avg_y = sum(d["yield"] for d in data_list) / len(data_list)
            total_grade_a = sum(d["grade_a"] for d in data_list)
            ratio_a = (total_grade_a / total_prod * 100.0) if total_prod > 0 else 0.0
            
            # Cycle Time
            # data_list는 이미 ASC 정렬됨
            intervals = []
            if len(data_list) > 1:
                for i in range(1, len(data_list)):
                    t1 = data_list[i]["dt"]
                    t0 = data_list[i-1]["dt"]
                    
                    # datetime 객체 보장
                    if isinstance(t1, str): t1 = datetime.datetime.strptime(t1, "%Y-%m-%d %H:%M:%S")
                    if isinstance(t0, str): t0 = datetime.datetime.strptime(t0, "%Y-%m-%d %H:%M:%S")
                    
                    diff = (t1 - t0).total_seconds()
                    intervals.append(diff / 60.0)
                avg_cyc = sum(intervals) / len(intervals)
            else:
                avg_cyc = 0.0

            return {
                "total_production": total_prod,
                "avg_yield": avg_y,
                "grade_a_ratio": ratio_a,
                "avg_cycle": avg_cyc
            }

        today_stats = calc_stats_from_rows(today_rows)
        yesterday_stats = calc_stats_from_rows(yesterday_rows)

        return jsonify({
            "today": today_stats,
            "yesterday": yesterday_stats,
            "target_date": str(target_date)
        })

    except Exception as e:
        print(f"[Stats Daily Error] {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        cur.close()
        conn.close()

@log_bp.route("/stats/trend", methods=["GET"])
def get_stats_trend():
    conn = get_conn()
    if conn is None:
        return jsonify({"error": "데이터베이스 연결에 실패했습니다."}), 503
    cur = conn.cursor()
    try:
        # 최근 10회차 시뮬레이션의 수율 및 등급 분포 추세 조회
        sql = """
            SELECT tsv_num, created_at, avg_yield, total_stacks, grade_a, grade_b, grade_c
            FROM simulation_log
            ORDER BY tsv_num DESC
            LIMIT 10
        """
        cur.execute(sql)
        rows = cur.fetchall()
        
        trend_data = []
        for r in rows:
            # Row accessing helper
            def get_val(row, key, idx):
                if isinstance(row, dict): return row.get(key)
                return row[idx]

            tsv_num = get_val(r, 'tsv_num', 0)
            created_at = get_val(r, 'created_at', 1)
            avg_yield = get_val(r, 'avg_yield', 2)
            total = get_val(r, 'total_stacks', 3)
            ga = get_val(r, 'grade_a', 4)
            gb = get_val(r, 'grade_b', 5)
            gc = get_val(r, 'grade_c', 6)
            
            # Calculate A-Grade Ratio
            ratio_a = (ga / total * 100.0) if total and total > 0 else 0.0
            
            trend_data.append({
                "id": tsv_num,
                "date": str(created_at),
                "yield": avg_yield,
                "production": total,
                "grade_a": ga,
                "grade_b": gb,
                "grade_c": gc,
                "grade_a_ratio": ratio_a
            })
        
        # 그래프용이므로 과거->현재 순으로 정렬
        trend_data.reverse()
        
        return jsonify(trend_data)
        
    except Exception as e:
        print(f"[Stats Trend Error] {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        cur.close()
        conn.close()

@log_bp.route("/detail/<int:tsv_num>", methods=["GET"])
def get_batch_detail(tsv_num):
    conn = get_conn()
    if conn is None:
        return jsonify({"error": "데이터베이스 연결에 실패했습니다."}), 503
    cur = conn.cursor()
    try:
        # grouped_data 테이블에서 해당 배치의 적층 케이스 상세 조회
        # idx, final_yield, final_grade, created_at 등
        sql = """
            SELECT idx, final_yield, final_grade, created_at
            FROM grouped_data
            WHERE tsv_num = %s
            ORDER BY idx ASC
        """
        cur.execute(sql, (tsv_num,))
        rows = cur.fetchall()
        
        details = []
        for r in rows:
            def get_val(row, key, idx):
                if isinstance(row, dict): return row.get(key)
                return row[idx]

            idx = get_val(r, 'idx', 0)
            yld = get_val(r, 'final_yield', 1)
            grade = get_val(r, 'final_grade', 2)
            ts = get_val(r, 'created_at', 3)
            
            details.append({
                "case_id": f"HBM-{str(idx).zfill(5)}",
                "yield": yld,
                "grade": grade,
                "time": str(ts)
            })
            
        return jsonify(details)
        
    except Exception as e:
        print(f"[Batch Detail Error] {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        cur.close()
        conn.close()
