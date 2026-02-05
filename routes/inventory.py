from flask import Blueprint, request, jsonify
import pymysql

from db import get_conn

inventory_bp = Blueprint("inventory", __name__, url_prefix="/inventory")


def _parse_int(value, default=None, min_value=None, max_value=None):
    try:
        if value is None or value == "":
            return default
        n = int(value)
        if min_value is not None:
            n = max(n, min_value)
        if max_value is not None:
            n = min(n, max_value)
        return n
    except Exception:
        return default


def _fetch_chips(where_sql="", params=None, limit=200, offset=0):
    conn = get_conn()
    if conn is None:
        return None, (jsonify({"error": "데이터베이스 연결 실패"}), 500)

    params = params or []
    cur = conn.cursor(pymysql.cursors.DictCursor)
    try:
        sql = f"""
            SELECT
                wafer_idx,
                chip_uid,
                tsv_matrix,
                failure_type,
                coor_x,
                coor_y,
                die_status,
                created_at
            FROM chip_data
            {where_sql}
            ORDER BY created_at DESC
            LIMIT %s OFFSET %s
        """
        cur.execute(sql, [*params, limit, offset])
        rows = cur.fetchall() or []
        return rows, None
    finally:
        cur.close()
        conn.close()

def _count_chips(where_sql="", params=None):
    """필터 조건에 맞는 전체 칩 개수 반환"""
    conn = get_conn()
    if conn is None:
        return None, (jsonify({"error": "데이터베이스 연결 실패"}), 500)

    params = params or []
    cur = conn.cursor(pymysql.cursors.DictCursor)
    try:
        sql = f"SELECT COUNT(*) AS total FROM chip_data {where_sql}"
        cur.execute(sql, params)
        row = cur.fetchone() or {}
        return int(row.get("total", 0)), None
    finally:
        cur.close()
        conn.close()


def _fetch_one_chip(where_sql, params):
    conn = get_conn()
    if conn is None:
        return None, (jsonify({"error": "데이터베이스 연결 실패"}), 500)

    cur = conn.cursor(pymysql.cursors.DictCursor)
    try:
        sql = f"""
            SELECT
                wafer_idx,
                chip_uid,
                tsv_matrix,
                failure_type,
                coor_x,
                coor_y,
                die_status,
                created_at
            FROM chip_data
            {where_sql}
            LIMIT 1
        """
        cur.execute(sql, params)
        row = cur.fetchone()
        return row, None
    finally:
        cur.close()
        conn.close()


@inventory_bp.route("/chips", methods=["GET"])
def list_chips():
    """
    칩 목록 조회 API
    GET /inventory/chips
    
    Query params:
      - lot_name: 로트명으로 필터링 (chip_uid의 접두사)
      - wafer_idx: 웨이퍼 인덱스로 필터링
      - failure_type: 불량 유형으로 필터링
      - limit, offset: 페이지네이션
    """
    # 기본값을 크게 잡아(예: 10,000) UI에서 "전체 칩"을 바로 볼 수 있게 함
    limit_raw = request.args.get("limit")
    if isinstance(limit_raw, str) and limit_raw.strip().lower() in ("all", "*"):
        limit = 100000
    else:
        limit = _parse_int(limit_raw, default=10000, min_value=1, max_value=100000)
        # 프론트가 과거 최대치(1000)로 요청하는 경우가 있어, 1000 이상이면 "전체 요청"으로 취급
        if limit is not None and limit >= 1000:
            limit = 100000
    offset = _parse_int(request.args.get("offset"), default=0, min_value=0)

    lot_name = (request.args.get("lot_name") or request.args.get("lotName") or "").strip()
    wafer_idx = _parse_int(request.args.get("wafer_idx") or request.args.get("waferIdx"), default=None, min_value=0)
    failure_type = (request.args.get("failure_type") or request.args.get("failureType") or "").strip()

    where = []
    params = []

    if lot_name:
        # chip_uid starts with lot_name (e.g. "{lot}X{x}Y{y}D{status}")
        where.append("chip_uid LIKE %s")
        params.append(f"{lot_name}%")
    if wafer_idx is not None:
        where.append("wafer_idx = %s")
        params.append(wafer_idx)
    if failure_type:
        where.append("failure_type = %s")
        params.append(failure_type)

    where_sql = f"WHERE {' AND '.join(where)}" if where else ""

    try:
        total, err = _count_chips(where_sql=where_sql, params=params)
        if err:
            return err
        rows, err = _fetch_chips(where_sql=where_sql, params=params, limit=limit, offset=offset)
        if err:
            return err
        return jsonify({
            "chips": rows,
            "count": len(rows),
            "total": total,
            "limit": limit,
            "offset": offset
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@inventory_bp.route("/chips/<chip_uid>", methods=["GET"])
def get_chip(chip_uid):
    """
    칩 상세 정보 조회 API
    GET /inventory/chips/<chip_uid>
    
    예: GET /inventory/chips/ABC123X5Y10D1
    """
    chip_uid = (chip_uid or "").strip()
    if not chip_uid:
        return jsonify({"error": "chip_uid(또는 chipId)가 필요합니다."}), 400

    try:
        row, err = _fetch_one_chip("WHERE chip_uid = %s", [chip_uid])
        if err:
            return err
        if not row:
            return jsonify({"error": "Not found"}), 404
        return jsonify(row), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


