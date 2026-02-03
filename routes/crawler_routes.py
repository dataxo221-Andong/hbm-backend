from flask import Blueprint, request, jsonify
from crawler_service import HBMDataCrawler, crawler
from datetime import datetime
import asyncio

crawler_bp = Blueprint("crawler", __name__, url_prefix="/api/crawler")

@crawler_bp.route("/inventory", methods=["GET"])
def get_inventory_data():
    """재고 관리 데이터 크롤링"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(crawler.crawl_inventory_data())
        loop.close()
        
        return jsonify(result), 200
    except Exception as e:
        return jsonify({
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@crawler_bp.route("/logs", methods=["GET"])
def get_logs_data():
    """로그 데이터 크롤링"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(crawler.crawl_logs_data())
        loop.close()
        
        return jsonify(result), 200
    except Exception as e:
        return jsonify({
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@crawler_bp.route("/stacking", methods=["GET"])
def get_stacking_data():
    """적층 구조 데이터 크롤링"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(crawler.crawl_stacking_data())
        loop.close()
        
        return jsonify(result), 200
    except Exception as e:
        return jsonify({
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@crawler_bp.route("/wafermodeling", methods=["GET"])
def get_wafermodeling_data():
    """웨이퍼 모델링 데이터 크롤링"""
    try:
        # HTTP 요청 대신 직접 함수 호출 (더 안정적)
        from routes.wafer import get_all_wafers
        
        # get_all_wafers()는 Flask Response를 반환하므로 JSON 추출
        response = get_all_wafers()
        wafer_data = response.get_json() if hasattr(response, 'get_json') else response
        
        # 디버깅: 받은 데이터 확인
        print(f"🔍 [DEBUG] 크롤러가 받은 웨이퍼 데이터:")
        print(f"  - 타입: {type(wafer_data)}")
        if isinstance(wafer_data, dict):
            print(f"  - 키: {list(wafer_data.keys())}")
            if "wafers" in wafer_data:
                print(f"  - 웨이퍼 수: {len(wafer_data.get('wafers', []))}")
                if len(wafer_data.get('wafers', [])) > 0:
                    print(f"  - 첫 번째 웨이퍼 ID: {wafer_data.get('wafers', [])[0].get('id', 'N/A')}")
        
        # 크롤러 형식에 맞게 변환
        result = {
            "timestamp": datetime.now().isoformat(),
            "source": "wafermodeling",
            "data": wafer_data,
            "summary": {
                "total_wafers": len(wafer_data.get("wafers", [])) if isinstance(wafer_data, dict) else 0,
                "crawled_at": datetime.now().isoformat()
            }
        }
        
        print(f"✅ Wafer Modeling 데이터 크롤링 완료: {result['summary']['total_wafers']}개 항목")
        return jsonify(result), 200
    except Exception as e:
        print(f"❌ 웨이퍼 모델링 데이터 크롤링 오류: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@crawler_bp.route("/all", methods=["GET"])
def get_all_dashboard_data():
    """모든 dashboard 데이터 크롤링"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(crawler.crawl_all_dashboard_data())
        loop.close()
        
        return jsonify(result), 200
    except Exception as e:
        return jsonify({
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@crawler_bp.route("/test/<endpoint>", methods=["GET"])
def test_endpoint(endpoint: str):
    """엔드포인트 테스트"""
    try:
        result = crawler.test_endpoint_sync(f"/{endpoint}")
        return jsonify(result), 200
    except Exception as e:
        return jsonify({
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@crawler_bp.route("/health", methods=["GET"])
def health():
    """크롤러 헬스 체크"""
    return jsonify({
        "status": "healthy",
        "message": "Data crawler is ready",
        "base_url": crawler.base_url,
        "timestamp": datetime.now().isoformat()
    }), 200


