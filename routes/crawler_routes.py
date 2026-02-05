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
    """웨이퍼 모델링 데이터 크롤링 (웹 API 호출 방식)"""
    try:
        # 크롤러 서비스를 통해 웹 API 호출
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(crawler.crawl_wafermodeling_data())
        loop.close()
        
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


