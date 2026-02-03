import aiohttp
import asyncio
from datetime import datetime
import json
import requests
from typing import Optional, Dict, Any, List
import os
from dotenv import load_dotenv

load_dotenv()

class HBMDataCrawler:
    """HBM Dashboard 각 페이지의 데이터를 크롤링하는 클래스"""
    
    def __init__(self, base_url: Optional[str] = None):
        """
        초기화
        base_url이 None이면 환경변수에서 가져오거나 기본값 사용
        """
        if base_url is None:
            # 환경변수에서 백엔드 URL 가져오기
            self.base_url = os.getenv("HBM_BACKEND_URL", "http://localhost:5000")
        else:
            self.base_url = base_url
        
        print(f"🌐 HBMDataCrawler 초기화 - 서버: {self.base_url}")
    
    def test_endpoint_sync(self, endpoint: str) -> Dict[str, Any]:
        """동기적으로 엔드포인트 테스트"""
        try:
            url = f"{self.base_url}{endpoint}"
            print(f"🧪 엔드포인트 테스트: {url}")
            
            response = requests.get(url, timeout=30)
            
            result = {
                "endpoint": endpoint,
                "url": url,
                "status_code": response.status_code,
                "success": response.status_code == 200,
                "response_size": len(response.content),
                "content_type": response.headers.get('content-type', 'unknown')
            }
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    result["data_type"] = type(data).__name__
                    
                    if isinstance(data, list):
                        result["data_length"] = len(data)
                        print(f"✅ 테스트 성공: {endpoint} - {len(data)}개 항목")
                        if len(data) > 0:
                            result["sample_item"] = data[0]
                            result["all_keys"] = list(data[0].keys()) if isinstance(data[0], dict) else []
                    elif isinstance(data, dict):
                        result["data_length"] = len(data) if isinstance(data, dict) else 0
                        result["dict_keys"] = list(data.keys())
                        print(f"✅ 테스트 성공: {endpoint} - dict 객체")
                        
                except Exception as parse_error:
                    result["data_type"] = "non-json"
                    result["parse_error"] = str(parse_error)
                    print(f"⚠️ JSON 파싱 실패: {endpoint}")
            else:
                result["error"] = response.text[:200]
                print(f"❌ 테스트 실패: {endpoint} - HTTP {response.status_code}")
                
            return result
            
        except requests.exceptions.RequestException as e:
            print(f"❌ 네트워크 오류: {endpoint} - {e}")
            return {
                "endpoint": endpoint,
                "success": False,
                "error": f"Network error: {str(e)}"
            }
        except Exception as e:
            print(f"❌ 예외 발생: {endpoint} - {e}")
            return {
                "endpoint": endpoint,
                "success": False,
                "error": f"Exception: {str(e)}"
            }
    
    async def fetch_api_data(self, endpoint: str, method: str = "GET", data: Optional[Dict] = None) -> Optional[Any]:
        """API 데이터 가져오기 (비동기)"""
        try:
            url = f"{self.base_url}{endpoint}"
            print(f"🔍 API 호출 시도: {url}")
            
            timeout = aiohttp.ClientTimeout(total=30)
            connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
            
            async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
                if method == "GET":
                    async with session.get(url, headers={
                        'Accept': 'application/json',
                        'User-Agent': 'HBM-DataCrawler/1.0'
                    }) as response:
                        return await self._process_response(response, endpoint)
                elif method == "POST":
                    async with session.post(url, json=data, headers={
                        'Accept': 'application/json',
                        'Content-Type': 'application/json',
                        'User-Agent': 'HBM-DataCrawler/1.0'
                    }) as response:
                        return await self._process_response(response, endpoint)
                        
        except asyncio.TimeoutError:
            print(f"⏰ API 타임아웃: {endpoint}")
            return None
        except aiohttp.ClientError as e:
            print(f"❌ 클라이언트 오류: {endpoint} - {e}")
            return None
        except Exception as e:
            print(f"❌ API 데이터 가져오기 예외: {endpoint} - {e}")
            import traceback
            traceback.print_exc()
            return None
    
    async def _process_response(self, response: aiohttp.ClientResponse, endpoint: str) -> Optional[Any]:
        """응답 처리"""
        print(f"📡 응답 상태: {response.status} - {endpoint}")
        
        if response.status == 200:
            try:
                data = await response.json()
                
                if isinstance(data, list):
                    print(f"✅ API 데이터 수신 성공: {endpoint} - {len(data)}개 항목")
                    return data
                elif isinstance(data, dict):
                    print(f"✅ API 데이터 수신 성공: {endpoint} - dict 객체")
                    # dict 내부에 배열이 있는지 확인
                    if 'data' in data and isinstance(data['data'], list):
                        return data['data']
                    elif 'items' in data and isinstance(data['items'], list):
                        return data['items']
                    else:
                        return data
                else:
                    print(f"✅ API 데이터 수신: {endpoint} - {type(data)}")
                    return data
                    
            except aiohttp.ContentTypeError as e:
                print(f"❌ JSON 파싱 오류: {endpoint} - {e}")
                text_content = await response.text()
                print(f"📄 응답 내용 (처음 500자): {text_content[:500]}")
                return None
        else:
            error_text = await response.text()
            print(f"❌ API 응답 오류: {endpoint} - HTTP {response.status}")
            print(f"❌ 오류 내용: {error_text[:200]}...")
            return None
    
    async def crawl_inventory_data(self) -> Dict[str, Any]:
        """재고 관리 데이터 크롤링 (inventory 페이지)"""
        try:
            print("📦 Inventory 데이터 크롤링 시작...")
            
            # 재고 관련 API 엔드포인트 호출
            # 실제 API가 없으면 데모 데이터 구조 반환
            inventory_data = await self.fetch_api_data("/api/inventory")
            
            if inventory_data is None:
                # 데모 데이터 구조 반환
                print("⚠️ API 데이터 없음, 데모 데이터 구조 반환")
                inventory_data = {
                    "items": [],
                    "summary": {
                        "total_items": 0,
                        "low_stock_count": 0,
                        "critical_stock_count": 0
                    }
                }
            
            # 데이터 정규화
            result = {
                "timestamp": datetime.now().isoformat(),
                "source": "inventory",
                "data": inventory_data,
                "summary": {
                    "total_items": len(inventory_data.get("items", [])) if isinstance(inventory_data, dict) else len(inventory_data) if isinstance(inventory_data, list) else 0,
                    "crawled_at": datetime.now().isoformat()
                }
            }
            
            print(f"✅ Inventory 데이터 크롤링 완료: {result['summary']['total_items']}개 항목")
            return result
            
        except Exception as e:
            print(f"❌ Inventory 데이터 크롤링 오류: {e}")
            import traceback
            traceback.print_exc()
            return {
                "timestamp": datetime.now().isoformat(),
                "source": "inventory",
                "error": str(e),
                "data": None
            }
    
    async def crawl_logs_data(self) -> Dict[str, Any]:
        """로그 데이터 크롤링 (logs 페이지)"""
        try:
            print("📋 Logs 데이터 크롤링 시작...")
            
            # 로그 관련 API 엔드포인트 호출
            logs_data = await self.fetch_api_data("/api/logs")
            
            if logs_data is None:
                # 데모 데이터 구조 반환
                print("⚠️ API 데이터 없음, 데모 데이터 구조 반환")
                logs_data = {
                    "logs": [],
                    "summary": {
                        "total_logs": 0,
                        "completed_count": 0,
                        "failed_count": 0,
                        "processing_count": 0
                    }
                }
            
            # 데이터 정규화
            result = {
                "timestamp": datetime.now().isoformat(),
                "source": "logs",
                "data": logs_data,
                "summary": {
                    "total_logs": len(logs_data.get("logs", [])) if isinstance(logs_data, dict) else len(logs_data) if isinstance(logs_data, list) else 0,
                    "crawled_at": datetime.now().isoformat()
                }
            }
            
            print(f"✅ Logs 데이터 크롤링 완료: {result['summary']['total_logs']}개 항목")
            return result
            
        except Exception as e:
            print(f"❌ Logs 데이터 크롤링 오류: {e}")
            import traceback
            traceback.print_exc()
            return {
                "timestamp": datetime.now().isoformat(),
                "source": "logs",
                "error": str(e),
                "data": None
            }
    
    async def crawl_stacking_data(self) -> Dict[str, Any]:
        """적층 구조 데이터 크롤링 (stacking 페이지)"""
        try:
            print("🔬 Stacking 데이터 크롤링 시작...")
            
            # 적층 관련 API 엔드포인트 호출
            stacking_data = await self.fetch_api_data("/stack/analyze")
            
            if stacking_data is None:
                # 데모 데이터 구조 반환
                print("⚠️ API 데이터 없음, 데모 데이터 구조 반환")
                stacking_data = {
                    "stacks": [],
                    "summary": {
                        "total_stacks": 0,
                        "good_stacks": 0,
                        "defect_stacks": 0
                    }
                }
            
            # 데이터 정규화
            result = {
                "timestamp": datetime.now().isoformat(),
                "source": "stacking",
                "data": stacking_data,
                "summary": {
                    "total_stacks": len(stacking_data.get("stacks", [])) if isinstance(stacking_data, dict) else len(stacking_data) if isinstance(stacking_data, list) else 0,
                    "crawled_at": datetime.now().isoformat()
                }
            }
            
            print(f"✅ Stacking 데이터 크롤링 완료: {result['summary']['total_stacks']}개 항목")
            return result
            
        except Exception as e:
            print(f"❌ Stacking 데이터 크롤링 오류: {e}")
            import traceback
            traceback.print_exc()
            return {
                "timestamp": datetime.now().isoformat(),
                "source": "stacking",
                "error": str(e),
                "data": None
            }
    
    async def crawl_wafermodeling_data(self) -> Dict[str, Any]:
        """웨이퍼 모델링 데이터 크롤링 (wafermodeling 페이지)"""
        try:
            print("🔬 Wafer Modeling 데이터 크롤링 시작...")
            
            # HTTP 요청으로 웨이퍼 관련 API 엔드포인트 호출
            # 참고: crawler_routes.py에서 직접 함수 호출로 대체됨
            wafer_data = await self.fetch_api_data("/wafer")
            
            if wafer_data is None:
                # 데모 데이터 구조 반환
                print("⚠️ API 데이터 없음, 데모 데이터 구조 반환")
                wafer_data = {
                    "wafers": [],
                    "statistics": {
                        "total_wafers": 0,
                        "total_good_die": 0,
                        "total_bad_die": 0,
                        "defect_rate": 0
                    },
                    "summary": {
                        "total_wafers": 0,
                        "completed_count": 0,
                        "processing_count": 0,
                        "pending_count": 0
                    }
                }
            
            # 디버깅: 받은 데이터 확인
            print(f"🔍 [DEBUG] 크롤러가 받은 웨이퍼 데이터:")
            print(f"  - 타입: {type(wafer_data)}")
            if isinstance(wafer_data, dict):
                print(f"  - 키: {list(wafer_data.keys())}")
                if "wafers" in wafer_data:
                    print(f"  - 웨이퍼 수: {len(wafer_data.get('wafers', []))}")
                    if len(wafer_data.get('wafers', [])) > 0:
                        first_wafer = wafer_data.get('wafers', [])[0]
                        print(f"  - 첫 번째 웨이퍼 ID: {first_wafer.get('id', 'N/A')}")
                        print(f"  - 첫 번째 웨이퍼 lot_name: {first_wafer.get('lot_name', 'N/A')}")
            
            # 데이터 정규화
            result = {
                "timestamp": datetime.now().isoformat(),
                "source": "wafermodeling",
                "data": wafer_data,
                "summary": {
                    "total_wafers": len(wafer_data.get("wafers", [])) if isinstance(wafer_data, dict) else len(wafer_data) if isinstance(wafer_data, list) else 0,
                    "crawled_at": datetime.now().isoformat()
                }
            }
            
            print(f"✅ Wafer Modeling 데이터 크롤링 완료: {result['summary']['total_wafers']}개 항목")
            return result
            
        except Exception as e:
            print(f"❌ Wafer Modeling 데이터 크롤링 오류: {e}")
            import traceback
            traceback.print_exc()
            return {
                "timestamp": datetime.now().isoformat(),
                "source": "wafermodeling",
                "error": str(e),
                "data": None
            }
    
    async def crawl_all_dashboard_data(self) -> Dict[str, Any]:
        """모든 dashboard 데이터 크롤링"""
        try:
            print("🚀 모든 Dashboard 데이터 크롤링 시작...")
            
            # 모든 데이터를 병렬로 크롤링
            results = await asyncio.gather(
                self.crawl_inventory_data(),
                self.crawl_logs_data(),
                self.crawl_stacking_data(),
                self.crawl_wafermodeling_data(),
                return_exceptions=True
            )
            
            # 결과 정리
            all_data = {
                "timestamp": datetime.now().isoformat(),
                "inventory": results[0] if not isinstance(results[0], Exception) else {"error": str(results[0])},
                "logs": results[1] if not isinstance(results[1], Exception) else {"error": str(results[1])},
                "stacking": results[2] if not isinstance(results[2], Exception) else {"error": str(results[2])},
                "wafermodeling": results[3] if not isinstance(results[3], Exception) else {"error": str(results[3])},
                "summary": {
                    "total_sources": 4,
                    "successful": sum(1 for r in results if not isinstance(r, Exception)),
                    "failed": sum(1 for r in results if isinstance(r, Exception)),
                    "crawled_at": datetime.now().isoformat()
                }
            }
            
            print(f"✅ 모든 Dashboard 데이터 크롤링 완료: {all_data['summary']['successful']}/{all_data['summary']['total_sources']} 성공")
            return all_data
            
        except Exception as e:
            print(f"❌ 전체 데이터 크롤링 오류: {e}")
            import traceback
            traceback.print_exc()
            return {
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "data": None
            }
    
    def crawl_all_dashboard_data_sync(self) -> Dict[str, Any]:
        """동기 방식으로 모든 dashboard 데이터 크롤링"""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.crawl_all_dashboard_data())


# 전역 인스턴스
crawler = HBMDataCrawler()


