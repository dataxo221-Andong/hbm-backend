import aiohttp
import asyncio
from datetime import datetime
import json
import requests
from typing import Optional, Dict, Any, List
import os
import sys
from builtins import print as _builtin_print
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager
import time

load_dotenv()

def _print_safe(*args, **kwargs):
    """
    Windows 콘솔(cp949 등)에서 이모지 출력 시 UnicodeEncodeError로 프로세스가 죽는 문제 방지.
    인코딩 불가 문자는 대체 문자로 치환해서 출력한다.
    """
    try:
        _builtin_print(*args, **kwargs)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "utf-8"
        sep = kwargs.get("sep", " ")
        end = kwargs.get("end", "\n")
        text = sep.join(str(a) for a in args) + end
        safe = text.encode(enc, errors="replace").decode(enc, errors="replace")
        _builtin_print(safe, end="")

# 이 모듈 내부의 print()를 안전 버전으로 오버라이드
print = _print_safe  # type: ignore

class HBMDataCrawler:
    """HBM Dashboard 각 페이지의 데이터를 크롤링하는 클래스"""
    
    def __init__(self, base_url: Optional[str] = None, frontend_url: Optional[str] = None):
        """
        초기화
        base_url: 백엔드 API URL
        frontend_url: 프론트엔드 웹 페이지 URL
        """
        if base_url is None:
            # 환경변수에서 백엔드 URL 가져오기
            self.base_url = os.getenv("HBM_BACKEND_URL", "http://localhost:5000")
        else:
            self.base_url = base_url
        
        if frontend_url is None:
            # 환경변수에서 프론트엔드 URL 가져오기
            self.frontend_url = os.getenv("HBM_FRONTEND_URL", "http://localhost:3000")
        else:
            self.frontend_url = frontend_url
        
        print(f"🌐 HBMDataCrawler 초기화 - 백엔드: {self.base_url}, 프론트엔드: {self.frontend_url}")
    
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

            # 모델 선택: DRAM Die(HBM3) 하나로 고정 (프론트에서 이 필드를 사용하면 드롭다운이 1개만 표시됨)
            model_options = [
                {
                    "value": "DRAM Die (HBM3)",
                    "label": "DRAM Die (HBM3) (DRAM-HBM3-8GB)",
                }
            ]
            result["modelOptions"] = model_options
            result["selectedModel"] = model_options[0]["value"]
            if isinstance(inventory_data, dict):
                inventory_data.setdefault("modelOptions", model_options)
                inventory_data.setdefault("selectedModel", model_options[0]["value"])
            
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
    
    def _get_selenium_driver(self):
        """Selenium WebDriver 생성 (재사용 가능)"""
        chrome_options = Options()
        chrome_options.add_argument('--headless')  # 백그라운드 실행
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--window-size=1920,1080')
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        try:
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=chrome_options)
            return driver
        except Exception as e:
            print(f"⚠️ ChromeDriver 초기화 실패: {e}")
            print("💡 Chrome 브라우저가 설치되어 있는지 확인하세요.")
            return None
    
    def _crawl_wafer_page_html(self, page_url: str) -> Optional[Dict[str, Any]]:
        """웨이퍼 페이지 HTML 크롤링하여 데이터 추출"""
        driver = None
        try:
            print(f"🌐 웨이퍼 페이지 HTML 크롤링 시작: {page_url}")
            
            driver = self._get_selenium_driver()
            if driver is None:
                print("⚠️ Selenium 드라이버를 생성할 수 없습니다.")
                return None
            
            # Performance Log 활성화 (네트워크 요청 가로채기)
            driver.execute_cdp_cmd('Performance.enable', {})
            driver.execute_cdp_cmd('Network.enable', {})
            
            driver.get(page_url)
            wait = WebDriverWait(driver, 30)
            print("⏳ 페이지 로딩 및 JavaScript 실행 대기 중...")
            time.sleep(10)  # 페이지 로딩 및 API 호출 완료 대기
            
            try:
                wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            except TimeoutException:
                print("⚠️ 페이지 로딩 타임아웃, 계속 진행...")
            
            # 방법 1: 네트워크 요청 가로채기 (가장 정확)
            wafers_data = None
            try:
                logs = driver.get_log('performance')
                for log in logs:
                    message = json.loads(log['message'])
                    method = message.get('message', {}).get('method', '')
                    
                    if method == 'Network.responseReceived':
                        response = message.get('message', {}).get('params', {}).get('response', {})
                        url = response.get('url', '')
                        
                        if '/wafer/list' in url:
                            request_id = message.get('message', {}).get('params', {}).get('requestId', '')
                            try:
                                response_body = driver.execute_cdp_cmd('Network.getResponseBody', {'requestId': request_id})
                                if response_body and 'body' in response_body:
                                    wafers_data = json.loads(response_body['body'])
                                    print(f"✅ 웨이퍼 API 응답에서 데이터 추출 성공: {len(wafers_data.get('wafers', []))}개")
                                    break
                            except Exception as e:
                                print(f"⚠️ 응답 본문 가져오기 실패: {e}")
            except Exception as e:
                print(f"⚠️ 네트워크 요청 가로채기 실패: {e}")
            
            # 방법 2: 통계 정보도 가로채기
            stats_data = None
            try:
                logs = driver.get_log('performance')
                for log in logs:
                    message = json.loads(log['message'])
                    method = message.get('message', {}).get('method', '')
                    
                    if method == 'Network.responseReceived':
                        response = message.get('message', {}).get('params', {}).get('response', {})
                        url = response.get('url', '')
                        
                        if '/wafer/total_status' in url:
                            request_id = message.get('message', {}).get('params', {}).get('requestId', '')
                            try:
                                response_body = driver.execute_cdp_cmd('Network.getResponseBody', {'requestId': request_id})
                                if response_body and 'body' in response_body:
                                    stats_data = json.loads(response_body['body'])
                                    print(f"✅ 통계 API 응답에서 데이터 추출 성공")
                                    break
                            except:
                                pass
            except:
                pass
            
            driver.quit()
            
            # 데이터 정규화
            if wafers_data:
                wafers = wafers_data.get("wafers", []) if isinstance(wafers_data, dict) else []
                
                # 데이터 변환
                formatted_wafers = []
                for w in wafers:
                    lot_name = w.get('lot_name', '')
                    die_count = w.get('die_count', 0)
                    defect_count = w.get('defect_count', 0)
                    good_die = die_count - defect_count
                    
                    yield_value = None
                    if die_count > 0:
                        yield_value = round((good_die / die_count) * 100, 1)
                    
                    formatted_wafers.append({
                        "id": lot_name,
                        "lot_name": lot_name,
                        "batch": "BATCH",
                        "status": "completed" if w.get('total_grade') else "pending",
                        "yield": yield_value,
                        "grade": w.get('total_grade'),
                        "processedAt": w.get('created_at'),
                        "confidence": float(w.get('confidence', 0)) if w.get('confidence') else None,
                        "failure_type": w.get('failure_type'),
                        "waferMapData": {
                            "good": good_die,
                            "bad": defect_count,
                            "total": die_count
                        },
                        "imageUrl": w.get('wafer_map')
                    })
                
                # 통계 계산
                completed_wafers = [w for w in formatted_wafers if w.get('status') == 'completed']
                total_good_die = sum(w.get('waferMapData', {}).get('good', 0) for w in completed_wafers)
                total_bad_die = sum(w.get('waferMapData', {}).get('bad', 0) for w in completed_wafers)
                total_die = total_good_die + total_bad_die
                defect_rate = round((total_bad_die / total_die) * 100, 2) if total_die > 0 else 0
                
                # API 통계 정보 사용
                if stats_data and isinstance(stats_data, dict):
                    total_wafers = stats_data.get('totalWafers', len(completed_wafers))
                    if stats_data.get('totalDie', 0) > 0:
                        total_good_die = stats_data.get('totalDie', 0) - stats_data.get('defectCount', 0)
                        total_bad_die = stats_data.get('defectCount', 0)
                        total_die = stats_data.get('totalDie', 0)
                        defect_rate = round((total_bad_die / total_die) * 100, 2) if total_die > 0 else 0
                else:
                    total_wafers = len(completed_wafers)
                
                return {
                    "timestamp": datetime.now().isoformat(),
                    "source": "wafermodeling",
                    "data": {
                        "wafers": formatted_wafers,
                        "statistics": {
                            "total_wafers": total_wafers,
                            "total_good_die": int(total_good_die),
                            "total_bad_die": int(total_bad_die),
                            "defect_rate": defect_rate
                        },
                        "summary": {
                            "total_wafers": len(formatted_wafers),
                            "completed_count": len(completed_wafers),
                            "processing_count": 0,
                            "pending_count": len(formatted_wafers) - len(completed_wafers)
                        }
                    },
                    "summary": {
                        "total_wafers": len(formatted_wafers),
                        "crawled_at": datetime.now().isoformat()
                    }
                }
            
            return None
            
        except TimeoutException:
            print(f"⏰ 페이지 로딩 타임아웃: {page_url}")
            if driver:
                driver.quit()
            return None
        except Exception as e:
            print(f"❌ HTML 크롤링 오류: {e}")
            import traceback
            traceback.print_exc()
            if driver:
                driver.quit()
            return None
    
    def _crawl_inventory_page_html(self, page_url: str) -> Optional[Dict[str, Any]]:
        """재고 페이지 HTML 크롤링"""
        driver = None
        try:
            print(f"🌐 재고 페이지 HTML 크롤링 시작: {page_url}")
            
            driver = self._get_selenium_driver()
            if driver is None:
                return None
            
            driver.execute_cdp_cmd('Performance.enable', {})
            driver.execute_cdp_cmd('Network.enable', {})
            
            driver.get(page_url)
            time.sleep(10)
            
            # 네트워크 요청 가로채기
            inventory_items = []
            try:
                logs = driver.get_log('performance')
                for log in logs:
                    message = json.loads(log['message'])
                    method = message.get('message', {}).get('method', '')
                    
                    if method == 'Network.responseReceived':
                        response = message.get('message', {}).get('params', {}).get('response', {})
                        url = response.get('url', '')
                        
                        if '/chip-inspection' in url or '/inventory' in url:
                            request_id = message.get('message', {}).get('params', {}).get('requestId', '')
                            try:
                                response_body = driver.execute_cdp_cmd('Network.getResponseBody', {'requestId': request_id})
                                if response_body and 'body' in response_body:
                                    data = json.loads(response_body['body'])
                                    if isinstance(data, list):
                                        inventory_items = data
                                    elif isinstance(data, dict) and 'items' in data:
                                        inventory_items = data['items']
                                    print(f"✅ 재고 API 응답에서 데이터 추출 성공")
                                    break
                            except:
                                pass
            except Exception as e:
                print(f"⚠️ 네트워크 요청 가로채기 실패: {e}")
            
            driver.quit()
            
            # 프론트엔드에 하드코딩된 재고 데이터 사용 (네트워크 요청 실패 시)
            if not inventory_items:
                print("⚠️ 네트워크 요청에서 데이터 추출 실패, 하드코딩된 재고 데이터 사용")
                inventory_items = [
                    {
                        "id": "1",
                        "name": "DRAM Die (HBM3)",
                        "category": "dram_die",
                        "sku": "DRAM-HBM3-8GB",
                        "currentStock": 15420,
                        "minStock": 10000,
                        "maxStock": 25000,
                        "optimalStock": 18000,
                        "status": "optimal"
                    },
                    {
                        "id": "2",
                        "name": "Logic Die (Base)",
                        "category": "logic_die",
                        "sku": "LOGIC-BASE-V2",
                        "currentStock": 8540,
                        "minStock": 8000,
                        "maxStock": 20000,
                        "optimalStock": 12000,
                        "status": "low"
                    },
                    {
                        "id": "3",
                        "name": "HBM3 8단 스택",
                        "category": "hbm_stack",
                        "sku": "HBM3-8HI-24GB",
                        "currentStock": 2340,
                        "minStock": 2000,
                        "maxStock": 5000,
                        "optimalStock": 3500,
                        "status": "low"
                    },
                    {
                        "id": "5",
                        "name": "완제품 HBM3",
                        "category": "finished",
                        "sku": "HBM3-PKG-FINAL",
                        "currentStock": 890,
                        "minStock": 500,
                        "maxStock": 2000,
                        "optimalStock": 1200,
                        "status": "optimal"
                    },
                    {
                        "id": "6",
                        "name": "Base Die Substrate",
                        "category": "raw_die",
                        "sku": "SUB-BASE-300MM",
                        "currentStock": 22500,
                        "minStock": 15000,
                        "maxStock": 25000,
                        "optimalStock": 20000,
                        "status": "excess"
                    }
                ]
            
            return {
                "items": inventory_items,
                "summary": {
                    "total_items": len(inventory_items),
                    "crawled_from": "frontend_html"
                }
            }
            
        except Exception as e:
            print(f"❌ 재고 페이지 크롤링 오류: {e}")
            if driver:
                driver.quit()
            return None
    
    def _crawl_logs_page_html(self, page_url: str) -> Optional[Dict[str, Any]]:
        """로그 페이지 HTML 크롤링"""
        driver = None
        try:
            print(f"🌐 로그 페이지 HTML 크롤링 시작: {page_url}")
            
            driver = self._get_selenium_driver()
            if driver is None:
                return None
            
            driver.execute_cdp_cmd('Performance.enable', {})
            driver.execute_cdp_cmd('Network.enable', {})
            
            driver.get(page_url)
            time.sleep(10)
            
            # 네트워크 요청 가로채기
            logs_data = []
            try:
                logs = driver.get_log('performance')
                for log in logs:
                    message = json.loads(log['message'])
                    method = message.get('message', {}).get('method', '')
                    
                    if method == 'Network.responseReceived':
                        response = message.get('message', {}).get('params', {}).get('response', {})
                        url = response.get('url', '')
                        
                        if '/logs' in url or '/yield' in url:
                            request_id = message.get('message', {}).get('params', {}).get('requestId', '')
                            try:
                                response_body = driver.execute_cdp_cmd('Network.getResponseBody', {'requestId': request_id})
                                if response_body and 'body' in response_body:
                                    data = json.loads(response_body['body'])
                                    if isinstance(data, list):
                                        logs_data = data
                                    elif isinstance(data, dict) and 'logs' in data:
                                        logs_data = data['logs']
                                    print(f"✅ 로그 API 응답에서 데이터 추출 성공")
                                    break
                            except:
                                pass
            except Exception as e:
                print(f"⚠️ 네트워크 요청 가로채기 실패: {e}")
            
            driver.quit()
            
            return {
                "logs": logs_data,
                "summary": {
                    "total_logs": len(logs_data),
                    "crawled_from": "frontend_html"
                }
            }
            
        except Exception as e:
            print(f"❌ 로그 페이지 크롤링 오류: {e}")
            if driver:
                driver.quit()
            return None
    
    def _crawl_stacking_page_html(self, page_url: str) -> Optional[Dict[str, Any]]:
        """적층 페이지 HTML 크롤링"""
        driver = None
        try:
            print(f"🌐 적층 페이지 HTML 크롤링 시작: {page_url}")
            
            driver = self._get_selenium_driver()
            if driver is None:
                return None
            
            driver.execute_cdp_cmd('Performance.enable', {})
            driver.execute_cdp_cmd('Network.enable', {})
            
            driver.get(page_url)
            time.sleep(10)
            
            # 네트워크 요청 가로채기
            stacks_data = []
            try:
                logs = driver.get_log('performance')
                for log in logs:
                    message = json.loads(log['message'])
                    method = message.get('message', {}).get('method', '')
                    
                    if method == 'Network.responseReceived':
                        response = message.get('message', {}).get('params', {}).get('response', {})
                        url = response.get('url', '')
                        
                        if '/stack' in url or '/stacking' in url:
                            request_id = message.get('message', {}).get('params', {}).get('requestId', '')
                            try:
                                response_body = driver.execute_cdp_cmd('Network.getResponseBody', {'requestId': request_id})
                                if response_body and 'body' in response_body:
                                    data = json.loads(response_body['body'])
                                    if isinstance(data, list):
                                        stacks_data = data
                                    elif isinstance(data, dict) and 'stacks' in data:
                                        stacks_data = data['stacks']
                                    print(f"✅ 적층 API 응답에서 데이터 추출 성공")
                                    break
                            except:
                                pass
            except Exception as e:
                print(f"⚠️ 네트워크 요청 가로채기 실패: {e}")
            
            driver.quit()
            
            return {
                "stacks": stacks_data,
                "summary": {
                    "total_stacks": len(stacks_data),
                    "crawled_from": "frontend_html"
                }
            }
            
        except Exception as e:
            print(f"❌ 적층 페이지 크롤링 오류: {e}")
            if driver:
                driver.quit()
            return None
    
    async def crawl_wafermodeling_data(self) -> Dict[str, Any]:
        """웨이퍼 모델링 데이터 크롤링 (하이브리드 방식: API 우선, 실패 시 HTML 크롤링)"""
        try:
            print("=" * 60)
            print("🔬 [하이브리드 크롤링] Wafer Modeling 데이터 수집 시작")
            print("=" * 60)
            
            # 1단계: API 방식 시도 (빠르고 효율적)
            print("\n[1단계] API 방식 시도 중...")
            print("  → 백엔드 REST API 직접 호출")
            try:
                api_result = await self._crawl_wafermodeling_data_via_api()
                
                # API 방식 성공 확인 (데이터가 있고 에러가 없는 경우)
                if api_result and api_result.get('data') and not api_result.get('error'):
                    wafer_data = api_result.get('data', {})
                    wafers = wafer_data.get('wafers', [])
                    
                    if wafers and len(wafers) > 0:
                        print(f"✅ [성공] API 방식으로 {len(wafers)}개 웨이퍼 수집 완료")
                        print(f"   → 사용된 방식: REST API 직접 호출")
                        print(f"   → 수집 시간: 빠름 (API 직접 호출)")
                        api_result['crawl_method'] = 'API'
                        api_result['crawl_method_description'] = '백엔드 REST API를 직접 호출하여 데이터를 수집했습니다.'
                        return api_result
                    else:
                        print("⚠️ [실패] API 방식: 데이터는 받았지만 웨이퍼가 없음")
                else:
                    print("⚠️ [실패] API 방식: 데이터 수집 실패 또는 에러 발생")
            except Exception as api_error:
                print(f"⚠️ [실패] API 방식 오류: {api_error}")
            
            # 2단계: API 실패 시 HTML 크롤링 방식으로 전환
            print("\n[2단계] HTML 크롤링 방식으로 전환...")
            print("  → 프론트엔드 웹 페이지 크롤링 (Selenium)")
            try:
                # 동기 함수를 비동기로 실행 (Python 버전 호환성)
                loop = asyncio.get_event_loop()
                html_result = await loop.run_in_executor(
                    None,
                    self._crawl_wafer_page_html,
                    f"{self.frontend_url}/wafer"
                )
                
                if html_result and html_result.get('data'):
                    wafer_data = html_result.get('data', {})
                    wafers = wafer_data.get('wafers', [])
                    
                    if wafers and len(wafers) > 0:
                        print(f"✅ [성공] HTML 크롤링 방식으로 {len(wafers)}개 웨이퍼 수집 완료")
                        print(f"   → 사용된 방식: Selenium 기반 웹 페이지 크롤링")
                        print(f"   → 수집 시간: 느림 (브라우저 실행 필요)")
                        html_result['crawl_method'] = 'HTML'
                        html_result['crawl_method_description'] = '프론트엔드 웹 페이지를 Selenium으로 크롤링하여 데이터를 수집했습니다.'
                        return html_result
                    else:
                        print("⚠️ [실패] HTML 크롤링 방식: 데이터는 받았지만 웨이퍼가 없음")
                else:
                    print("⚠️ [실패] HTML 크롤링 방식: 데이터 수집 실패")
            except Exception as html_error:
                print(f"⚠️ [실패] HTML 크롤링 방식 오류: {html_error}")
                import traceback
                traceback.print_exc()
            
            # 둘 다 실패한 경우
            print("\n" + "=" * 60)
            print("❌ [실패] 모든 크롤링 방식 실패")
            print("   → API 방식: 실패")
            print("   → HTML 크롤링 방식: 실패")
            print("=" * 60)
            return {
                "timestamp": datetime.now().isoformat(),
                "source": "wafermodeling",
                "crawl_method": "FAILED",
                "crawl_method_description": "모든 크롤링 방식이 실패했습니다.",
                "error": "모든 크롤링 방식 실패 (API 및 HTML 크롤링 모두 실패)",
                "data": None
            }
            
        except Exception as e:
            print(f"\n❌ [오류] Wafer Modeling 데이터 수집 중 예외 발생: {e}")
            import traceback
            traceback.print_exc()
            return {
                "timestamp": datetime.now().isoformat(),
                "source": "wafermodeling",
                "error": str(e),
                "data": None
            }
    
    async def _crawl_wafermodeling_data_via_api(self) -> Dict[str, Any]:
        """웨이퍼 모델링 데이터 크롤링 (API 호출 방식 - 내부 함수)"""
        try:
            print("📡 API 호출 방식으로 데이터 수집...")
            
            # 1. 웨이퍼 목록 조회 (모든 페이지 수집)
            all_wafers = []
            page = 1
            limit = 100  # 한 번에 많이 가져오기
            total_wafers_count = 0
            
            while True:
                wafer_list_response = await self.fetch_api_data(f"/wafer/list?page={page}&limit={limit}")
                
                if wafer_list_response is None or not isinstance(wafer_list_response, dict):
                    print(f"⚠️ 페이지 {page} 데이터 없음")
                    break
                
                wafers = wafer_list_response.get("wafers", [])
                total_wafers_count = wafer_list_response.get("total", 0)
                
                if not wafers:
                    break
                
                all_wafers.extend(wafers)
                print(f"📄 페이지 {page}: {len(wafers)}개 웨이퍼 수집 (전체: {len(all_wafers)}/{total_wafers_count})")
                
                # 다음 페이지가 없으면 중단
                if len(all_wafers) >= total_wafers_count or len(wafers) < limit:
                    break
                
                page += 1
            
            # 2. 통계 정보 조회
            stats_response = await self.fetch_api_data("/wafer/total_status")
            
            # 3. 데이터 변환 (프론트엔드 형식에 맞게)
            formatted_wafers = []
            for w in all_wafers:
                lot_name = w.get('lot_name', '')
                die_count = w.get('die_count', 0)
                defect_count = w.get('defect_count', 0)
                good_die = die_count - defect_count
                
                # 수율 계산
                yield_value = None
                if die_count > 0:
                    yield_value = round((good_die / die_count) * 100, 1)
                
                formatted_wafers.append({
                    "id": lot_name,
                    "lot_name": lot_name,
                    "batch": "BATCH",
                    "status": "completed" if w.get('total_grade') else "pending",
                    "yield": yield_value,
                    "grade": w.get('total_grade'),
                    "processedAt": w.get('created_at'),
                    "confidence": float(w.get('confidence', 0)) if w.get('confidence') else None,
                    "failure_type": w.get('failure_type'),
                    "waferMapData": {
                        "good": good_die,
                        "bad": defect_count,
                        "total": die_count
                    },
                    "imageUrl": w.get('wafer_map')  # Firebase URL
                })
            
            # 4. 통계 계산
            completed_wafers = [w for w in formatted_wafers if w.get('status') == 'completed']
            total_good_die = sum(w.get('waferMapData', {}).get('good', 0) for w in completed_wafers)
            total_bad_die = sum(w.get('waferMapData', {}).get('bad', 0) for w in completed_wafers)
            total_die = total_good_die + total_bad_die
            defect_rate = round((total_bad_die / total_die) * 100, 2) if total_die > 0 else 0
            
            # 5. 통계 정보 (API에서 가져온 값 우선 사용)
            if stats_response and isinstance(stats_response, dict):
                total_wafers = stats_response.get('totalWafers', len(completed_wafers))
                # API 통계와 계산된 통계 중 더 정확한 값 사용
                if stats_response.get('totalDie', 0) > 0:
                    total_good_die = stats_response.get('totalDie', 0) - stats_response.get('defectCount', 0)
                    total_bad_die = stats_response.get('defectCount', 0)
                    total_die = stats_response.get('totalDie', 0)
                    defect_rate = round((total_bad_die / total_die) * 100, 2) if total_die > 0 else 0
            else:
                total_wafers = len(completed_wafers)
            
            # 6. 최종 데이터 구조 생성
            wafer_data = {
                "wafers": formatted_wafers,
                "statistics": {
                    "total_wafers": total_wafers,
                    "total_good_die": int(total_good_die),
                    "total_bad_die": int(total_bad_die),
                    "defect_rate": defect_rate
                },
                "summary": {
                    "total_wafers": len(formatted_wafers),
                    "completed_count": len(completed_wafers),
                    "processing_count": 0,
                    "pending_count": len(formatted_wafers) - len(completed_wafers)
                }
            }
            
            # 디버깅: 받은 데이터 확인
            print(f"🔍 [DEBUG] API 응답으로 받은 웨이퍼 데이터:")
            print(f"  - 총 웨이퍼 수: {len(formatted_wafers)}개")
            print(f"  - 완료된 웨이퍼: {len(completed_wafers)}개")
            print(f"  - 총 Good Die: {total_good_die}개")
            print(f"  - 총 Bad Die: {total_bad_die}개")
            print(f"  - 불량률: {defect_rate}%")
            if len(formatted_wafers) > 0:
                print(f"  - 첫 번째 웨이퍼 ID: {formatted_wafers[0].get('id', 'N/A')}")
            
            # 7. 크롤러 형식에 맞게 변환
            result = {
                "timestamp": datetime.now().isoformat(),
                "source": "wafermodeling",
                "data": wafer_data,
                "summary": {
                    "total_wafers": len(formatted_wafers),
                    "crawled_at": datetime.now().isoformat()
                }
            }
            
            print(f"✅ [API 호출 성공] {len(formatted_wafers)}개 웨이퍼 데이터 수집 완료")
            return result
            
        except Exception as e:
            print(f"❌ [API 호출 오류] Wafer Modeling 데이터 수집 중 오류 발생: {e}")
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


