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
            
            timeout = aiohttp.ClientTimeout(total=60)
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
        """재고 데이터 크롤링 - 웨이퍼처럼 실제 데이터 API 호출 (칩 목록 전체)"""
        try:
            print("📦 Inventory 데이터 크롤링 시작...")
            
            # 데이터 조회용 API: 칩 목록 전체 (웨이퍼 /wafer/list 와 동일하게 실제 데이터)
            chips_data = await self.fetch_api_data("/inventory/chips?limit=10000")
            
            if chips_data is None:
                print("⚠️ API 데이터 없음, 빈 데이터 반환")
                inventory_data = {
                    "items": [],
                    "chips": [],
                    "summary": {
                        "total_items": 0,
                        "total_chips": 0,
                        "good_chips": 0,
                        "bad_chips": 0,
                        "low_stock_count": 0,
                        "critical_stock_count": 0,
                        "by_failure_type": {},
                    }
                }
            else:
                # chips 데이터를 inventory 형식으로 변환
                chips = []
                if isinstance(chips_data, dict):
                    chips = chips_data.get("chips", [])
                    total = chips_data.get("total", len(chips))
                elif isinstance(chips_data, list):
                    chips = chips_data
                    total = len(chips)
                else:
                    chips = []
                    total = 0
                
                print(f"🔍 [DEBUG] 칩 데이터 수신: {len(chips)}개 칩 (전체: {total}개)")
                
                # 칩 데이터를 재고 항목 형식으로 변환
                items = []
                for chip in chips:
                    chip_uid = chip.get("chip_uid", "")
                    die_status = chip.get("die_status", 0)
                    failure_type = chip.get("failure_type", "None")
                    
                    # 재고 상태 판단 (die_status: 1=정상, 0=불량)
                    status = "optimal" if die_status == 1 else "low"
                    
                    items.append({
                        "id": chip_uid,
                        "name": f"Chip {chip_uid[:20]}..." if len(chip_uid) > 20 else f"Chip {chip_uid}",
                        "category": "chip",
                        "chip_uid": chip_uid,
                        "wafer_idx": chip.get("wafer_idx"),
                        "failure_type": failure_type,
                        "die_status": die_status,
                        "coor_x": chip.get("coor_x"),
                        "coor_y": chip.get("coor_y"),
                        "status": status,
                        "currentStock": 1,  # 각 칩은 1개 단위
                        "minStock": 0,
                        "maxStock": 1
                    })
                
                # 통계 계산
                good_chips = [c for c in chips if c.get("die_status") == 1]
                bad_chips = [c for c in chips if c.get("die_status") != 1]
                # 불량유형별 칩 수 (챗봇 재고 밸런스/병목·생산 제안용)
                by_failure_type = {}
                for c in bad_chips:
                    ft = c.get("failure_type") or c.get("failureType") or "None"
                    key = ft if isinstance(ft, str) and ft.strip() else "None"
                    by_failure_type[key] = by_failure_type.get(key, 0) + 1

                inventory_data = {
                    "items": items,
                    "chips": chips,  # 원본 데이터도 포함
                    "summary": {
                        "total_items": total,
                        "total_chips": total,
                        "good_chips": len(good_chips),
                        "bad_chips": len(bad_chips),
                        "low_stock_count": len(bad_chips),
                        "critical_stock_count": 0,
                        "by_failure_type": by_failure_type,
                    }
                }
            
            # 데이터 정규화
            result = {
                "timestamp": datetime.now().isoformat(),
                "source": "inventory",
                "data": inventory_data,
                "summary": {
                    "total_items": inventory_data.get("summary", {}).get("total_chips", 0),
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
        """로그 데이터 크롤링 - 웨이퍼처럼 실제 데이터 API 호출 (/log/list 전체 목록 + 통계)"""
        try:
            print("📋 Logs 데이터 크롤링 시작...")
            
            # 데이터 조회용 API: 전체 목록 + 일별/추세 통계 (프론트와 동일 소스)
            logs_list = await self.fetch_api_data("/log/list")
            daily_stats = await self.fetch_api_data("/log/stats/daily")
            trend_data = await self.fetch_api_data("/log/stats/trend")
            
            if logs_list is None:
                logs_list = []
            if isinstance(logs_list, dict):
                logs_list = logs_list.get("logs", logs_list.get("data", []))
            if not isinstance(logs_list, list):
                logs_list = []
            
            logs_data = {
                "logs": logs_list,
                "daily": daily_stats if isinstance(daily_stats, dict) else {},
                "trend": trend_data if isinstance(trend_data, list) else (trend_data if isinstance(trend_data, dict) else []),
            }
            
            # 데이터 정규화
            result = {
                "timestamp": datetime.now().isoformat(),
                "source": "logs",
                "data": logs_data,
                "summary": {
                    "total_logs": len(logs_list),
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
        """적층 데이터 크롤링 - 웨이퍼처럼 실제 데이터 API 호출 (스택 목록 + 최신 결과 상세)"""
        try:
            print("🔬 Stacking 데이터 크롤링 시작...")
            
            # 데이터 조회용 API: 스택 이력 목록 + 최신 tsv 결과 전체
            history = await self.fetch_api_data("/stack/list")
            
            if not history or len(history) == 0:
                print("⚠️ 히스토리 데이터 없음")
                return {
                    "timestamp": datetime.now().isoformat(),
                    "source": "stacking",
                    "data": {
                        "history": [],
                        "latest_result": None
                    },
                    "summary": {
                        "total_history": 0,
                        "latest_tsv_num": None,
                        "total_stacks": 0,
                        "crawled_at": datetime.now().isoformat()
                    }
                }
            
            # 2. 최신 tsv_num 찾기
            latest_tsv_num = history[0]['tsv_num']
            print(f"📊 최신 tsv_num: {latest_tsv_num}")
            
            # 3. 최신 스택 결과 가져오기
            stack_result = await self.fetch_api_data(f"/stack/result/{latest_tsv_num}")
            
            # 스택 개수 및 품질 등급(A/B/C) 분포 계산
            total_stacks = 0
            grade_a = 0
            grade_b = 0
            grade_c = 0
            if stack_result and isinstance(stack_result, dict):
                stacks = stack_result.get("stacks", [])
                if isinstance(stacks, list):
                    total_stacks = len(stacks)
                    for s in stacks:
                        g = (s.get("final_grade") or s.get("score") or "").upper().strip()
                        if g == "A":
                            grade_a += 1
                        elif g == "B":
                            grade_b += 1
                        elif g == "C":
                            grade_c += 1
            
            # 데이터 정규화
            result = {
                "timestamp": datetime.now().isoformat(),
                "source": "stacking",
                "data": {
                    "history": history,
                    "latest_result": stack_result
                },
                "summary": {
                    "total_history": len(history),
                    "latest_tsv_num": latest_tsv_num,
                    "total_stacks": total_stacks,
                    "grade_a": grade_a,
                    "grade_b": grade_b,
                    "grade_c": grade_c,
                    "crawled_at": datetime.now().isoformat()
                }
            }
            
            print(f"✅ Stacking 데이터 크롤링 완료: {result['summary']['total_stacks']}개 스택 (A:{grade_a}, B:{grade_b}, C:{grade_c})")
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
            
            # 1. 웨이퍼 목록 조회 (한 번에 전체 수집, 페이지 나누지 않음)
            limit_one_request = 10000
            wafer_list_response = await self.fetch_api_data(f"/wafer/list?page=1&limit={limit_one_request}")
            
            all_wafers = []
            if wafer_list_response and isinstance(wafer_list_response, dict):
                all_wafers = wafer_list_response.get("wafers", [])
                total_wafers_count = wafer_list_response.get("total", 0)
                print(f"📄 웨이퍼 한 번에 수집: {len(all_wafers)}개 (전체: {total_wafers_count}개)")
            else:
                print("⚠️ 웨이퍼 목록 응답 없음")
            
            # 2. 통계 정보 조회 (실패해도 이미 수집한 목록으로 통계 계산 가능)
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
        """모든 dashboard 데이터 크롤링 (로그, 스택, 웨이퍼, 칩 재고)"""
        try:
            print("🚀 모든 Dashboard 데이터 크롤링 시작...")
            
            # 4개 소스 병렬 크롤링 (inventory 포함)
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


