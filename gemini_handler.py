from google import genai
import os
import logging
from typing import Optional
from dotenv import load_dotenv
from pathlib import Path
import time

# 환경 변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 환경변수에서 API 키 가져오기
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# 전역 변수
client = None
chat_history = []  # 채팅 세션 대신 히스토리 관리
READY = False
INITIALIZATION_ATTEMPTS = 0
MAX_RETRIES = 3
REQUEST_COUNT = 0
SUCCESSFUL_REQUESTS = 0
FAILED_REQUESTS = 0
LAST_REQUEST_TIME = None

def initialize_gemini(retry_count=0):
    """Gemini 초기화 (google-genai SDK 사용)"""
    global client, chat_history, READY, INITIALIZATION_ATTEMPTS
    
    INITIALIZATION_ATTEMPTS += 1
    
    try:
        if not GEMINI_API_KEY:
            logger.error("❌ GEMINI_API_KEY 환경변수가 설정되지 않았습니다.")
            READY = False
            return False
        
        logger.info(f"🤖 Gemini 설정 시작... (시도 {INITIALIZATION_ATTEMPTS}회)")
        
        # google-genai Client 초기화
        # 환경변수 GEMINI_API_KEY를 자동으로 인식하거나 명시적으로 설정
        try:
            if GEMINI_API_KEY:
                client = genai.Client(api_key=GEMINI_API_KEY)
            else:
                # 환경변수에서 자동으로 GEMINI_API_KEY 읽음
                client = genai.Client()
        except Exception as client_error:
            logger.error(f"Client 초기화 실패: {client_error}")
            # 대체 방법 시도
            try:
                client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else genai.Client()
            except:
                raise Exception(f"Client 초기화 실패: {client_error}")
        
        logger.info("🧪 Gemini 연결 테스트...")
        
        # 사용 가능한 모델 목록 출력 (비활성화)
        # try:
        #     logger.info("📋 사용 가능한 모델 목록 조회 중...")
        #     available_models = client.models.list()
        #     logger.info("=" * 60)
        #     logger.info("✅ 사용 가능한 Gemini 모델 목록:")
        #     for model in available_models:
        #         model_name = getattr(model, 'name', str(model))
        #         logger.info(f"   - {model_name}")
        #     logger.info("=" * 60)
        # except Exception as list_error:
        #     logger.warning(f"⚠️ 모델 목록 조회 실패: {list_error}")
        
        # 연결 테스트
        test_prompt = "안녕하세요. 'OK'라고 간단히 답변해주세요."
        try:
            test_response = client.models.generate_content(
                model="gemini-2.5-pro",
                contents=test_prompt
            )
        except Exception as test_error:
            logger.error(f"테스트 요청 실패: {test_error}")
            raise
        
        # 응답 텍스트 추출
        test_text = None
        if hasattr(test_response, 'text'):
            test_text = test_response.text
        elif hasattr(test_response, 'candidates') and test_response.candidates:
            test_text = test_response.candidates[0].content.parts[0].text
        elif isinstance(test_response, str):
            test_text = test_response
        
        if test_text and test_text.strip():
            # 채팅 히스토리 초기화
            chat_history = []
            
            logger.info("✅ Gemini 초기화 성공")
            READY = True
            return True
        else:
            raise Exception("테스트 응답이 비어있습니다.")
            
    except Exception as e:
        logger.error(f"❌ Gemini 초기화 실패 (시도 {INITIALIZATION_ATTEMPTS}회): {e}")
        
        # 재시도 로직
        if retry_count < MAX_RETRIES:
            wait_time = (retry_count + 1) * 2
            logger.info(f"🔄 {wait_time}초 후 재시도... ({retry_count + 1}/{MAX_RETRIES})")
            time.sleep(wait_time)
            return initialize_gemini(retry_count + 1)
        else:
            logger.error(f"❌ 최대 재시도 횟수({MAX_RETRIES})를 초과했습니다.")
        
        READY = False
        return False

def get_gemini_response(prompt: str, use_chat_session: bool = False, max_retries: int = 3, apply_format: bool = False, temperature: float = 0.5) -> str:
    """
    Gemini API로부터 응답 받기 (google-genai SDK 사용)
    
    Args:
        prompt: 입력 프롬프트
        use_chat_session: 채팅 세션 사용 여부
        max_retries: 최대 재시도 횟수
        apply_format: 포맷 적용 여부
        temperature: 생성 온도 (0.0 ~ 1.0, 낮을수록 일관성 높음, 높을수록 창의성 높음)
    """
    global client, chat_history, READY, REQUEST_COUNT, SUCCESSFUL_REQUESTS, FAILED_REQUESTS, LAST_REQUEST_TIME
    
    REQUEST_COUNT += 1
    LAST_REQUEST_TIME = time.time()
    
    # 초기화 확인
    if not READY or client is None:
        logger.warning("Gemini가 준비되지 않음, 재초기화 시도")
        if not initialize_gemini():
            FAILED_REQUESTS += 1
            return "[오류] Gemini가 준비되지 않았습니다. API 키와 네트워크 연결을 확인해주세요."
    
    if not prompt or not prompt.strip():
        FAILED_REQUESTS += 1
        return "[오류] 입력 프롬프트가 비어있습니다."
    
    # 프롬프트 길이 제한 확인
    if len(prompt) > 30000:
        logger.warning(f"프롬프트가 너무 깁니다 ({len(prompt)} 문자). 자르기 진행...")
        prompt = prompt[:30000] + "\n\n[프롬프트가 길어서 일부 생략됨]"
    
    # 응답 시도
    for attempt in range(max_retries + 1):
        try:
            logger.info(f"🤖 Gemini 요청 (시도 {attempt + 1}/{max_retries + 1}): {prompt[:100]}...")
            start_time = time.time()
            
            # 시스템 인스트럭션 설정
            system_instruction = "당신은 HBM(High Bandwidth Memory) 제조 및 관리 전문가입니다. 웨이퍼 분석, 적층 구조, 수율 데이터, 재고 관리 등에 대해 정확하고 도움이 되는 답변을 한국어로 제공해주세요."
            
            # temperature에 따라 top_p, top_k 조정
            if temperature < 0.4:
                # 낮은 temperature: 정확성 강화
                top_p = 0.5
                top_k = 20
            else:
                # 높은 temperature: 창의성 강화
                top_p = 0.8
                top_k = 40
            
            if use_chat_session and chat_history:
                # 채팅 히스토리 사용 (컨텍스트 유지)
                # 이전 대화 내용을 포함하여 요청
                contents = chat_history + [{"role": "user", "content": prompt}]
                try:
                    response = client.models.generate_content(
                        model="gemini-2.5-pro",
                        contents=contents,
                        config={
                            "temperature": temperature,
                            "top_p": top_p,
                            "top_k": top_k,
                            "max_output_tokens": 8192,  # 2048에서 8192로 증가 (긴 응답 지원)
                            "system_instruction": system_instruction
                        }
                    )
                except Exception as api_error:
                    # API 형식이 다를 수 있으므로 다른 방식 시도
                    logger.warning(f"첫 번째 API 형식 실패, 대체 방식 시도: {api_error}")
                    response = client.models.generate_content(
                        model="gemini-2.5-pro",
                        contents=contents
                    )
            else:
                # 일반 생성 (매번 새로운 컨텍스트)
                try:
                    response = client.models.generate_content(
                        model="gemini-2.5-pro",
                        contents=prompt,
                        config={
                            "temperature": temperature,
                            "top_p": top_p,
                            "top_k": top_k,
                            "max_output_tokens": 8192,  # 2048에서 8192로 증가 (긴 응답 지원)
                            "system_instruction": system_instruction
                        }
                    )
                except Exception as api_error:
                    # API 형식이 다를 수 있으므로 다른 방식 시도
                    logger.warning(f"첫 번째 API 형식 실패, 대체 방식 시도: {api_error}")
                    response = client.models.generate_content(
                        model="gemini-2.5-pro",
                        contents=prompt
                    )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            # 응답 처리 (google-genai SDK 응답 형식)
            if response:
                # 응답에서 텍스트 추출
                if hasattr(response, 'text'):
                    result = response.text.strip()
                elif hasattr(response, 'candidates') and response.candidates:
                    # candidates 배열에서 텍스트 추출
                    result = response.candidates[0].content.parts[0].text.strip()
                elif isinstance(response, str):
                    result = response.strip()
                else:
                    # 응답 객체에서 직접 텍스트 추출 시도
                    try:
                        result = str(response).strip()
                    except:
                        raise Exception("응답 형식을 파싱할 수 없습니다.")
                
                if result:
                    # 응답 품질 검증
                    if len(result) < 10:
                        logger.warning(f"응답이 너무 짧습니다: {result}")
                    
                    # 채팅 히스토리 업데이트 (채팅 세션 사용 시)
                    if use_chat_session:
                        chat_history.append({"role": "user", "content": prompt})
                        chat_history.append({"role": "assistant", "content": result})
                        # 히스토리 길이 제한 (최근 20개 메시지 유지)
                        if len(chat_history) > 20:
                            chat_history = chat_history[-20:]
                    
                    SUCCESSFUL_REQUESTS += 1
                    logger.info(f"✅ Gemini 응답 성공 (시도 {attempt + 1}, {response_time:.2f}초): {result[:100]}...")
                    
                    return result
                else:
                    raise Exception("빈 응답을 받았습니다.")
            else:
                raise Exception("빈 응답을 받았습니다.")
        
        except Exception as e:
            error_msg = str(e).lower()
            logger.error(f"❌ Gemini 응답 실패 (시도 {attempt + 1}): {e}")
            
            # 치명적인 오류인 경우 즉시 중단
            if "api_key" in error_msg or "permission" in error_msg or "forbidden" in error_msg:
                FAILED_REQUESTS += 1
                logger.error(f"❌ Gemini 권한 오류: {e}")
                return "[오류] API 키 권한이 없습니다. API 키를 확인해주세요."
            
            # 마지막 시도가 아니면 재시도
            if attempt < max_retries:
                wait_time = min((attempt + 1) * 2, 10)
                logger.info(f"🔄 {wait_time}초 후 재시도... ({attempt + 1}/{max_retries})")
                time.sleep(wait_time)
                continue
            
            # 구체적인 오류 메시지 제공
            FAILED_REQUESTS += 1
            
            if "quota" in error_msg or "limit" in error_msg:
                logger.error(f"❌ Gemini 할당량 초과: {e}")
                return "[오류] API 사용량 한도를 초과했습니다. 잠시 후 다시 시도해주세요."
            elif "network" in error_msg or "connection" in error_msg:
                logger.error(f"❌ Gemini 네트워크 오류: {e}")
                return "[오류] 네트워크 연결에 문제가 있습니다. 인터넷 연결을 확인해주세요."
            elif "safety" in error_msg or "blocked" in error_msg:
                logger.error(f"❌ Gemini 안전 필터: {e}")
                return "[오류] 안전 정책에 위반되는 내용으로 판단되어 응답을 생성할 수 없습니다."
            elif "timeout" in error_msg:
                logger.error(f"❌ Gemini 응답 시간 초과: {e}")
                return "[오류] 응답 시간이 초과되었습니다. 잠시 후 다시 시도해주세요."
            else:
                logger.error(f"❌ Gemini 일반 오류: {e}")
                return f"[오류] Gemini 응답 실패: 서버에 일시적인 문제가 있을 수 있습니다. 잠시 후 다시 시도해주세요."
    
    FAILED_REQUESTS += 1
    return "[오류] 최대 재시도 횟수를 초과했습니다."

def get_api_status():
    """API 상태 정보 반환"""
    global LAST_REQUEST_TIME, client, chat_history
    
    last_request_formatted = None
    if LAST_REQUEST_TIME:
        last_request_formatted = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(LAST_REQUEST_TIME))
    
    return {
        "ready": READY,
        "api_key_set": bool(GEMINI_API_KEY),
        "api_key_length": len(GEMINI_API_KEY) if GEMINI_API_KEY else 0,
        "client_loaded": client is not None,
        "chat_history_active": len(chat_history) > 0 if chat_history else False,
        "chat_history_length": len(chat_history) if chat_history else 0,
        "initialization_attempts": INITIALIZATION_ATTEMPTS,
        "model_name": "gemini-2.5-pro",
        "statistics": {
            "total_requests": REQUEST_COUNT,
            "successful_requests": SUCCESSFUL_REQUESTS,
            "failed_requests": FAILED_REQUESTS,
            "success_rate": round((SUCCESSFUL_REQUESTS / REQUEST_COUNT * 100), 2) if REQUEST_COUNT > 0 else 0,
            "last_request_time": last_request_formatted
        }
    }

# 모듈 로드시 자동 초기화
try:
    logger.info("🚀 Gemini 모듈 초기화 시작...")
    success = initialize_gemini()
    if success:
        logger.info("🎉 Gemini 모듈 초기화 완료!")
    else:
        logger.warning("⚠️ Gemini 모듈 초기화 실패 - 기본 모드로 동작")
except Exception as e:
    logger.error(f"모듈 로드 중 Gemini 초기화 실패: {e}")


