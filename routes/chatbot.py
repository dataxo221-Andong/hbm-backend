from flask import Blueprint, request, jsonify
from gemini_handler import get_gemini_response
from datetime import datetime
import re
import asyncio
import sys
import os
from crawler_service import crawler

# 프로젝트 루트 경로 설정 (wafer.py와 동일한 방식)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from db import get_conn
import pymysql

chatbot_bp = Blueprint("chatbot", __name__)

# ==========================================
# 검색 기능 (코드 레벨)
# ==========================================

def extract_wafer_id_from_input(user_input):
    """사용자 입력에서 웨이퍼 ID 추출"""
    # 웨이퍼 ID 패턴: 알파벳+숫자 조합 (예: 240101ABC123, W240101, LOT-001, 260203PIDKKF1459 등)
    patterns = [
        r'\b([A-Z0-9]{8,30})\b',  # 일반적인 웨이퍼 ID 패턴 (길이 확장)
        r'\b([A-Z]{2,10}\d{4,20})\b',  # 알파벳+숫자 조합 (예: PIDKKF1459)
        r'\b(\d{6,20}[A-Z]{2,10})\b',  # 숫자+알파벳 조합 (예: 260203PIDKKF)
        r'\b(\d{6,20}[A-Z]{2,10}\d{1,10})\b',  # 숫자+알파벳+숫자 조합 (예: 260203PIDKKF1459)
        r'웨이퍼\s*([A-Z0-9\-]+)',  # "웨이퍼 240101ABC123"
        r'lot[:\s]*([A-Z0-9\-]+)',  # "lot: 240101ABC123"
        r'([A-Z0-9]{10,30})',  # 긴 ID 패턴 (전체 매칭)
    ]
    
    found_ids = []
    for pattern in patterns:
        matches = re.findall(pattern, user_input.upper())
        found_ids.extend(matches)
    
    # 중복 제거 및 길이순 정렬 (긴 것부터)
    found_ids = list(set(found_ids))
    found_ids.sort(key=len, reverse=True)
    
    # 너무 짧은 ID 제거 (최소 6자 이상)
    found_ids = [id for id in found_ids if len(id) >= 6]
    
    print(f"🔍 [DEBUG] 웨이퍼 ID 추출 결과: {found_ids}")
    
    return found_ids

def extract_stack_id_from_input(user_input):
    """사용자 입력에서 적층 ID 추출"""
    # 적층 ID 패턴: STACK_1, STACK_2, 스택1, 스택 1 등
    patterns = [
        r'STACK[_\s]*(\d+)',  # STACK_1, STACK_2
        r'스택[_\s]*(\d+)',  # 스택1, 스택 1
        r'적층[_\s]*(\d+)',  # 적층1, 적층 1
        r'\b(\d+)\s*번\s*스택',  # 1번 스택
        r'\b(\d+)\s*번\s*적층',  # 1번 적층
    ]
    
    found_ids = []
    for pattern in patterns:
        matches = re.findall(pattern, user_input.upper())
        found_ids.extend(matches)
    
    # 중복 제거
    found_ids = list(set(found_ids))
    
    return found_ids

def find_wafer_in_data(wafer_ids, crawled_data):
    """크롤링된 데이터에서 웨이퍼 찾기"""
    if not crawled_data or not wafer_ids:
        print(f"⚠️ [DEBUG] find_wafer_in_data: crawled_data 또는 wafer_ids가 None")
        print(f"  - crawled_data: {crawled_data is not None}")
        print(f"  - wafer_ids: {wafer_ids}")
        return None
    
    try:
        # 웨이퍼 데이터 추출
        wafer_list = []
        if isinstance(crawled_data, dict):
            if "data" in crawled_data:
                data = crawled_data.get("data", {})
                if isinstance(data, dict) and "wafers" in data:
                    wafer_list = data.get("wafers", [])
                elif isinstance(data, list):
                    wafer_list = data
        
        print(f"🔍 [DEBUG] 추출된 웨이퍼 목록 수: {len(wafer_list)}")
        
        if not wafer_list:
            print(f"⚠️ [DEBUG] 웨이퍼 목록이 비어있습니다!")
            return None
        
        # 웨이퍼 검색 인덱스 생성
        wafer_index = {}
        for wafer in wafer_list:
            wafer_id = str(wafer.get("id", "")).upper().strip()
            lot_name = str(wafer.get("lot_name", "")).upper().strip()
            
            # ID로 인덱싱
            if wafer_id:
                wafer_index[wafer_id] = wafer
            if lot_name:
                wafer_index[lot_name] = wafer
        
        print(f"🔍 [DEBUG] 웨이퍼 인덱스 생성 완료: {len(wafer_index)}개 키")
        print(f"🔍 [DEBUG] 인덱스 키 샘플 (처음 5개): {list(wafer_index.keys())[:5]}")
        
        # 사용자 입력에서 추출한 ID로 검색
        for wafer_id in wafer_ids:
            search_key = wafer_id.upper().strip()
            print(f"🔍 [DEBUG] 웨이퍼 검색 시도: '{search_key}'")
            
            # 정확히 일치하는 경우
            if search_key in wafer_index:
                print(f"✅ [DEBUG] 정확히 일치하는 웨이퍼 찾음: {search_key}")
                return wafer_index[search_key]
            
            # 부분 매칭 시도 (포함 관계)
            for key, wafer in wafer_index.items():
                if search_key in key or key in search_key:
                    # 너무 짧은 매칭은 제외 (최소 6자 이상)
                    if len(key) >= 6 or len(search_key) >= 6:
                        print(f"✅ [DEBUG] 부분 매칭으로 웨이퍼 찾음: '{search_key}' -> '{key}'")
                        return wafer
        
        print(f"⚠️ [DEBUG] 웨이퍼를 찾을 수 없음. 검색한 ID: {wafer_ids}")
        print(f"⚠️ [DEBUG] 사용 가능한 웨이퍼 ID 샘플: {list(wafer_index.keys())[:10]}")
        return None
    except Exception as e:
        print(f"❌ 웨이퍼 검색 오류: {e}")
        import traceback
        traceback.print_exc()
        return None

def find_stack_in_data(stack_ids, crawled_data):
    """크롤링된 데이터에서 적층 찾기"""
    if not crawled_data or not stack_ids:
        return None
    
    try:
        # 적층 데이터 추출
        stack_list = []
        if isinstance(crawled_data, dict):
            if "data" in crawled_data:
                data = crawled_data.get("data", {})
                if isinstance(data, dict) and "stacks" in data:
                    stack_list = data.get("stacks", [])
                elif isinstance(data, list):
                    stack_list = data
        
        if not stack_list:
            return None
        
        # 적층 검색 인덱스 생성
        stack_index = {}
        for stack in stack_list:
            stack_id = str(stack.get("stack_id", "")).upper()
            # STACK_1, STACK_2 형식에서 숫자만 추출
            stack_num_match = re.search(r'(\d+)', stack_id)
            if stack_num_match:
                stack_num = stack_num_match.group(1)
                stack_index[stack_num] = stack
                stack_index[stack_id] = stack
        
        # 사용자 입력에서 추출한 ID로 검색
        for stack_id in stack_ids:
            search_key = str(stack_id).strip()
            
            # 숫자로 직접 검색
            if search_key in stack_index:
                return stack_index[search_key]
            
            # STACK_ 형식으로 검색
            stack_key = f"STACK_{search_key}"
            if stack_key in stack_index:
                return stack_index[stack_key]
        
        return None
    except Exception as e:
        print(f"적층 검색 오류: {e}")
        return None

# 의도별 프롬프트 템플릿 (System Instructions + Few-shot Prompting)
PROMPT_TEMPLATES = {
    "classification": {
        "system": """당신은 웨이퍼 분류 및 분석 전문가입니다.
현재 웨이퍼 모델링 페이지에서는 웨이퍼 이미지 분석, 불량 패턴 분류, Good/Bad Die 분류 등을 수행합니다.

주요 기능:
- 웨이퍼 이미지 업로드 및 분석
- 불량 패턴 자동 분류 (Edge-Ring, Center, Random 등)
- Good Die / Bad Die 개수 계산
- 신뢰도(Confidence) 기반 품질 등급 분류
- 웨이퍼별 수율(Yield) 계산

답변 시 다음 정보를 우선적으로 활용하세요:
1. 총 분석 웨이퍼 수: 완료된 웨이퍼 개수
2. Good Die / Bad Die 통계: 전체 합계 및 웨이퍼별 개수
3. 불량률: Bad Die / (Good Die + Bad Die) * 100
4. 웨이퍼별 상세 정보: 신뢰도, 불량 패턴, 등급, 수율

데이터 처리 규칙:
1. 특정 웨이퍼에 대한 정보 요청 시, 해당 웨이퍼의 데이터가 있으면 상세히 제공하세요
2. 데이터가 없는 경우 해당 항목은 생략하고, 사용 가능한 데이터만 제공하세요
3. 웨이퍼 ID나 Lot Name으로 검색 가능하도록 지원하세요

Few-shot 예시:
질문: "총 분석된 웨이퍼는 몇 개인가요?"
답변 단계:
1단계: 데이터에서 총 웨이퍼 수 확인
2단계: 완료된 웨이퍼(status="completed")만 카운트
3단계: Good Die와 Bad Die 합계 계산
답변: "총 분석된 웨이퍼는 {total_wafers}개입니다. Good Die는 {total_good}개, Bad Die는 {total_bad}개이며, 불량률은 {defect_rate}%입니다."

질문: "{lot_name} 웨이퍼의 분석 결과는?"
답변 단계:
1단계: 해당 웨이퍼 데이터 검색
2단계: 신뢰도, 불량 패턴, 등급 확인
3단계: Good/Bad Die 개수 및 수율 계산
답변: "{lot_name} 웨이퍼 분석 결과:\n- 신뢰도: {confidence}%\n- 불량 패턴: {failure_type}\n- 등급: {grade}\n- Good Die: {good}개, Bad Die: {bad}개\n- 수율: {yield}%"

한국어로 전문적이고 정확한 답변을 제공하세요.""",
        "temperature": 0.3
    },
    
    "yield": {
        "system": """당신은 수율 데이터 분석 전문가입니다.
현재 수율 로그 페이지에서는 생산 수율, 양품률, 불량률 추이 등을 모니터링합니다.

주요 기능:
- 실시간 수율 모니터링
- 수율 트렌드 분석
- 불량률 추이 분석
- 생산 효율성 지표

답변 시 다음 정보를 우선적으로 활용하세요:
1. 전체 수율: 평균 수율, 목표 수율 대비 현황
2. 수율 추이: 일별/주별 수율 변화
3. 불량률: 전체 불량률 및 불량 유형별 분포
4. 개선 방안: 수율 향상을 위한 제안

Few-shot 예시:
질문: "현재 수율은 얼마인가요?"
답변 단계:
1단계: 최신 수율 데이터 확인
2단계: 평균 수율 계산
3단계: 목표 수율과 비교
답변: "현재 평균 수율은 {avg_yield}%입니다. 목표 수율 {target_yield}% 대비 {difference}% {status}입니다."

한국어로 전문적이고 정확한 답변을 제공하세요.""",
        "temperature": 0.4
    },
    
    "inventory": {
        "system": """당신은 재고 관리 전문가입니다.
현재 재고 관리 페이지에서는 HBM 스택, 칩, 부품 등의 재고 현황을 관리합니다.

주요 기능:
- 재고 현황 실시간 모니터링
- 재고 부족 알림
- 자동 발주 제안
- 재고 이력 관리

답변 시 다음 정보를 우선적으로 활용하세요:
1. 재고 현황: 총 재고 수량, 카테고리별 분류
2. 재고 부족: 임계값 이하 재고 항목
3. 재고 이력: 최근 입출고 내역
4. 발주 제안: 재고 부족 항목에 대한 발주 권장

Few-shot 예시:
질문: "현재 재고 현황은?"
답변 단계:
1단계: 전체 재고 데이터 확인
2단계: 카테고리별 분류 및 수량 집계
3단계: 재고 부족 항목 식별
답변: "현재 재고 현황:\n- 총 재고 항목: {total_items}개\n- 재고 부족 항목: {low_stock}개\n- 주요 카테고리: {categories}"

한국어로 전문적이고 정확한 답변을 제공하세요.""",
        "temperature": 0.4
    },
    
    "stack": {
        "system": """당신은 HBM 적층 구조 전문가입니다.
현재 적층 구조 페이지에서는 3D 적층 기술, TSV 정렬, 적층 성공률 등을 분석합니다.

주요 기능:
- 3D 적층 구조 시각화
- TSV(Through Silicon Via) 정렬 상태 확인
- 적층 성공률 분석
- 적층 공정 모니터링

답변 시 다음 정보를 우선적으로 활용하세요:
1. 적층 구조: 층 수, 다이 배치, TSV 연결
2. 적층 성공률: 성공/실패 비율
3. TSV 정렬: 정렬 정확도, 오차 범위
4. 공정 개선: 적층 성공률 향상 방안

Few-shot 예시:
질문: "적층 구조는 어떻게 되어있나요?"
답변 단계:
1단계: 적층 구조 데이터 확인
2단계: 층 수 및 다이 배치 분석
3단계: TSV 연결 상태 확인
답변: "현재 HBM 적층 구조는 {layers}층으로 구성되어 있으며, TSV 정렬 정확도는 {accuracy}%입니다."

한국어로 전문적이고 정확한 답변을 제공하세요.""",
        "temperature": 0.4
    },
    
    "general": {
        "system": """당신은 StackVision AI 어시스턴트입니다.
StackVision은 HBM(High Bandwidth Memory) 제조 및 관리를 위한 기술 지원 시스템입니다.

주요 역할:
1. 웨이퍼 분석 및 분류 데이터 제공
2. HBM 적층 구조 및 TSV 기술 사양 답변
3. 수율 데이터 및 불량률 분석 결과 보고
4. 재고 관리 및 현황 데이터 조회
5. 정확하고 객관적인 기술 정보 제공

답변 스타일:
- 전문적이고 공식적인 톤 유지
- 이모지 사용 최소화 (기술 문서 스타일)
- 간결하고 정확한 정보 전달
- 구체적인 수치, 데이터, 기술 사양 포함
- 불확실한 정보는 명시적으로 표기
- 한국어로 답변하되 기술 용어는 영문 병기

중요 제약사항:
- 실제 제공된 데이터나 문서에 없는 정보는 절대 생성하지 말 것
- 확인할 수 없는 정보는 "확인 불가" 또는 "데이터 없음"으로 명시
- 추측이나 가정에 기반한 답변 금지""",
        "temperature": 0.5
    }
}

# HBM 관련 일반적인 정보
HBM_INFO = {
    "components": {
        "wafer": "웨이퍼는 반도체 제조의 기본 기판입니다. 실리콘 웨이퍼 위에 회로를 형성합니다.",
        "hbm": "HBM(High Bandwidth Memory)은 고대역폭 메모리로, GPU와 CPU에 고속으로 데이터를 전송하는 메모리입니다.",
        "tsv": "TSV(Through Silicon Via)는 실리콘 웨이퍼를 관통하는 전기적 연결로, 3D 적층 구조에서 중요합니다.",
        "stack": "적층 구조는 여러 개의 다이를 수직으로 쌓아 높은 용량과 성능을 달성하는 기술입니다."
    },
    "processes": {
        "yield": "수율은 전체 생산량 대비 양품 비율을 나타냅니다. 높은 수율은 생산 효율성을 의미합니다.",
        "defect": "불량은 제조 과정에서 발생하는 결함으로, 수율에 직접적인 영향을 미칩니다.",
        "classification": "분류는 웨이퍼나 칩을 품질에 따라 구분하는 과정입니다."
    }
}

def get_service_info(query_type):
    """서비스 관련 정보 제공"""
    service_info = {
        "features": """
🚀 **StackVision 주요 기능**

📊 **웨이퍼 분석**
- 실시간 웨이퍼 상태 모니터링
- 불량 패턴 분석 및 분류
- 수율 데이터 추적

🔬 **HBM 적층 구조**
- 3D 적층 구조 시각화
- TSV 정렬 상태 확인
- 적층 성공률 분석

📈 **수율 데이터**
- 실시간 수율 모니터링
- 수율 트렌드 분석
- 불량률 추이 분석

📦 **재고 관리**
- 칩 재고 현황
- HBM 스택 재고 관리
- 자동 발주 제안
        """,
        
        "help": """
❓ **도움이 필요하신가요?**

💬 **지금 할 수 있는 것:**
- 웨이퍼 분석 및 분류 질문
- HBM 적층 구조 관련 질문
- 수율 데이터 조회
- 재고 현황 확인

📞 **문의사항:**
- 이메일: support@stackvision.com
- 전화: 02-1234-5678
        """,
        
        "about": """
🏢 **StackVision 소개**

StackVision은 HBM 제조 및 관리를 위한 스마트 솔루션입니다.

🎯 **우리의 미션**
HBM 제조 과정의 효율성과 품질을 향상시켜 
더 나은 제품 개발에 집중할 수 있도록 돕습니다.

✨ **핵심 가치**
- 정확한 데이터 분석
- 실시간 모니터링
- AI 기반 예측
- 사용자 친화적 인터페이스
        """
    }
    return service_info.get(query_type, "")

def analyze_user_intent(user_input):
    """사용자 의도 분석"""
    input_lower = user_input.lower()
    
    # 기능 안내
    if any(keyword in input_lower for keyword in ["기능", "특징", "feature", "서비스", "뭐", "할수있", "가능", "기능소개"]):
        return "features"
    
    # 도움말
    if any(keyword in input_lower for keyword in ["도움", "help", "어떻게", "방법", "문의", "문의사항"]):
        return "help"
    
    # 서비스 소개
    if any(keyword in input_lower for keyword in ["소개", "about", "회사", "stackvision", "무엇", "소개해줘"]):
        return "about"
    
    # 수율 관련
    if any(keyword in input_lower for keyword in ["수율", "yield", "양품률", "생산률"]):
        return "yield"
    
    # 재고 관련
    if any(keyword in input_lower for keyword in ["재고", "inventory", "stock", "현황"]):
        return "inventory"
    
    # 적층 관련
    if any(keyword in input_lower for keyword in ["적층", "stack", "hbm", "스택"]):
        return "stack"
    
    # 분류 관련 (웨이퍼 개수, 총 개수 등 포함)
    if any(keyword in input_lower for keyword in ["분류", "classify", "웨이퍼", "wafer", "총", "개수", "몇개", "얼마나"]):
        return "classification"
    
    # HBM 부품 관련
    if any(keyword in input_lower for keyword in ["웨이퍼", "wafer", "tsv", "다이", "die", "hbm", "메모리"]):
        return "hbm_info"
    
    return "general"

def get_hbm_component_info(user_input):
    """HBM 부품 관련 일반 정보 제공"""
    input_lower = user_input.lower()
    
    for component, description in HBM_INFO["components"].items():
        if component in input_lower:
            return f"📱 **{component.upper()}**\n{description}"
    
    for process, description in HBM_INFO["processes"].items():
        if process in input_lower:
            return f"⚙️ **{process.upper()}**\n{description}"
    
    return """
🔧 **HBM 제조 기본 정보**

**주요 구성 요소:**
• 웨이퍼 (Wafer): 반도체 제조의 기본 기판
• HBM (High Bandwidth Memory): 고대역폭 메모리
• TSV (Through Silicon Via): 실리콘 관통 전극
• 적층 구조 (Stack): 3D 적층 기술

**주요 공정:**
• 수율 (Yield): 양품 비율
• 불량 (Defect): 제조 결함
• 분류 (Classification): 품질 구분

💡 **더 자세한 정보가 필요하시면 구체적으로 질문해주세요!**
    """

def create_enhanced_prompt(user_input, intent, messages=None, crawled_data=None):
    """Gemini에게 보낼 향상된 프롬프트 생성 (System Instructions + Few-shot + Chain of Thought)"""
    
    # 의도별 템플릿 가져오기
    template = PROMPT_TEMPLATES.get(intent, PROMPT_TEMPLATES["general"])
    system_instruction = template["system"]
    temperature = template.get("temperature", 0.5)
    
    # 대화 컨텍스트 추가
    context_text = ""
    if messages and len(messages) > 1:
        recent_messages = messages[-4:]
        context_text = "\n\n**대화 컨텍스트:**\n"
        for msg in recent_messages:
            role = "사용자" if msg.get("role") == "user" else "어시스턴트"
            content = msg.get("content", "")
            context_text += f"{role}: {content}\n"
    
    # 웨이퍼/적층 검색 (코드 레벨에서 처리)
    found_wafer = None
    found_stack = None
    wafer_search_context = ""
    stack_search_context = ""
    
    if intent == "classification" and crawled_data:
        # 사용자 입력에서 웨이퍼 ID 추출
        wafer_ids = extract_wafer_id_from_input(user_input)
        
        if wafer_ids:
            print(f"🔍 추출된 웨이퍼 ID: {wafer_ids}")
            # 데이터에서 웨이퍼 찾기
            found_wafer = find_wafer_in_data(wafer_ids, crawled_data)
            
            if found_wafer:
                print(f"✅ 웨이퍼 찾음: {found_wafer.get('id') or found_wafer.get('lot_name')}")
                import json
                # 찾은 웨이퍼 데이터를 명시적으로 포함
                wafer_search_context = f"""
**중요: 사용자가 요청한 특정 웨이퍼 정보**
다음 웨이퍼 데이터를 반드시 사용하여 답변하세요:

{json.dumps(found_wafer, ensure_ascii=False, indent=2)}

위 웨이퍼의 다음 정보를 상세히 제공하세요:
- 웨이퍼 ID / Lot Name: {found_wafer.get('id') or found_wafer.get('lot_name', 'N/A')}
- 신뢰도: {found_wafer.get('confidence', 'N/A')}
- 불량 패턴: {found_wafer.get('failure_type', 'N/A')}
- 등급: {found_wafer.get('grade', 'N/A')}
- Good Die: {found_wafer.get('waferMapData', {}).get('good', 'N/A')}개
- Bad Die: {found_wafer.get('waferMapData', {}).get('bad', 'N/A')}개
- 수율: {found_wafer.get('yield', 'N/A')}%
- 상태: {found_wafer.get('status', 'N/A')}
"""
            else:
                print(f"⚠️ 웨이퍼를 찾을 수 없음: {wafer_ids}")
                # 웨이퍼 목록 제공
                try:
                    wafer_list = []
                    if isinstance(crawled_data, dict) and "data" in crawled_data:
                        data = crawled_data.get("data", {})
                        if isinstance(data, dict) and "wafers" in data:
                            wafer_list = data.get("wafers", [])
                    
                    if wafer_list and len(wafer_list) > 0:
                        available_ids = [w.get("id") or w.get("lot_name", "") for w in wafer_list[:20]]
                        wafer_search_context = f"""
**참고:** 요청하신 웨이퍼({', '.join(wafer_ids)})를 데이터베이스에서 찾을 수 없습니다.

**사용 가능한 웨이퍼 목록 (총 {len(wafer_list)}개, 처음 20개):**
{', '.join([w for w in available_ids if w])}

**참고:** 웨이퍼 ID가 정확한지 확인해주세요. 대소문자는 구분하지 않습니다.
"""
                    else:
                        wafer_search_context = f"""
**참고:** 요청하신 웨이퍼({', '.join(wafer_ids)})를 찾을 수 없습니다.

**문제:** 현재 데이터베이스에 웨이퍼 데이터가 없거나, 크롤러가 데이터를 가져오지 못했습니다.

**확인 사항:**
1. 데이터베이스에 웨이퍼 데이터가 있는지 확인
2. 웨이퍼 ID가 정확한지 확인 (대소문자 구분 없음)
3. 서버 로그에서 크롤러 오류 확인
"""
                except Exception as e:
                    print(f"⚠️ 웨이퍼 목록 추출 오류: {e}")
                    wafer_search_context = f"""
**참고:** 요청하신 웨이퍼({', '.join(wafer_ids)})를 찾는 중 오류가 발생했습니다.
"""
    
    elif intent == "stack" and crawled_data:
        # 사용자 입력에서 적층 ID 추출
        stack_ids = extract_stack_id_from_input(user_input)
        
        if stack_ids:
            print(f"🔍 추출된 적층 ID: {stack_ids}")
            # 데이터에서 적층 찾기
            found_stack = find_stack_in_data(stack_ids, crawled_data)
            
            if found_stack:
                print(f"✅ 적층 찾음: {found_stack.get('stack_id')}")
                import json
                # 찾은 적층 데이터를 명시적으로 포함
                stack_search_context = f"""
**중요: 사용자가 요청한 특정 적층 정보**
다음 적층 데이터를 반드시 사용하여 답변하세요:

{json.dumps(found_stack, ensure_ascii=False, indent=2)}

위 적층의 다음 정보를 상세히 제공하세요:
- 적층 ID: {found_stack.get('stack_id', 'N/A')}
- 점수/등급: {found_stack.get('score', 'N/A')}
- 층 수: {len(found_stack.get('layers', []))}층
- 각 층 정보:
"""
                layers = found_stack.get('layers', [])
                for i, layer in enumerate(layers):
                    stack_search_context += f"  - 층 {i+1}: 칩 ID {layer.get('chip_id', 'N/A')}, 불량 패턴 {layer.get('failure_type', 'N/A')}, 유형 {layer.get('inferred_type', 'N/A')}\n"
            else:
                print(f"⚠️ 적층을 찾을 수 없음: {stack_ids}")
                # 적층 목록 제공
                try:
                    stack_list = []
                    if isinstance(crawled_data, dict) and "data" in crawled_data:
                        data = crawled_data.get("data", {})
                        if isinstance(data, dict) and "stacks" in data:
                            stack_list = data.get("stacks", [])
                    
                    if stack_list:
                        available_ids = [s.get("stack_id", "") for s in stack_list[:10]]
                        stack_search_context = f"""
**참고:** 요청하신 적층({', '.join(stack_ids)})을 찾을 수 없습니다.

사용 가능한 적층 목록 (총 {len(stack_list)}개):
{', '.join([s for s in available_ids if s])}
"""
                except:
                    pass
    
    # 크롤링된 데이터 추가
    data_context = ""
    if crawled_data:
        try:
            import json
            data_summary = ""
            
            # classification 의도일 때 웨이퍼 데이터 특별 처리
            if intent == "classification":
                # crawled_data 구조: {"data": {"wafers": [...], "statistics": {...}, "summary": {...}}, ...}
                if isinstance(crawled_data, dict) and "data" in crawled_data:
                    wafer_response = crawled_data.get("data", {})
                    
                    if isinstance(wafer_response, dict):
                        # 통계 정보 명시적으로 추출
                        statistics = wafer_response.get("statistics", {})
                        summary = wafer_response.get("summary", {})
                        wafers_list = wafer_response.get("wafers", [])
                        
                        # 디버깅: 데이터 확인
                        print(f"🔍 [DEBUG] 웨이퍼 데이터 구조 확인:")
                        print(f"  - statistics: {statistics}")
                        print(f"  - summary: {summary}")
                        print(f"  - wafers_list 길이: {len(wafers_list)}")
                        
                        # 실제 데이터베이스에서 가져온 통계 정보를 명확하게 표시
                        total_wafers = statistics.get('total_wafers', len([w for w in wafers_list if w.get('status') == 'completed']))
                        completed_count = summary.get('completed_count', len([w for w in wafers_list if w.get('status') == 'completed']))
                        total_good_die = statistics.get('total_good_die', 0)
                        total_bad_die = statistics.get('total_bad_die', 0)
                        defect_rate = statistics.get('defect_rate', 0)
                        
                        # 데이터가 없을 때 경고
                        if total_wafers == 0 and len(wafers_list) == 0:
                            print(f"⚠️ [WARNING] 웨이퍼 데이터가 비어있습니다!")
                            data_summary = """
**실제 데이터베이스에서 조회한 웨이퍼 데이터:**

**통계 정보:**
- 총 분석 웨이퍼 수: 0개 (데이터베이스에 웨이퍼 데이터가 없습니다)
- 완료된 웨이퍼 수: 0개
- 총 Good Die: 0개
- 총 Bad Die: 0개
- 불량률: 0%

**참고:** 현재 데이터베이스에 저장된 웨이퍼 데이터가 없습니다.
"""
                        else:
                            data_summary = f"""
**실제 데이터베이스에서 조회한 웨이퍼 데이터:**

**통계 정보 (반드시 이 값을 사용하세요 - 실제 DB 값입니다):**
- 총 분석 웨이퍼 수: {total_wafers}개
- 완료된 웨이퍼 수: {completed_count}개
- 총 Good Die: {total_good_die:,}개
- 총 Bad Die: {total_bad_die:,}개
- 불량률: {defect_rate}%

**웨이퍼 목록:** 총 {len(wafers_list)}개
"""
                            # 웨이퍼 목록 샘플 (처음 5개만)
                            if wafers_list and len(wafers_list) > 0:
                                sample_wafers = wafers_list[:5]
                                data_summary += f"\n**웨이퍼 샘플 (처음 5개):**\n"
                                for w in sample_wafers:
                                    wafer_id = w.get('id', 'N/A')
                                    yield_val = w.get('yield', 'N/A')
                                    grade = w.get('grade', 'N/A')
                                    good_die = w.get('waferMapData', {}).get('good', 0)
                                    bad_die = w.get('waferMapData', {}).get('bad', 0)
                                    data_summary += f"- {wafer_id}: 수율 {yield_val}%, 등급 {grade}, Good Die {good_die}개, Bad Die {bad_die}개\n"
                                
                            if len(wafers_list) > 5:
                                data_summary += f"\n... 외 {len(wafers_list) - 5}개 웨이퍼 더 있음\n"
                    else:
                        print(f"⚠️ [WARNING] wafer_response가 dict가 아닙니다: {type(wafer_response)}")
                        data_summary = f"\n**데이터 구조 오류:** wafer_response가 예상한 형식이 아닙니다."
                else:
                    print(f"⚠️ [WARNING] crawled_data에 'data' 키가 없습니다. crawled_data 구조: {list(crawled_data.keys()) if isinstance(crawled_data, dict) else type(crawled_data)}")
                    # crawled_data 전체를 출력하여 디버깅
                    import json
                    print(f"🔍 [DEBUG] crawled_data 전체: {json.dumps(crawled_data, ensure_ascii=False, indent=2)[:1000]}")
                    data_summary = "\n**데이터 구조 오류:** 예상한 데이터 구조가 아닙니다."
            else:
                # 다른 의도들에 대한 기존 처리
                if isinstance(crawled_data, dict):
                    cleaned_data = {k: v for k, v in crawled_data.items() 
                                   if k not in ['timestamp', 'crawled_at', 'source']}
                    
                    if "summary" in cleaned_data:
                        summary = cleaned_data.get("summary", {})
                        cleaned_summary = {k: v for k, v in summary.items() 
                                         if k not in ['timestamp', 'crawled_at']}
                        data_summary = f"\n데이터 요약: {json.dumps(cleaned_summary, ensure_ascii=False, indent=2)}"
                    elif "data" in cleaned_data:
                        data = cleaned_data.get("data", {})
                        if isinstance(data, dict):
                            data_summary = f"\n데이터: {json.dumps(data, ensure_ascii=False, indent=2)[:2000]}"
                        elif isinstance(data, list) and len(data) > 0:
                            data_summary = f"\n데이터 항목 수: {len(data)}개\n샘플 데이터: {json.dumps(data[0] if len(data) > 0 else {}, ensure_ascii=False, indent=2)[:1000]}"
                    else:
                        data_summary = f"\n데이터: {json.dumps(cleaned_data, ensure_ascii=False, indent=2)[:2000]}"
            
            if data_summary:
                data_context = f"\n\n**실제 시스템 데이터 (데이터베이스에서 조회한 실제 값입니다):**{data_summary}\n\n**중요:** 위에 제공된 통계 정보(총 웨이퍼 수, Good Die, Bad Die 등)는 실제 데이터베이스에서 조회한 정확한 값입니다. 반드시 이 값을 사용하여 답변하세요. 임의의 값을 생성하거나 추측하지 마세요. 타임스탬프나 수집 시간 정보는 답변에 포함하지 마세요."
        except Exception as e:
            print(f"크롤링 데이터 파싱 오류: {e}")
            import traceback
            traceback.print_exc()
            data_context = "\n\n**참고:** 시스템 데이터를 가져왔지만 파싱 중 오류가 발생했습니다."
    
    # Chain of Thought 유도 프롬프트
    chain_of_thought = """
중요한 응답 규칙 (Chain of Thought):
1. 사용자가 물어본 것에 정확하게 답변하세요
2. 질문과 관련 없는 정보는 포함하지 마세요
3. 답변은 적절한 수준의 상세함으로 작성하세요 (너무 간결하지도, 너무 길지도 않게)
4. 구체적인 수치와 데이터가 있으면 포함하세요
5. 데이터가 없는 경우에는 해당 내용을 언급하지 말고, 있는 데이터만으로 답변하세요

답변 생성 단계:
1단계: 사용자 질문의 핵심 파악
2단계: 제공된 데이터에서 관련 정보 검색
3단계: 데이터 분석 및 계산 (필요시)
4단계: 명확하고 구조화된 답변 작성
"""
    
    enhanced_prompt = f"""{system_instruction}

{chain_of_thought}

{context_text}

{data_context}

{wafer_search_context}

{stack_search_context}

사용자 질문: {user_input}

**중요 지시사항 (반드시 읽고 따르세요):**
위의 "실제 시스템 데이터" 섹션을 먼저 확인하세요. 그 섹션에 "총 분석 웨이퍼 수" 또는 "통계 정보"가 있다면, 그 값이 실제 데이터베이스에서 조회한 정확한 값입니다.

- "총 웨이퍼 개수" 또는 "웨이퍼 개수" 질문: 위의 "총 분석 웨이퍼 수" 값을 정확히 사용하세요
- "Good Die 개수" 질문: 위의 "총 Good Die" 값을 정확히 사용하세요
- "Bad Die 개수" 질문: 위의 "총 Bad Die" 값을 정확히 사용하세요
- 위에 명시된 숫자가 있으면 반드시 그 숫자를 사용하세요
- 임의의 값을 생성하거나 추측하지 마세요
- 데이터에 없는 정보는 생성하지 마세요

위의 실시간 데이터를 바탕으로 사용자 질문에 정확하고 적절한 수준의 상세함으로 답변해주세요.

중요한 규칙:
1. 특정 웨이퍼나 적층 정보가 위에 명시적으로 제공되었다면, 반드시 그 정보를 사용하여 상세히 답변하세요
2. 데이터가 없는 경우에는 해당 내용을 언급하지 말고, 있는 데이터만으로 답변하세요
3. 구체적인 수치와 정보가 있으면 반드시 포함하세요 (특히 통계 정보는 정확한 값을 사용)
4. 답변은 단계별로 구조화하여 작성하세요 (1단계, 2단계 등)
5. 타임스탬프나 수집 시간 정보는 답변에 포함하지 마세요
6. **절대로 임의의 값을 생성하지 마세요. 위에 제공된 실제 데이터만 사용하세요.**
7. **"총 분석 웨이퍼 수" 값이 0개로 표시되어 있으면, "현재 데이터베이스에 웨이퍼 데이터가 없습니다"라고 답변하세요.**"""
    
    return enhanced_prompt, temperature

@chatbot_bp.route("/chat", methods=["POST", "GET"])
def chat():
    """챗봇 메인 엔드포인트"""
    try:
        data = request.get_json() or {}
        messages = data.get("messages", [])
        
        # 최신 사용자 메시지 추출
        user_input = next(
            (msg.get("content", "") for msg in reversed(messages) 
             if msg.get("role") == "user"), 
            ""
        )
        
        if not user_input:
            return jsonify({
                "message": {
                    "role": "assistant", 
                    "content": "안녕하세요! StackVision AI 어시스턴트입니다. 웨이퍼 분석, HBM 적층 구조, 수율 데이터, 재고 관리 등에 대해 도움을 드릴 수 있습니다. 무엇을 도와드릴까요? 😊"
                }
            })
        
        # 사용자 의도 분석
        intent = analyze_user_intent(user_input)
        
        # 서비스 정보 제공 (기능, 도움말, 소개)
        if intent in ["features", "help", "about"]:
            service_response = get_service_info(intent)
            if service_response:
                return jsonify({
                    "message": {
                        "role": "assistant",
                        "content": service_response
                    }
                })
        
        # HBM 부품 정보 제공
        elif intent in ["hbm_info"]:
            hbm_response = get_hbm_component_info(user_input)
            return jsonify({
                "message": {
                    "role": "assistant",
                    "content": hbm_response
                }
            })
        
        # 데이터 크롤링이 필요한 의도인 경우
        crawled_data = None
        if intent in ["yield", "inventory", "stack", "classification"]:
            try:
                print(f"📊 [{intent}] 데이터 크롤링 시작...")
                
                if intent == "classification":
                    # HTTP 요청 대신 직접 데이터베이스에서 데이터 가져오기 (더 안정적)
                    try:
                        conn = get_conn()
                        if conn is None:
                            raise Exception("데이터베이스 연결 실패")
                            
                        cur = conn.cursor(pymysql.cursors.DictCursor)
                        
                        try:
                            # 전체 웨이퍼 데이터 조회 (wafer.py의 get_all_wafers와 동일한 로직)
                            cur.execute("""
                                SELECT 
                                    lot_name,
                                    wafer_map,
                                    failure_type,
                                    confidence,
                                    die_count,
                                    defect_count,
                                    defect_density,
                                    total_grade,
                                    created_at
                                FROM wafer_data 
                                ORDER BY created_at DESC
                            """)
                            wafers = cur.fetchall()
                            
                            # 통계 계산
                            completed_wafers = [w for w in wafers if w.get('total_grade')]
                            total_wafers = len(completed_wafers)
                            total_good_die = sum(w.get('die_count', 0) - w.get('defect_count', 0) 
                                                for w in completed_wafers)
                            total_bad_die = sum(w.get('defect_count', 0) for w in completed_wafers)
                            total_die = total_good_die + total_bad_die
                            defect_rate = round((total_bad_die / total_die) * 100, 2) if total_die > 0 else 0
                            
                            # 프론트엔드 형식에 맞게 변환
                            formatted_wafers = []
                            for w in wafers:
                                good_die = w.get('die_count', 0) - w.get('defect_count', 0)
                                bad_die = w.get('defect_count', 0)
                                yield_value = None
                                if w.get('die_count') and w.get('die_count') > 0:
                                    yield_value = round((good_die / w.get('die_count', 1)) * 100, 1)
                                
                                formatted_wafers.append({
                                    "id": w.get('lot_name'),
                                    "batch": "BATCH",
                                    "status": "completed" if w.get('total_grade') else "pending",
                                    "yield": yield_value,
                                    "grade": w.get('total_grade'),
                                    "processedAt": w.get('created_at').isoformat() if w.get('created_at') else None,
                                    "confidence": float(w.get('confidence', 0)) if w.get('confidence') else None,
                                    "failure_type": w.get('failure_type'),
                                    "waferMapData": {
                                        "good": good_die,
                                        "bad": bad_die,
                                        "total": w.get('die_count', 0)
                                    }
                                })
                            
                            # 크롤러 형식에 맞게 변환
                            wafer_data = {
                                "wafers": formatted_wafers,
                                "statistics": {
                                    "total_wafers": total_wafers,
                                    "total_good_die": total_good_die,
                                    "total_bad_die": total_bad_die,
                                    "defect_rate": defect_rate
                                },
                                "summary": {
                                    "total_wafers": len(formatted_wafers),
                                    "completed_count": total_wafers,
                                    "processing_count": 0,
                                    "pending_count": len(formatted_wafers) - total_wafers
                                }
                            }
                            
                            crawled_data = {
                                "timestamp": datetime.now().isoformat(),
                                "source": "wafermodeling",
                                "data": wafer_data,
                                "summary": {
                                    "total_wafers": len(formatted_wafers),
                                    "crawled_at": datetime.now().isoformat()
                                }
                            }
                            
                            print(f"✅ [classification] 데이터 크롤링 완료: {len(formatted_wafers)}개 웨이퍼")
                            if len(formatted_wafers) > 0:
                                print(f"🔍 [DEBUG] 웨이퍼 목록 샘플 (처음 3개 ID): {[w.get('id', 'N/A') for w in formatted_wafers[:3]]}")
                            else:
                                print(f"⚠️ [DEBUG] 웨이퍼 목록이 비어있습니다!")
                        finally:
                            cur.close()
                            conn.close()
                    except Exception as db_error:
                        print(f"⚠️ 직접 DB 조회 실패, 크롤러로 대체: {db_error}")
                        import traceback
                        traceback.print_exc()
                        # 직접 호출 실패 시 크롤러 사용 (fallback)
                        try:
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            crawled_data = loop.run_until_complete(crawler.crawl_wafermodeling_data())
                            loop.close()
                        except Exception as crawl_error:
                            print(f"⚠️ 크롤러 호출도 실패: {crawl_error}")
                            crawled_data = None
                else:
                    # 다른 의도들은 기존 방식 유지
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    if intent == "yield":
                        crawled_data = loop.run_until_complete(crawler.crawl_logs_data())
                    elif intent == "inventory":
                        crawled_data = loop.run_until_complete(crawler.crawl_inventory_data())
                    elif intent == "stack":
                        crawled_data = loop.run_until_complete(crawler.crawl_stacking_data())
                    
                    loop.close()
                
                if crawled_data:
                    print(f"✅ [{intent}] 데이터 크롤링 완료")
                else:
                    print(f"⚠️ [{intent}] 데이터 크롤링 결과 없음")
            except Exception as crawl_error:
                print(f"⚠️ [{intent}] 데이터 크롤링 오류: {crawl_error}")
                import traceback
                traceback.print_exc()
                crawled_data = None
        
        # 일반적인 질문은 Gemini에게 전달 (향상된 프롬프트 사용)
        enhanced_prompt, temperature = create_enhanced_prompt(user_input, intent, messages, crawled_data)
        response = get_gemini_response(enhanced_prompt, temperature=temperature)
        
        if not response or response.startswith("[오류]"):
            response = """
죄송합니다. 현재 일시적인 오류가 발생했습니다. 😅

🔄 **다시 시도해보시거나 아래 기능을 이용해보세요:**
- "기능" 입력 → 서비스 기능 안내
- "수율" 입력 → 수율 데이터 조회
- "재고" 입력 → 재고 현황 확인
- "적층" 입력 → HBM 적층 구조 정보
- "도움" 입력 → 전체 도움말 보기

💡 **HBM 제조 정보가 필요하시면:**
웨이퍼, TSV, 적층, 수율 등의 키워드를 입력해보세요!
            """
        
        return jsonify({
            "message": {
                "role": "assistant",
                "content": response
            }
        })
        
    except Exception as e:
        print(f"Chatbot API 오류: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "message": {
                "role": "assistant",
                "content": f"""
🚨 **서비스 일시 오류**

죄송합니다. 잠시 후 다시 시도해주세요.

📞 **지속적인 문제 발생시:**
- 이메일: support@stackvision.com
- 전화: 02-1234-5678

💡 **기본 도움말:**
"도움", "기능", "수율", "재고", "적층" 등을 입력해보세요!
                """
            }
        }), 500

@chatbot_bp.route("/health", methods=["GET"])
def health():
    """챗봇 헬스 체크"""
    try:
        from gemini_handler import READY, get_api_status
        
        status = get_api_status() if hasattr(get_api_status, '__call__') else {"ready": READY}
        
        return jsonify({
            "status": "healthy" if status.get("ready", False) else "degraded",
            "message": "챗봇 서비스가 정상 작동 중입니다.",
            "timestamp": datetime.now().isoformat(),
            "gemini_ready": status.get("ready", False)
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500


