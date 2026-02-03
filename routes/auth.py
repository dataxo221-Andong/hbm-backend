from flask import Blueprint, request, jsonify, current_app
from flask_bcrypt import Bcrypt
import sys
import os

# 프로젝트 루트를 sys.path에 추가 (app.py가 실행되는 위치)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from db import get_conn
import pymysql
import jwt
import datetime

auth_bp = Blueprint(
    "auth",
    __name__,
    url_prefix="/auth"
)

# JWT 시크릿 키 (환경 변수로 관리)
JWT_SECRET = os.environ.get('JWT_SECRET', 'your-super-secret-jwt-key-change-this-in-production')
JWT_EXPIRATION_HOURS = 168  # 7일


# --------------------
# 공통 유틸 함수
# --------------------
def get_json_data():
    if not request.is_json:
        return None, jsonify({"error": "JSON 데이터가 없습니다."}), 400
    data = request.get_json()
    if data is None:
        return None, jsonify({"error": "JSON 데이터가 없습니다."}), 400
    return data, None, None


def generate_token(user_id, email):
    """JWT 토큰 생성"""
    payload = {
        "userId": str(user_id),
        "email": email,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=JWT_EXPIRATION_HOURS),
        "iat": datetime.datetime.utcnow()
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")


def validate_signup_input(data):
    """회원가입 입력 검증"""
    name = data.get("name")
    email = data.get("email")
    password = data.get("password")
    confirmPassword = data.get("confirmPassword")

    if not name or not email or not password or not confirmPassword:
        return None, None, None, None, jsonify({"error": "모든 필드를 입력해주세요."}), 400

    # 문자열 타입 확인
    if not isinstance(name, str) or not isinstance(email, str) or not isinstance(password, str) or not isinstance(confirmPassword, str):
        return None, None, None, None, jsonify({"error": "모든 필드는 문자열이어야 합니다."}), 400

    name = name.strip()
    email = email.strip().lower()
    password = password.strip()
    confirmPassword = confirmPassword.strip()

    if not name or not email or not password or not confirmPassword:
        return None, None, None, None, jsonify({"error": "모든 필드는 빈 값일 수 없습니다."}), 400

    # 비밀번호 확인
    if password != confirmPassword:
        return None, None, None, None, jsonify({"error": "비밀번호가 일치하지 않습니다."}), 400

    # 비밀번호 길이 검증
    if len(password) < 6:
        return None, None, None, None, jsonify({"error": "비밀번호는 최소 6자 이상이어야 합니다."}), 400

    # 이메일 형식 검증
    if "@" not in email or "." not in email.split("@")[1]:
        return None, None, None, None, jsonify({"error": "올바른 이메일 형식이 아닙니다."}), 400

    return name, email, password, confirmPassword, None, None


def validate_login_input(data):
    """로그인 입력 검증"""
    email = data.get("email")
    password = data.get("password")

    if not email or not password:
        return None, None, jsonify({"error": "이메일과 비밀번호를 입력해주세요."}), 400

    # 문자열 타입 확인
    if not isinstance(email, str) or not isinstance(password, str):
        return None, None, jsonify({"error": "이메일과 비밀번호는 문자열이어야 합니다."}), 400

    email = email.strip().lower()
    password = password.strip()

    if not email or not password:
        return None, None, jsonify({"error": "이메일과 비밀번호는 빈 값일 수 없습니다."}), 400

    return email, password, None, None


# --------------------
# 회원가입
# --------------------
@auth_bp.route("/register", methods=["POST"])
def register():
    data, error_res, status = get_json_data()
    if error_res:
        return error_res, status

    name, email, password, confirmPassword, error_res, status = validate_signup_input(data)
    if error_res:
        return error_res, status

    # Bcrypt를 통해 비밀번호 해싱 (app.py에서 초기화된 인스턴스 사용)
    try:
        bcrypt = Bcrypt(current_app)
        hashed_pw = bcrypt.generate_password_hash(password).decode('utf-8')
    except Exception as e:
        print(f"비밀번호 해싱 에러: {str(e)}")
        return jsonify({"error": f"비밀번호 해싱 실패: {str(e)}"}), 500

    conn = get_conn()
    if conn is None:
        return jsonify({"error": "데이터베이스 연결 실패"}), 500

    cur = conn.cursor()
    try:
        # 이메일 중복 확인
        cur.execute(
            "SELECT id FROM user_auth WHERE email = %s",
            (email,)
        )
        existing_user = cur.fetchone()
        if existing_user:
            return jsonify({"error": "이미 사용 중인 이메일입니다."}), 409

        # 사용자 생성
        cur.execute(
            "INSERT INTO user_auth (email, password, name) VALUES (%s, %s, %s)",
            (email, hashed_pw, name)
        )
        conn.commit()
        
        # 생성된 사용자 정보 가져오기
        user_id = cur.lastrowid
        cur.execute(
            "SELECT id, name, email FROM user_auth WHERE id = %s",
            (user_id,)
        )
        new_user = cur.fetchone()

        # JWT 토큰 생성
        token = generate_token(user_id, email)

        # 비밀번호 제외하고 응답
        user_response = {
            "id": new_user["id"],
            "name": new_user.get("name"),
            "email": new_user.get("email"),
        }

        return jsonify({
            "message": "회원가입이 완료되었습니다.",
            "user": user_response,
            "token": token
        }), 201

    except (pymysql.err.IntegrityError, pymysql.IntegrityError) as e:
        conn.rollback()
        # 에러 코드 1062는 중복 키 에러
        if hasattr(e, 'args') and len(e.args) > 0 and e.args[0] == 1062:
            return jsonify({"error": "이미 사용 중인 이메일입니다."}), 409
        return jsonify({"error": "이미 사용 중인 이메일입니다."}), 409

    except Exception as e:
        conn.rollback()
        print(f"회원가입 에러: {str(e)}")
        return jsonify({"error": f"회원가입 실패: {str(e)}"}), 500

    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()


# --------------------
# 로그인
# --------------------
@auth_bp.route("/login", methods=["POST"])
def login():
    data, error_res, status = get_json_data()
    if error_res:
        return error_res, status

    email, password, error_res, status = validate_login_input(data)
    if error_res:
        return error_res, status

    conn = get_conn()
    if conn is None:
        return jsonify({"error": "데이터베이스 연결 실패"}), 500

    cur = conn.cursor()
    try:
        # email로 사용자 찾기
        cur.execute(
            "SELECT id, password, name, email FROM user_auth WHERE email = %s",
            (email,)
        )
        result = cur.fetchone()

        if not result:
            return jsonify({"error": "이메일 또는 비밀번호가 올바르지 않습니다."}), 401

        stored_pw = result["password"]

        # Bcrypt를 통해 비밀번호 확인 (app.py에서 초기화된 인스턴스 사용)
        password_valid = False
        try:
            bcrypt = Bcrypt(current_app)
            
            # 저장된 비밀번호가 bcrypt 형식인지 확인 ($2b$ 또는 $2a$로 시작)
            is_bcrypt_hash = stored_pw.startswith('$2b$') or stored_pw.startswith('$2a$') or stored_pw.startswith('$2y$')
            
            if is_bcrypt_hash:
                # bcrypt 해시인 경우
                password_valid = bcrypt.check_password_hash(stored_pw, password)
            else:
                # 기존 werkzeug 해시인 경우 (마이그레이션 필요)
                from werkzeug.security import check_password_hash as werkzeug_check
                password_valid = werkzeug_check(stored_pw, password)
                # 로그인 성공 시 bcrypt로 재해싱하여 업데이트
                if password_valid:
                    try:
                        new_hash = bcrypt.generate_password_hash(password).decode('utf-8')
                        cur.execute(
                            "UPDATE user_auth SET password = %s WHERE id = %s",
                            (new_hash, result["id"])
                        )
                        conn.commit()
                        print(f"사용자 {result['id']}의 비밀번호를 bcrypt로 마이그레이션했습니다.")
                    except Exception as e:
                        print(f"비밀번호 마이그레이션 실패: {str(e)}")
                        # 마이그레이션 실패해도 로그인은 계속 진행
        except Exception as e:
            print(f"비밀번호 확인 에러: {str(e)}")
            return jsonify({"error": f"비밀번호 확인 실패: {str(e)}"}), 500
        
        if password_valid:
            # JWT 토큰 생성
            user_email = result.get("email")
            token = generate_token(result["id"], user_email)

            # 비밀번호 제외하고 응답
            user_response = {
                "id": result["id"],
                "name": result.get("name"),
                "email": user_email,
            }

            return jsonify({
                "message": "로그인 성공",
                "user": user_response,
                "token": token
            }), 200
        else:
            return jsonify({"error": "이메일 또는 비밀번호가 올바르지 않습니다."}), 401

    except Exception as e:
        print(f"로그인 에러: {str(e)}")
        return jsonify({"error": f"로그인 실패: {str(e)}"}), 500

    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

