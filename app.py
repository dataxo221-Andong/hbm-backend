from flask import Flask
from flask_cors import CORS
from flask_bcrypt import Bcrypt
from routes.auth import auth_bp
from dotenv import load_dotenv
import os
import socket
import sys
from builtins import print as _builtin_print

# Windows 콘솔(cp949 등)에서 이모지 출력 시 UnicodeEncodeError로 서버가 죽는 문제 방지
def _print_safe(*args, **kwargs):
    try:
        _builtin_print(*args, **kwargs)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "utf-8"
        sep = kwargs.get("sep", " ")
        end = kwargs.get("end", "\n")
        text = sep.join(str(a) for a in args) + end
        safe = text.encode(enc, errors="replace").decode(enc, errors="replace")
        _builtin_print(safe, end="")

print = _print_safe  # type: ignore

# 환경 변수 로드
load_dotenv()

app = Flask(__name__)

# Bcrypt 초기화 (앱 레벨에서 초기화)
bcrypt = Bcrypt(app)

# CORS 설정 - 개발 환경: 모든 origin 허용
# 프로덕션에서는 특정 origin만 허용하도록 수정 필요
CORS(app, resources={
    r"/*": {
        "origins": "*",  # 개발 환경: 모든 origin 허용
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": False  # "*" origin 사용 시 반드시 False (CORS 정책)
    }
})

# Blueprint 등록
app.register_blueprint(auth_bp)

from routes.wafer import wafer_bp
app.register_blueprint(wafer_bp)

from routes.stack import stack_bp
app.register_blueprint(stack_bp)

from routes.inventory import inventory_bp
app.register_blueprint(inventory_bp)

from routes.chatbot import chatbot_bp
app.register_blueprint(chatbot_bp)

from routes.crawler_routes import crawler_bp
app.register_blueprint(crawler_bp)

@app.route('/')
def health_check():
    return {"status": "ok", "message": "Backend server is running"}

@app.route('/health')
def health():
    return {"status": "ok"}, 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    
    # 네트워크 IP 주소 자동 감지
    def get_local_ip():
        try:
            # 외부 서버에 연결하여 로컬 IP 확인
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "192.168.37.20"  # 기본값 (사용자 Wi-Fi IP)
    
    local_ip = get_local_ip()
    
    print("=" * 50)
    print("🚀 Flask 서버 시작")
    print("=" * 50)
    print(f"📍 로컬 접속:")
    print(f"   http://localhost:{port}")
    print(f"   http://127.0.0.1:{port}")
    print(f"")
    print(f"🌐 네트워크 접속:")
    print(f"   http://{local_ip}:{port}")
    print(f"")
    print(f"✅ Health Check:")
    print(f"   http://{local_ip}:{port}/health")
    print("=" * 50)
    print(f"🔧 서버 설정:")
    print(f"   Host: 0.0.0.0 (모든 인터페이스에서 수신)")
    print(f"   Port: {port}")
    print(f"   Debug: True")
    print("=" * 50)
    
    # 더 명확한 서버 설정
    app.run(
        debug=True,
        host='0.0.0.0',  # 모든 네트워크 인터페이스에서 수신
        port=port,
        threaded=True,  # 멀티스레드 활성화
        use_reloader=False  # 네트워크 접근 시 리로더 비활성화 권장
    )

