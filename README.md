# HBM Backend API

Flask 기반 백엔드 API 서버입니다.

## 설치 방법

1. Python 가상환경 생성 및 활성화:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

2. 패키지 설치:
```bash
pip install -r requirements.txt
```

3. 환경 변수 설정:
```bash
# .env.example을 .env로 복사하고 값 수정
cp .env.example .env
```

4. 데이터베이스 설정:
- `project` 데이터베이스의 `user_auth` 테이블이 이미 생성되어 있어야 합니다 (아래 SQL 참고)

## 데이터베이스 스키마

```sql
USE project;

CREATE TABLE IF NOT EXISTS user_auth (
    id INT AUTO_INCREMENT PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL,
    name VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_email (email)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
```

**참고:** 데이터베이스는 이미 생성되어 있다고 가정합니다.

## 서버 실행

```bash
python app.py
```

서버는 기본적으로 `http://localhost:5000`에서 실행됩니다.

## API 엔드포인트

### 회원가입
- **POST** `/auth/register`
- Request Body:
```json
{
  "name": "홍길동",
  "email": "user@example.com",
  "password": "password123",
  "confirmPassword": "password123"
}
```
- Response:
```json
{
  "message": "회원가입이 완료되었습니다.",
  "user": {
    "id": 1,
    "name": "홍길동",
    "email": "user@example.com"
  },
  "token": "jwt_token_here"
}
```

### 로그인
- **POST** `/auth/login`
- Request Body:
```json
{
  "email": "user@example.com",
  "password": "password123"
}
```
- Response:
```json
{
  "message": "로그인 성공",
  "user": {
    "id": 1,
    "name": "홍길동",
    "email": "user@example.com"
  },
  "token": "jwt_token_here"
}
```

## API 엔드포인트 주소

- 개발 환경: `http://localhost:5000`
- 프로덕션: 실제 배포된 서버 주소

**참고:** 프론트엔드 연동은 `lib` 폴더의 API 유틸리티를 사용하세요.






