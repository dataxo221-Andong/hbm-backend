import pymysql

def get_conn():
    """
    데이터베이스 연결을 반환합니다.
    """
    try:
        conn = pymysql.connect(
            host="52.79.35.75",
            user="mixup",
            password="6404",
            database="project",
            charset="utf8mb4",
            cursorclass=pymysql.cursors.DictCursor,
            autocommit=False
        )
        return conn
    except Exception as e:
        print(f"데이터베이스 연결 실패: {str(e)}")
        return None



