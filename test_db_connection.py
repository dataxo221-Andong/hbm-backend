import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from db import get_conn

def test():
    print("Starting DB connection test...")
    try:
        conn = get_conn()
        if conn:
            print("SUCCESS: Connected to database successfully!")
            try:
                with conn.cursor() as cursor:
                    # Check current user and database
                    cursor.execute("SELECT USER(), DATABASE()")
                    result = cursor.fetchone()
                    print(f"INFO: {result}")
                conn.close()
            except Exception as e:
                print(f"WARNING: Error executing query: {e}")
        else:
            print("FAILURE: Could not connect to database.")
            print("Check your settings in db.py (host, user, password, database).")
            print("Ensure MySQL server is running.")
    except Exception as e:
        print(f"ERROR: An unexpected error occurred: {e}")

if __name__ == "__main__":
    test()
