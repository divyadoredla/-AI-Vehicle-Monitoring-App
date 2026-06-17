import sqlite3
import hashlib
import os
import tempfile

# Use /tmp for cloud deployments (Streamlit Cloud, Hugging Face, etc.)
# Falls back to local path when running locally
_LOCAL_DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "users.db")
_CLOUD_DB = os.path.join(tempfile.gettempdir(), "ai_vehicle_users.db")

def _get_db_path():
    """Return writable DB path — local dir if writable, else /tmp."""
    local_dir = os.path.dirname(os.path.abspath(__file__))
    if os.access(local_dir, os.W_OK):
        return _LOCAL_DB
    return _CLOUD_DB

DB_NAME = _get_db_path()

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (username TEXT PRIMARY KEY, password TEXT)''')
    conn.commit()
    conn.close()
    # Seed a default admin account on first run
    add_user("admin", "admin123")

def hash_password(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def add_user(username, password):
    init_db_table()
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users VALUES (?,?)", (username, hash_password(password)))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def init_db_table():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (username TEXT PRIMARY KEY, password TEXT)''')
    conn.commit()
    conn.close()

def check_user(username, password):
    init_db_table()
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username=? AND password=?",
              (username, hash_password(password)))
    user = c.fetchone()
    conn.close()
    return user is not None

if __name__ == "__main__":
    init_db()
    print(f"Database initialized at: {DB_NAME}")
    if add_user("admin", "admin123"):
        print("Default user 'admin' created with password 'admin123'.")
    else:
        print("Default user already exists.")
