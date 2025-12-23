# -*- coding: utf-8 -*-
"""
Authentication Manager for Multi-User Support
Handles user login, password hashing, session management, and quotas
"""

import sqlite3
import hashlib
import secrets
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Database path
DB_PATH = Path(__file__).parent / "users.db"


class AuthManager:
    """Manages user authentication and sessions"""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or str(DB_PATH)
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database with required tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Users table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS users (
                        user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT UNIQUE NOT NULL,
                        email TEXT UNIQUE NOT NULL,
                        password_hash TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_login TIMESTAMP,
                        is_active BOOLEAN DEFAULT 1
                    )
                ''')
                
                # Sessions table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id TEXT PRIMARY KEY,
                        user_id INTEGER NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        expires_at TIMESTAMP NOT NULL,
                        ip_address TEXT,
                        user_agent TEXT,
                        FOREIGN KEY (user_id) REFERENCES users(user_id)
                    )
                ''')
                
                # Quotas table (daily limits)
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS quotas (
                        quota_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER NOT NULL UNIQUE,
                        daily_requests INTEGER DEFAULT 500,
                        daily_cache_size_mb INTEGER DEFAULT 100,
                        max_concurrent_sessions INTEGER DEFAULT 3,
                        reset_date DATE DEFAULT CURRENT_DATE,
                        requests_used INTEGER DEFAULT 0,
                        cache_used_mb REAL DEFAULT 0,
                        FOREIGN KEY (user_id) REFERENCES users(user_id)
                    )
                ''')
                
                # Request logs table (for tracking quotas)
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS request_logs (
                        log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER NOT NULL,
                        session_id TEXT NOT NULL,
                        ticker TEXT,
                        action TEXT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        request_duration_ms REAL,
                        success BOOLEAN,
                        error_message TEXT,
                        FOREIGN KEY (user_id) REFERENCES users(user_id)
                    )
                ''')
                
                conn.commit()
                logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
            raise
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash password using SHA256 + salt"""
        salt = secrets.token_hex(32)
        pwd_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
        return f"{salt}${pwd_hash.hex()}"
    
    @staticmethod
    def verify_password(password: str, password_hash: str) -> bool:
        """Verify password against hash"""
        try:
            salt, pwd_hash = password_hash.split('$')
            new_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
            return new_hash.hex() == pwd_hash
        except Exception as e:
            logger.error(f"Password verification error: {e}")
            return False
    
    def register_user(self, username: str, email: str, password: str) -> Tuple[bool, str]:
        """Register a new user"""
        try:
            # Validate inputs
            if not username or len(username) < 3:
                return False, "Username must be at least 3 characters"
            if not email or '@' not in email:
                return False, "Invalid email address"
            if not password or len(password) < 8:
                return False, "Password must be at least 8 characters"
            
            password_hash = self.hash_password(password)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if user exists
                cursor.execute("SELECT user_id FROM users WHERE username = ? OR email = ?", 
                             (username, email))
                if cursor.fetchone():
                    return False, "Username or email already exists"
                
                # Create user
                cursor.execute(
                    "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
                    (username, email, password_hash)
                )
                user_id = cursor.lastrowid
                
                # Create default quota
                cursor.execute(
                    "INSERT INTO quotas (user_id) VALUES (?)",
                    (user_id,)
                )
                
                conn.commit()
                logger.info(f"User registered: {username} (ID: {user_id})")
                return True, f"User {username} registered successfully"
        
        except sqlite3.IntegrityError as e:
            logger.error(f"Registration error: {e}")
            return False, "Registration failed - duplicate username or email"
        except Exception as e:
            logger.error(f"Registration error: {e}")
            return False, "Registration failed"
    
    def login(self, username: str, password: str, ip_address: str = "", 
              user_agent: str = "") -> Tuple[Optional[str], Optional[int], str]:
        """
        Login user and create session
        Returns: (session_id, user_id, message)
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get user
                cursor.execute(
                    "SELECT user_id, password_hash, is_active FROM users WHERE username = ?",
                    (username,)
                )
                result = cursor.fetchone()
                
                if not result:
                    return None, None, "Invalid username or password"
                
                user_id, password_hash, is_active = result
                
                if not is_active:
                    return None, None, "Account is inactive"
                
                if not self.verify_password(password, password_hash):
                    return None, None, "Invalid username or password"
                
                # Create session
                session_id = secrets.token_urlsafe(32)
                expires_at = datetime.now() + timedelta(days=7)
                
                cursor.execute(
                    "INSERT INTO sessions (session_id, user_id, expires_at, ip_address, user_agent) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (session_id, user_id, expires_at, ip_address, user_agent)
                )
                
                # Update last login
                cursor.execute(
                    "UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE user_id = ?",
                    (user_id,)
                )
                
                conn.commit()
                logger.info(f"User logged in: {username} (ID: {user_id}, Session: {session_id[:8]}...)")
                return session_id, user_id, "Login successful"
        
        except Exception as e:
            logger.error(f"Login error: {e}")
            return None, None, "Login failed"
    
    def validate_session(self, session_id: str) -> Tuple[bool, Optional[int], str]:
        """
        Validate session token
        Returns: (is_valid, user_id, message)
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute(
                    "SELECT user_id, expires_at FROM sessions WHERE session_id = ?",
                    (session_id,)
                )
                result = cursor.fetchone()
                
                if not result:
                    return False, None, "Invalid session"
                
                user_id, expires_at_str = result
                expires_at = datetime.fromisoformat(expires_at_str)
                
                if datetime.now() > expires_at:
                    # Delete expired session
                    cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
                    conn.commit()
                    return False, None, "Session expired"
                
                # Update last activity
                cursor.execute(
                    "UPDATE sessions SET last_activity = CURRENT_TIMESTAMP WHERE session_id = ?",
                    (session_id,)
                )
                conn.commit()
                
                return True, user_id, "Session valid"
        
        except Exception as e:
            logger.error(f"Session validation error: {e}")
            return False, None, "Session validation failed"
    
    def logout(self, session_id: str) -> Tuple[bool, str]:
        """Logout user by invalidating session"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
                conn.commit()
                logger.info(f"User logged out: Session {session_id[:8]}...")
                return True, "Logout successful"
        except Exception as e:
            logger.error(f"Logout error: {e}")
            return False, "Logout failed"
    
    def get_user_info(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get user information"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT user_id, username, email, created_at, last_login, is_active "
                    "FROM users WHERE user_id = ?",
                    (user_id,)
                )
                result = cursor.fetchone()
                
                if not result:
                    return None
                
                return {
                    'user_id': result[0],
                    'username': result[1],
                    'email': result[2],
                    'created_at': result[3],
                    'last_login': result[4],
                    'is_active': result[5]
                }
        except Exception as e:
            logger.error(f"Error getting user info: {e}")
            return None
    
    def change_password(self, user_id: int, old_password: str, new_password: str) -> Tuple[bool, str]:
        """Change user password"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get current password hash
                cursor.execute("SELECT password_hash FROM users WHERE user_id = ?", (user_id,))
                result = cursor.fetchone()
                
                if not result:
                    return False, "User not found"
                
                if not self.verify_password(old_password, result[0]):
                    return False, "Current password is incorrect"
                
                if len(new_password) < 8:
                    return False, "New password must be at least 8 characters"
                
                new_hash = self.hash_password(new_password)
                cursor.execute(
                    "UPDATE users SET password_hash = ? WHERE user_id = ?",
                    (new_hash, user_id)
                )
                conn.commit()
                logger.info(f"Password changed for user ID: {user_id}")
                return True, "Password changed successfully"
        
        except Exception as e:
            logger.error(f"Password change error: {e}")
            return False, "Password change failed"


# Global auth manager instance
auth_manager = AuthManager()
