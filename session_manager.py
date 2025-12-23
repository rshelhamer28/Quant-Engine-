# -*- coding: utf-8 -*-
"""
Session Manager for Database-Backed Sessions
Replaces Streamlit's browser-based session state with persistent server-side sessions
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent / "users.db"


class SessionManager:
    """Manages persistent server-side user sessions"""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or str(DB_PATH)
        self.init_session_storage()
    
    def init_session_storage(self):
        """Initialize session storage table"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS session_storage (
                        storage_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER NOT NULL,
                        session_id TEXT NOT NULL,
                        key TEXT NOT NULL,
                        value TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (user_id) REFERENCES users(user_id),
                        UNIQUE(session_id, key)
                    )
                ''')
                conn.commit()
                logger.info("Session storage initialized")
        except Exception as e:
            logger.error(f"Session storage init error: {e}")
    
    def set_session_value(self, user_id: int, session_id: str, key: str, 
                         value: Any) -> Tuple[bool, str]:
        """Store a value in session (replaces st.session_state[key] = value)"""
        try:
            # Convert to JSON if not string
            if not isinstance(value, str):
                value = json.dumps(value)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Try update first
                cursor.execute(
                    "UPDATE session_storage SET value = ?, updated_at = CURRENT_TIMESTAMP "
                    "WHERE session_id = ? AND key = ?",
                    (value, session_id, key)
                )
                
                # If no rows updated, insert
                if cursor.rowcount == 0:
                    cursor.execute(
                        "INSERT INTO session_storage (user_id, session_id, key, value) "
                        "VALUES (?, ?, ?, ?)",
                        (user_id, session_id, key, value)
                    )
                
                conn.commit()
                return True, f"Stored {key}"
        
        except Exception as e:
            logger.error(f"Error storing session value: {e}")
            return False, f"Storage failed: {e}"
    
    def get_session_value(self, session_id: str, key: str, default: Any = None) -> Any:
        """Retrieve a value from session (replaces st.session_state[key])"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT value FROM session_storage WHERE session_id = ? AND key = ?",
                    (session_id, key)
                )
                result = cursor.fetchone()
                
                if not result:
                    return default
                
                value = result[0]
                
                # Try to parse JSON
                try:
                    return json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    return value
        
        except Exception as e:
            logger.error(f"Error retrieving session value: {e}")
            return default
    
    def get_all_session_values(self, session_id: str) -> Dict[str, Any]:
        """Get all values for a session (replaces st.session_state dict)"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT key, value FROM session_storage WHERE session_id = ?",
                    (session_id,)
                )
                results = cursor.fetchall()
                
                data = {}
                for key, value in results:
                    try:
                        data[key] = json.loads(value)
                    except (json.JSONDecodeError, TypeError):
                        data[key] = value
                
                return data
        
        except Exception as e:
            logger.error(f"Error getting session values: {e}")
            return {}
    
    def delete_session_value(self, session_id: str, key: str) -> Tuple[bool, str]:
        """Delete a value from session (replaces del st.session_state[key])"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "DELETE FROM session_storage WHERE session_id = ? AND key = ?",
                    (session_id, key)
                )
                conn.commit()
                return True, f"Deleted {key}"
        
        except Exception as e:
            logger.error(f"Error deleting session value: {e}")
            return False, f"Delete failed: {e}"
    
    def clear_session(self, session_id: str) -> Tuple[bool, str]:
        """Clear all values for a session"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "DELETE FROM session_storage WHERE session_id = ?",
                    (session_id,)
                )
                conn.commit()
                logger.info(f"Session cleared: {session_id[:8]}...")
                return True, "Session cleared"
        
        except Exception as e:
            logger.error(f"Error clearing session: {e}")
            return False, f"Clear failed: {e}"
    
    def session_exists(self, session_id: str, key: str) -> bool:
        """Check if key exists in session"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT 1 FROM session_storage WHERE session_id = ? AND key = ? LIMIT 1",
                    (session_id, key)
                )
                return cursor.fetchone() is not None
        except Exception as e:
            logger.error(f"Error checking session key: {e}")
            return False
    
    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """Get session statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Count items
                cursor.execute(
                    "SELECT COUNT(*) FROM session_storage WHERE session_id = ?",
                    (session_id,)
                )
                item_count = cursor.fetchone()[0]
                
                # Get storage size
                cursor.execute(
                    "SELECT SUM(LENGTH(value)) FROM session_storage WHERE session_id = ?",
                    (session_id,)
                )
                size_bytes = cursor.fetchone()[0] or 0
                
                # Get creation and update times
                cursor.execute(
                    "SELECT MIN(created_at), MAX(updated_at) FROM session_storage WHERE session_id = ?",
                    (session_id,)
                )
                times = cursor.fetchone()
                
                return {
                    'items': item_count,
                    'size_bytes': size_bytes,
                    'size_mb': round(size_bytes / 1024 / 1024, 2),
                    'created_at': times[0] if times[0] else None,
                    'last_updated': times[1] if times[1] else None
                }
        
        except Exception as e:
            logger.error(f"Error getting session stats: {e}")
            return {'items': 0, 'size_bytes': 0, 'size_mb': 0}
    
    def cleanup_old_sessions(self, user_id: int = None, days: int = 7) -> Tuple[int, str]:
        """Clean up old session data (admin function)"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Delete storage for old expired sessions
                query = (
                    "DELETE FROM session_storage WHERE session_id IN "
                    "(SELECT session_id FROM sessions WHERE expires_at < CURRENT_TIMESTAMP"
                )
                
                if user_id:
                    query += f" AND user_id = {user_id}"
                
                query += ")"
                
                cursor.execute(query)
                deleted = cursor.rowcount
                
                conn.commit()
                logger.info(f"Cleaned up {deleted} expired session records")
                return deleted, f"Deleted {deleted} old session records"
        
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
            return 0, f"Cleanup failed: {e}"
    
    def export_session_data(self, session_id: str) -> Dict[str, Any]:
        """Export all session data (for backup/analysis)"""
        try:
            data = self.get_all_session_values(session_id)
            stats = self.get_session_stats(session_id)
            
            return {
                'session_id': session_id,
                'data': data,
                'stats': stats,
                'exported_at': datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Export error: {e}")
            return {}


# Global session manager instance
session_manager = SessionManager()
