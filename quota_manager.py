# -*- coding: utf-8 -*-
"""
Quota Manager for Multi-User Support
Handles API rate limits, storage quotas, and concurrent session limits
"""

import sqlite3
from datetime import datetime, date
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent / "users.db"


class QuotaManager:
    """Manages user quotas and limits"""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or str(DB_PATH)
    
    def get_quota(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get user quota information"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT quota_id, daily_requests, daily_cache_size_mb, "
                    "max_concurrent_sessions, reset_date, requests_used, cache_used_mb "
                    "FROM quotas WHERE user_id = ?",
                    (user_id,)
                )
                result = cursor.fetchone()
                
                if not result:
                    return None
                
                return {
                    'quota_id': result[0],
                    'daily_requests': result[1],
                    'daily_cache_size_mb': result[2],
                    'max_concurrent_sessions': result[3],
                    'reset_date': result[4],
                    'requests_used': result[5],
                    'cache_used_mb': result[6],
                    'requests_remaining': result[1] - result[5],
                    'cache_remaining_mb': result[2] - result[6]
                }
        except Exception as e:
            logger.error(f"Error getting quota: {e}")
            return None
    
    def reset_daily_quota(self, user_id: int) -> Tuple[bool, str]:
        """Reset daily request and cache usage counters"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE quotas SET requests_used = 0, cache_used_mb = 0, "
                    "reset_date = CURRENT_DATE WHERE user_id = ?",
                    (user_id,)
                )
                conn.commit()
                logger.info(f"Daily quota reset for user ID: {user_id}")
                return True, "Quota reset"
        except Exception as e:
            logger.error(f"Error resetting quota: {e}")
            return False, "Reset failed"
    
    def check_request_quota(self, user_id: int) -> Tuple[bool, str, int]:
        """
        Check if user can make another API request
        Returns: (allowed, message, requests_remaining)
        """
        try:
            quota = self.get_quota(user_id)
            if not quota:
                return False, "User not found", 0
            
            # Check if quota should be reset (new day)
            today = str(date.today())
            if quota['reset_date'] < today:
                self.reset_daily_quota(user_id)
                quota = self.get_quota(user_id)
            
            if quota['requests_used'] >= quota['daily_requests']:
                return False, f"Daily request limit ({quota['daily_requests']}) exceeded", 0
            
            remaining = quota['requests_remaining']
            return True, "Request allowed", remaining
        
        except Exception as e:
            logger.error(f"Error checking request quota: {e}")
            return False, "Quota check failed", 0
    
    def check_cache_quota(self, user_id: int, needed_mb: float) -> Tuple[bool, str, float]:
        """
        Check if user has cache space available
        Returns: (allowed, message, space_remaining_mb)
        """
        try:
            quota = self.get_quota(user_id)
            if not quota:
                return False, "User not found", 0
            
            # Check if quota should be reset (new day)
            today = str(date.today())
            if quota['reset_date'] < today:
                self.reset_daily_quota(user_id)
                quota = self.get_quota(user_id)
            
            if quota['cache_used_mb'] + needed_mb > quota['daily_cache_size_mb']:
                return False, f"Cache limit ({quota['daily_cache_size_mb']}MB) would be exceeded", quota['cache_remaining_mb']
            
            return True, "Cache space available", quota['cache_remaining_mb'] - needed_mb
        
        except Exception as e:
            logger.error(f"Error checking cache quota: {e}")
            return False, "Quota check failed", 0
    
    def check_concurrent_sessions(self, user_id: int) -> Tuple[bool, str, int]:
        """
        Check concurrent session limit
        Returns: (allowed, message, active_sessions)
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get max concurrent sessions
                cursor.execute(
                    "SELECT max_concurrent_sessions FROM quotas WHERE user_id = ?",
                    (user_id,)
                )
                result = cursor.fetchone()
                if not result:
                    return False, "User not found", 0
                max_sessions = result[0]
                
                # Count active sessions (not expired)
                cursor.execute(
                    "SELECT COUNT(*) FROM sessions "
                    "WHERE user_id = ? AND expires_at > CURRENT_TIMESTAMP",
                    (user_id,)
                )
                active = cursor.fetchone()[0]
                
                if active >= max_sessions:
                    return False, f"Max concurrent sessions ({max_sessions}) reached", active
                
                return True, "Session limit OK", active
        
        except Exception as e:
            logger.error(f"Error checking concurrent sessions: {e}")
            return False, "Session check failed", 0
    
    def log_request(self, user_id: int, session_id: str, ticker: str, 
                   action: str, duration_ms: float, success: bool, 
                   error_msg: str = None) -> Tuple[bool, str]:
        """Log request for quota tracking"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute(
                    "INSERT INTO request_logs "
                    "(user_id, session_id, ticker, action, request_duration_ms, success, error_message) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (user_id, session_id, ticker, action, duration_ms, success, error_msg)
                )
                
                # Increment request counter
                cursor.execute(
                    "UPDATE quotas SET requests_used = requests_used + 1 "
                    "WHERE user_id = ?",
                    (user_id,)
                )
                
                conn.commit()
                return True, "Request logged"
        
        except Exception as e:
            logger.error(f"Error logging request: {e}")
            return False, "Log failed"
    
    def update_cache_usage(self, user_id: int, delta_mb: float) -> Tuple[bool, str]:
        """Update cache usage (delta can be positive or negative)"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE quotas SET cache_used_mb = cache_used_mb + ? "
                    "WHERE user_id = ?",
                    (delta_mb, user_id)
                )
                conn.commit()
                return True, "Cache usage updated"
        
        except Exception as e:
            logger.error(f"Error updating cache usage: {e}")
            return False, "Update failed"
    
    def get_user_stats(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get comprehensive user statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get quota
                cursor.execute(
                    "SELECT daily_requests, daily_cache_size_mb, max_concurrent_sessions, "
                    "requests_used, cache_used_mb FROM quotas WHERE user_id = ?",
                    (user_id,)
                )
                quota = cursor.fetchone()
                
                # Get active sessions
                cursor.execute(
                    "SELECT COUNT(*) FROM sessions "
                    "WHERE user_id = ? AND expires_at > CURRENT_TIMESTAMP",
                    (user_id,)
                )
                active_sessions = cursor.fetchone()[0]
                
                # Get request stats from today
                cursor.execute(
                    "SELECT COUNT(*), AVG(request_duration_ms), SUM(CASE WHEN success THEN 1 ELSE 0 END) "
                    "FROM request_logs WHERE user_id = ? AND DATE(timestamp) = CURRENT_DATE",
                    (user_id,)
                )
                req_result = cursor.fetchone()
                
                if not quota:
                    return None
                
                return {
                    'daily_limit': quota[0],
                    'daily_used': quota[3],
                    'daily_remaining': quota[0] - quota[3],
                    'cache_limit_mb': quota[1],
                    'cache_used_mb': quota[2],
                    'cache_remaining_mb': quota[1] - quota[2],
                    'concurrent_limit': quota[2],
                    'concurrent_active': active_sessions,
                    'today_requests': req_result[0] if req_result[0] else 0,
                    'avg_request_ms': req_result[1] if req_result[1] else 0,
                    'successful_requests': req_result[2] if req_result[2] else 0
                }
        
        except Exception as e:
            logger.error(f"Error getting user stats: {e}")
            return None
    
    def set_custom_quota(self, user_id: int, daily_requests: int = None, 
                        daily_cache_size_mb: int = None, 
                        max_concurrent_sessions: int = None) -> Tuple[bool, str]:
        """Set custom quota limits for a user (admin function)"""
        try:
            updates = []
            params = []
            
            if daily_requests is not None:
                updates.append("daily_requests = ?")
                params.append(daily_requests)
            if daily_cache_size_mb is not None:
                updates.append("daily_cache_size_mb = ?")
                params.append(daily_cache_size_mb)
            if max_concurrent_sessions is not None:
                updates.append("max_concurrent_sessions = ?")
                params.append(max_concurrent_sessions)
            
            if not updates:
                return False, "No updates specified"
            
            params.append(user_id)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                query = f"UPDATE quotas SET {', '.join(updates)} WHERE user_id = ?"
                cursor.execute(query, params)
                conn.commit()
                logger.info(f"Custom quota set for user ID: {user_id}")
                return True, "Quota updated"
        
        except Exception as e:
            logger.error(f"Error setting custom quota: {e}")
            return False, "Update failed"


# Global quota manager instance
quota_manager = QuotaManager()
