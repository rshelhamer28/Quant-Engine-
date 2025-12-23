# -*- coding: utf-8 -*-
"""
Performance Monitoring & Alerting System
Tracks application metrics and generates alerts
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import sqlite3
from pathlib import Path
import json

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent / "users.db"


class PerformanceMonitor:
    """Monitors application performance metrics"""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or str(DB_PATH)
        self.init_metrics_table()
        
        # Alert thresholds
        self.THRESHOLDS = {
            'error_rate': 0.01,  # 1% error rate
            'response_time_p95': 5.0,  # 5 seconds
            'response_time_p99': 10.0,  # 10 seconds
            'memory_per_user': 100,  # MB
            'database_query_time': 0.05,  # 50ms
        }
        
        self.alerts: List[Dict] = []
    
    def init_metrics_table(self):
        """Initialize metrics storage table"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS performance_metrics (
                        metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        metric_type TEXT NOT NULL,
                        metric_value REAL NOT NULL,
                        metric_label TEXT,
                        details TEXT
                    )
                ''')
                
                # Create alerts table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS performance_alerts (
                        alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        alert_type TEXT NOT NULL,
                        severity TEXT NOT NULL,
                        message TEXT NOT NULL,
                        threshold REAL,
                        actual_value REAL,
                        acknowledged INTEGER DEFAULT 0
                    )
                ''')
                
                conn.commit()
                logger.info("Metrics table initialized")
        except Exception as e:
            logger.error(f"Metrics table init error: {e}")
    
    def record_metric(self, metric_type: str, value: float, label: str = None, details: str = None):
        """Record a performance metric"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO performance_metrics 
                    (metric_type, metric_value, metric_label, details)
                    VALUES (?, ?, ?, ?)
                ''', (metric_type, value, label, details))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to record metric: {e}")
    
    def check_alert_threshold(self, metric_type: str, value: float, threshold: float = None) -> bool:
        """
        Check if a metric exceeds its threshold and create alert if needed
        
        Returns:
            True if alert was created, False otherwise
        """
        if threshold is None:
            threshold = self.THRESHOLDS.get(metric_type)
        
        if threshold is None:
            return False
        
        if value > threshold:
            self.create_alert(
                alert_type=metric_type,
                severity="WARNING" if value < threshold * 1.5 else "CRITICAL",
                message=f"{metric_type} exceeded threshold",
                threshold=threshold,
                actual_value=value
            )
            return True
        
        return False
    
    def create_alert(self, alert_type: str, severity: str, message: str, 
                    threshold: float = None, actual_value: float = None):
        """Create a performance alert"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO performance_alerts 
                    (alert_type, severity, message, threshold, actual_value)
                    VALUES (?, ?, ?, ?, ?)
                ''', (alert_type, severity, message, threshold, actual_value))
                conn.commit()
            
            alert = {
                'type': alert_type,
                'severity': severity,
                'message': message,
                'timestamp': datetime.now().isoformat()
            }
            self.alerts.append(alert)
            logger.warning(f"[ALERT] {severity}: {message}")
        
        except Exception as e:
            logger.error(f"Failed to create alert: {e}")
    
    def get_recent_alerts(self, hours: int = 24, acknowledged: bool = False) -> List[Dict]:
        """Get recent alerts"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                time_cutoff = datetime.now() - timedelta(hours=hours)
                
                cursor.execute('''
                    SELECT alert_type, severity, message, threshold, actual_value, timestamp
                    FROM performance_alerts
                    WHERE timestamp > ? AND acknowledged = ?
                    ORDER BY timestamp DESC
                ''', (time_cutoff.isoformat(), 1 if acknowledged else 0))
                
                alerts = cursor.fetchall()
                return [
                    {
                        'type': row[0],
                        'severity': row[1],
                        'message': row[2],
                        'threshold': row[3],
                        'actual_value': row[4],
                        'timestamp': row[5]
                    }
                    for row in alerts
                ]
        except Exception as e:
            logger.error(f"Failed to get alerts: {e}")
            return []
    
    def acknowledge_alert(self, alert_type: str):
        """Mark alerts as acknowledged"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE performance_alerts
                    SET acknowledged = 1
                    WHERE alert_type = ? AND acknowledged = 0
                ''', (alert_type,))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to acknowledge alert: {e}")
    
    def get_metrics_summary(self, hours: int = 1) -> Dict:
        """Get summary of metrics from recent period"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                time_cutoff = datetime.now() - timedelta(hours=hours)
                
                cursor.execute('''
                    SELECT metric_type, COUNT(*), AVG(metric_value), MIN(metric_value), MAX(metric_value)
                    FROM performance_metrics
                    WHERE timestamp > ?
                    GROUP BY metric_type
                ''', (time_cutoff.isoformat(),))
                
                summary = {}
                for row in cursor.fetchall():
                    summary[row[0]] = {
                        'count': row[1],
                        'avg': round(row[2], 3),
                        'min': round(row[3], 3),
                        'max': round(row[4], 3)
                    }
                
                return summary
        except Exception as e:
            logger.error(f"Failed to get metrics summary: {e}")
            return {}
    
    def cleanup_old_metrics(self, days: int = 30):
        """Delete metrics older than specified days"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cutoff = datetime.now() - timedelta(days=days)
                
                cursor.execute('''
                    DELETE FROM performance_metrics
                    WHERE timestamp < ?
                ''', (cutoff.isoformat(),))
                
                deleted = cursor.rowcount
                conn.commit()
                
                logger.info(f"Cleanup: Deleted {deleted} old metrics")
                return deleted
        except Exception as e:
            logger.error(f"Metrics cleanup failed: {e}")
            return 0


class RequestTimer:
    """Context manager for timing requests"""
    
    def __init__(self, monitor: PerformanceMonitor, operation_name: str):
        self.monitor = monitor
        self.operation_name = operation_name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time
        self.monitor.record_metric(
            metric_type=f"request_time_{self.operation_name}",
            value=elapsed,
            details="seconds"
        )
        
        # Check thresholds
        if self.operation_name == "price_fetch":
            self.monitor.check_alert_threshold("response_time_p95", elapsed, 5.0)
        elif self.operation_name == "calculation":
            self.monitor.check_alert_threshold("response_time_p95", elapsed, 2.0)
        elif self.operation_name == "database_query":
            self.monitor.check_alert_threshold("database_query_time", elapsed, 0.05)


# Singleton instance
monitor = PerformanceMonitor()
