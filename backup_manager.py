# -*- coding: utf-8 -*-
"""
Database Backup Manager
Handles automated daily backups and restore procedures
"""

import sqlite3
import shutil
import logging
from datetime import datetime
from pathlib import Path
import os

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent / "users.db"
BACKUP_DIR = Path(__file__).parent / "backups"


class BackupManager:
    """Manages database backups and restoration"""
    
    def __init__(self, db_path: str = None, backup_dir: str = None):
        self.db_path = db_path or str(DB_PATH)
        self.backup_dir = Path(backup_dir or str(BACKUP_DIR))
        self.backup_dir.mkdir(exist_ok=True)
        logger.info(f"BackupManager initialized. Backup dir: {self.backup_dir}")
    
    def create_backup(self, backup_name: str = None) -> tuple[bool, str]:
        """
        Create a backup of the database
        
        Args:
            backup_name: Optional custom backup name (default: timestamp-based)
        
        Returns:
            (success: bool, message: str)
        """
        try:
            if not os.path.exists(self.db_path):
                return False, "Database not found"
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = backup_name or f"users_backup_{timestamp}.db"
            backup_path = self.backup_dir / backup_name
            
            # Create backup
            shutil.copy2(self.db_path, str(backup_path))
            
            # Verify backup
            if not backup_path.exists():
                return False, "Backup verification failed"
            
            file_size = backup_path.stat().st_size
            logger.info(f"Backup created: {backup_path} ({file_size} bytes)")
            
            return True, f"Backup created: {backup_name} ({file_size} bytes)"
        
        except Exception as e:
            logger.error(f"Backup creation failed: {e}")
            return False, f"Backup failed: {str(e)}"
    
    def restore_backup(self, backup_filename: str) -> tuple[bool, str]:
        """
        Restore database from backup
        
        Args:
            backup_filename: Name of backup file to restore
        
        Returns:
            (success: bool, message: str)
        """
        try:
            backup_path = self.backup_dir / backup_filename
            
            if not backup_path.exists():
                return False, f"Backup file not found: {backup_filename}"
            
            # Create safety backup before restore
            safety_backup = self.backup_dir / f"safety_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
            if os.path.exists(self.db_path):
                shutil.copy2(self.db_path, str(safety_backup))
                logger.info(f"Safety backup created: {safety_backup}")
            
            # Restore from backup
            shutil.copy2(str(backup_path), self.db_path)
            
            # Verify restoration
            if not os.path.exists(self.db_path):
                # Restore safety backup if something went wrong
                if safety_backup.exists():
                    shutil.copy2(str(safety_backup), self.db_path)
                return False, "Restore verification failed"
            
            logger.info(f"Database restored from: {backup_filename}")
            return True, f"Database restored from: {backup_filename}"
        
        except Exception as e:
            logger.error(f"Restore failed: {e}")
            return False, f"Restore failed: {str(e)}"
    
    def list_backups(self) -> list[str]:
        """List all available backups"""
        try:
            if not self.backup_dir.exists():
                return []
            
            backups = sorted([f.name for f in self.backup_dir.glob("users_backup_*.db")], reverse=True)
            return backups
        except Exception as e:
            logger.error(f"Error listing backups: {e}")
            return []
    
    def cleanup_old_backups(self, keep_count: int = 7) -> tuple[int, str]:
        """
        Delete old backups, keeping only the most recent ones
        
        Args:
            keep_count: Number of recent backups to keep (default: 7)
        
        Returns:
            (deleted_count: int, message: str)
        """
        try:
            backups = self.list_backups()
            deleted_count = 0
            
            if len(backups) > keep_count:
                to_delete = backups[keep_count:]
                for backup in to_delete:
                    try:
                        (self.backup_dir / backup).unlink()
                        deleted_count += 1
                        logger.info(f"Deleted old backup: {backup}")
                    except Exception as e:
                        logger.error(f"Failed to delete {backup}: {e}")
            
            message = f"Cleanup complete: {deleted_count} old backups deleted, {len(backups[:keep_count])} kept"
            logger.info(message)
            return deleted_count, message
        
        except Exception as e:
            logger.error(f"Backup cleanup failed: {e}")
            return 0, f"Cleanup failed: {str(e)}"
    
    def get_backup_info(self, backup_filename: str) -> dict:
        """Get information about a backup file"""
        try:
            backup_path = self.backup_dir / backup_filename
            
            if not backup_path.exists():
                return {"error": "Backup not found"}
            
            stat = backup_path.stat()
            return {
                "filename": backup_filename,
                "size_bytes": stat.st_size,
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error getting backup info: {e}")
            return {"error": str(e)}
    
    def verify_backup_integrity(self, backup_filename: str) -> tuple[bool, str]:
        """Verify that a backup file is a valid SQLite database"""
        try:
            backup_path = self.backup_dir / backup_filename
            
            if not backup_path.exists():
                return False, "Backup file not found"
            
            # Try to open and query the backup
            conn = sqlite3.connect(str(backup_path))
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            conn.close()
            
            if tables:
                return True, f"Backup is valid ({len(tables)} tables found)"
            else:
                return False, "Backup appears empty"
        
        except sqlite3.DatabaseError:
            return False, "Backup is corrupted or not a valid SQLite database"
        except Exception as e:
            logger.error(f"Integrity check failed: {e}")
            return False, str(e)


# Singleton instance
backup_manager = BackupManager()
