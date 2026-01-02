"""
Stealth File Access Utility
FOR AUTHORIZED SECURITY TESTING ONLY

Provides stealthy file access methods that avoid triggering macOS permission prompts.
"""

import os
import stat
import subprocess
import platform
from pathlib import Path
from typing import Optional, List, Dict, Any
import tempfile


class StealthFileAccess:
    """Stealthy file access helper to avoid permission prompts"""
    
    @staticmethod
    def can_access(path: str) -> bool:
        """
        Check if we can access file/directory without triggering permission prompt
        
        Args:
            path: Path to check
        
        Returns:
            True if accessible, False otherwise
        """
        try:
            # Use stat() instead of open() to avoid permission prompts
            os.stat(path)
            return True
        except (PermissionError, OSError, FileNotFoundError):
            return False
        except Exception:
            return False
    
    @staticmethod
    def safe_read_file(path: str, encoding: str = 'utf-8', errors: str = 'ignore') -> Optional[str]:
        """
        Safely read file without triggering permission prompts
        
        Args:
            path: File path
            encoding: File encoding
            errors: Error handling mode
        
        Returns:
            File contents or None if inaccessible
        """
        # Check access first
        if not StealthFileAccess.can_access(path):
            return None
        
        try:
            # Try direct read first
            with open(path, 'r', encoding=encoding, errors=errors) as f:
                return f.read()
        except PermissionError:
            # Fallback: Use subprocess (bypasses some sandbox restrictions)
            try:
                if platform.system() == "Darwin":  # macOS
                    # Use cat command which may bypass some restrictions
                    result = subprocess.run(
                        ['cat', path],
                        capture_output=True,
                        text=True,
                        timeout=5,
                        stderr=subprocess.DEVNULL
                    )
                    if result.returncode == 0:
                        return result.stdout
                elif platform.system() == "Linux":
                    result = subprocess.run(
                        ['cat', path],
                        capture_output=True,
                        text=True,
                        timeout=5,
                        stderr=subprocess.DEVNULL
                    )
                    if result.returncode == 0:
                        return result.stdout
            except Exception:
                pass
        except (UnicodeDecodeError, IOError, OSError):
            pass
        except Exception:
            pass
        
        return None
    
    @staticmethod
    def safe_list_directory(path: str) -> List[str]:
        """
        Safely list directory contents without triggering permission prompts
        
        Args:
            path: Directory path
        
        Returns:
            List of file/directory names, empty if inaccessible
        """
        if not StealthFileAccess.can_access(path):
            return []
        
        try:
            return os.listdir(path)
        except PermissionError:
            # Fallback: Use subprocess
            try:
                if platform.system() == "Darwin":
                    result = subprocess.run(
                        ['ls', '-1', path],
                        capture_output=True,
                        text=True,
                        timeout=5,
                        stderr=subprocess.DEVNULL
                    )
                    if result.returncode == 0:
                        return [line.strip() for line in result.stdout.split('\n') if line.strip()]
            except Exception:
                pass
        except Exception:
            pass
        
        return []
    
    @staticmethod
    def safe_walk_directory(path: str, max_depth: int = 3) -> List[str]:
        """
        Safely walk directory tree without triggering permission prompts
        
        Args:
            path: Root directory path
            max_depth: Maximum depth to traverse
        
        Returns:
            List of file paths found
        """
        files = []
        
        if not StealthFileAccess.can_access(path):
            return files
        
        def _walk(current_path: str, depth: int):
            if depth > max_depth:
                return
            
            try:
                entries = StealthFileAccess.safe_list_directory(current_path)
                
                for entry in entries:
                    full_path = os.path.join(current_path, entry)
                    
                    if StealthFileAccess.can_access(full_path):
                        try:
                            if os.path.isfile(full_path):
                                files.append(full_path)
                            elif os.path.isdir(full_path):
                                _walk(full_path, depth + 1)
                        except Exception:
                            continue
            except Exception:
                pass
        
        try:
            _walk(path, 0)
        except Exception:
            pass
        
        return files
    
    @staticmethod
    def safe_write_file(path: str, content: str, mode: str = 'w') -> bool:
        """
        Safely write file without triggering permission prompts
        
        Args:
            path: File path
            content: Content to write
            mode: Write mode
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Prefer /tmp for all writes to avoid permission prompts
            if not path.startswith('/tmp') and not path.startswith('/var/tmp'):
                # If target is protected, write to /tmp instead
                filename = os.path.basename(path)
                path = os.path.join(tempfile.gettempdir(), filename)
            
            with open(path, mode, encoding='utf-8') as f:
                f.write(content)
            return True
        except (PermissionError, IOError, OSError):
            return False
        except Exception:
            return False
    
    @staticmethod
    def get_protected_directories() -> List[str]:
        """
        Get list of macOS protected directories that trigger permission prompts
        
        Returns:
            List of protected directory patterns
        """
        if platform.system() != "Darwin":
            return []
        
        home = os.path.expanduser("~")
        return [
            os.path.join(home, "Documents"),
            os.path.join(home, "Desktop"),
            os.path.join(home, "Downloads"),
            os.path.join(home, "Pictures"),
            os.path.join(home, "Movies"),
            os.path.join(home, "Music"),
            os.path.join(home, "Library"),
        ]
    
    @staticmethod
    def get_windows_paths() -> Dict[str, List[str]]:
        """
        Get Windows-specific paths that don't require special permissions
        
        Returns:
            Dictionary containing Windows path categories
        """
        return {
            "temp": [
                os.path.expanduser("~\\AppData\\Local\\Temp"),
                "C:\\Windows\\Temp",
                "C:\\Temp"
            ],
            "config": [
                os.path.expanduser("~\\AppData\\Roaming"),
                os.path.expanduser("~\\AppData\\Local"),
                "C:\\ProgramData"
            ],
            "logs": [
                "C:\\Windows\\Logs",
                "C:\\Windows\\System32\\LogFiles"
            ],
            "user_data": [
                os.path.expanduser("~\\Documents"),
                os.path.expanduser("~\\Desktop"),
                os.path.expanduser("~\\Downloads")
            ]
        }
    
    @staticmethod
    def is_protected_path(path: str) -> bool:
        """
        Check if path is in a protected directory
        
        Args:
            path: Path to check
        
        Returns:
            True if protected, False otherwise
        """
        protected = StealthFileAccess.get_protected_directories()
        abs_path = os.path.abspath(path)
        
        for protected_dir in protected:
            if abs_path.startswith(protected_dir):
                return True
        
        return False
    
    @staticmethod
    def get_safe_temp_path(filename: str) -> str:
        """
        Get a safe temporary path for file operations
        
        Args:
            filename: Desired filename
        
        Returns:
            Safe temporary path
        """
        return os.path.join(tempfile.gettempdir(), filename)

