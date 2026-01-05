"""
OS Detection Module
FOR AUTHORIZED SECURITY TESTING ONLY

Usage for Red Teaming:
---------------------
The OS Detection module identifies the operating system of target systems both locally
and remotely. This is critical for tailoring attacks to specific OS environments and
ensuring compatibility with Windows, Linux, and macOS targets.

Example Usage:
    from utils.logger import FrameworkLogger
    from core.modules.os_detection import OSDetection
    
    # Initialize logger and module
    logger = FrameworkLogger("os_detection")
    os_detection = OSDetection(logger)
    
    # Detect local OS
    local_os = os_detection.detect_local_os()
    print(f"Local OS: {local_os['os_type']} {local_os['version']}")
    
    # Detect remote OS
    remote_os = os_detection.detect_remote_os(target="192.168.1.100")
    print(f"Remote OS: {remote_os['os_type']} {remote_os['confidence']}")
    
    # Detect OS from service banners
    banner_os = os_detection.detect_from_banner("192.168.1.100", port=22)
    print(f"Banner OS: {banner_os['os_type']}")

Red Team Use Cases:
- Initial OS identification
- Remote OS fingerprinting
- OS-specific attack selection
- Windows/Linux/macOS detection
- Version detection
- Architecture detection
"""

from typing import Dict, List, Optional
import time
import platform
import socket
import subprocess
import os
import re
import random
from utils.logger import FrameworkLogger


class OSDetection:
    """OS detection module"""
    
    def __init__(self, logger: FrameworkLogger):
        """
        Initialize OS detection module
        
        Args:
            logger: Logger instance
        """
        self.logger = logger
        self.detection_results = {}
    
    def detect_local_os(self) -> Dict:
        """
        Detect local operating system
        
        Returns:
            Dictionary containing local OS information
        """
        self.logger.info("Detecting local operating system")
        
        results = {
            "os_type": platform.system(),
            "os_name": platform.system(),
            "os_version": platform.version(),
            "os_release": platform.release(),
            "architecture": platform.machine(),
            "processor": platform.processor(),
            "hostname": platform.node(),
            "confidence": 100.0,
            "timestamp": time.time()
        }
        
        # Normalize OS type
        if results["os_type"] == "Darwin":
            results["os_type"] = "macOS"
            results["os_name"] = "macOS"
        elif results["os_type"] == "Windows":
            results["os_type"] = "Windows"
            # Get Windows version details
            try:
                version_info = self._get_windows_version()
                results.update(version_info)
            except Exception:
                pass
        elif results["os_type"] == "Linux":
            results["os_type"] = "Linux"
            # Get Linux distribution
            try:
                distro_info = self._get_linux_distro()
                results.update(distro_info)
            except Exception:
                pass
        
        self.detection_results["local"] = results
        return results
    
    def detect_remote_os(self, target: str, ports: Optional[List[int]] = None) -> Dict:
        """
        Detect remote operating system
        
        Args:
            target: Target IP address or hostname
            ports: Ports to probe (optional)
        
        Returns:
            Dictionary containing remote OS detection results
        """
        self.logger.info(f"Detecting remote OS for {target}")
        
        if ports is None:
            ports = [22, 80, 135, 139, 445, 3389]
        
        results = {
            "target": target,
            "os_type": "Unknown",
            "confidence": 0.0,
            "detection_methods": [],
            "indicators": [],
            "timestamp": time.time()
        }
        
        # Method 1: TTL-based detection
        ttl_result = self._detect_from_ttl(target)
        if ttl_result:
            results["indicators"].append(ttl_result)
            if ttl_result["confidence"] > results["confidence"]:
                results["os_type"] = ttl_result["os_type"]
                results["confidence"] = ttl_result["confidence"]
                results["detection_methods"].append("ttl")
        
        # Method 2: Service banner detection
        for i, port in enumerate(ports):
            # Stealthy delay between port checks
            if i > 0:
                delay = random.uniform(0.3, 1.0)  # Random delay 300ms-1s
                time.sleep(delay)
            
            banner_result = self.detect_from_banner(target, port)
            if banner_result and banner_result.get("os_type") != "Unknown":
                results["indicators"].append(banner_result)
                if banner_result["confidence"] > results["confidence"]:
                    results["os_type"] = banner_result["os_type"]
                    results["confidence"] = banner_result["confidence"]
                    results["detection_methods"].append(f"banner_{port}")
        
        # Method 3: Port-based detection
        port_result = self._detect_from_ports(target, ports)
        if port_result:
            results["indicators"].append(port_result)
            if port_result["confidence"] > results["confidence"]:
                results["os_type"] = port_result["os_type"]
                results["confidence"] = port_result["confidence"]
                results["detection_methods"].append("ports")
        
        # Method 4: SMB detection (Windows-specific)
        if self._check_smb(target):
            results["indicators"].append({
                "method": "smb",
                "os_type": "Windows",
                "confidence": 85.0
            })
            if results["confidence"] < 85.0:
                results["os_type"] = "Windows"
                results["confidence"] = 85.0
                results["detection_methods"].append("smb")
        
        self.detection_results[target] = results
        return results
    
    def detect_from_banner(self, target: str, port: int) -> Dict:
        """
        Detect OS from service banner
        
        Args:
            target: Target IP address
            port: Port to connect to
        
        Returns:
            Dictionary containing banner-based OS detection
        """
        results = {
            "target": target,
            "port": port,
            "os_type": "Unknown",
            "confidence": 0.0,
            "banner": None,
            "timestamp": time.time()
        }
        
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3)
            sock.connect((target, port))
            
            # Try to get banner
            try:
                sock.send(b"\r\n")
                banner = sock.recv(1024).decode('utf-8', errors='ignore').strip()
                results["banner"] = banner
                
                # Analyze banner for OS indicators
                banner_lower = banner.lower()
                
                # Windows indicators
                windows_indicators = ['windows', 'microsoft', 'iis', 'smb', 'ms-wbt-server']
                if any(indicator in banner_lower for indicator in windows_indicators):
                    results["os_type"] = "Windows"
                    results["confidence"] = 80.0
                
                # Linux indicators
                linux_indicators = ['linux', 'ubuntu', 'debian', 'centos', 'redhat', 'apache', 'nginx', 'openssh']
                if any(indicator in banner_lower for indicator in linux_indicators):
                    results["os_type"] = "Linux"
                    results["confidence"] = 75.0
                
                # macOS indicators
                mac_indicators = ['darwin', 'macos', 'mac os']
                if any(indicator in banner_lower for indicator in mac_indicators):
                    results["os_type"] = "macOS"
                    results["confidence"] = 80.0
                
            except Exception:
                pass
            
            sock.close()
            
        except Exception as e:
            self.logger.debug(f"Banner detection failed for {target}:{port}: {e}")
        
        return results
    
    def _detect_from_ttl(self, target: str) -> Optional[Dict]:
        """
        Detect OS from TTL value
        
        Args:
            target: Target IP address
        
        Returns:
            Dictionary containing TTL-based OS detection
        """
        try:
            # Ping target to get TTL
            if platform.system() == "Windows":
                cmd = f'ping -n 1 {target}'
            else:
                cmd = f'ping -c 1 {target}'
            
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
            
            # Parse TTL from output
            ttl_match = re.search(r'TTL[=:]?\s*(\d+)', result.stdout, re.IGNORECASE)
            if ttl_match:
                ttl = int(ttl_match.group(1))
                
                # TTL ranges for different OSes
                # Windows: typically 128 (can be 64 on newer versions)
                # Linux: typically 64
                # macOS: typically 64
                
                if ttl <= 64:
                    return {
                        "method": "ttl",
                        "ttl": ttl,
                        "os_type": "Linux/macOS",
                        "confidence": 60.0
                    }
                elif ttl >= 128:
                    return {
                        "method": "ttl",
                        "ttl": ttl,
                        "os_type": "Windows",
                        "confidence": 70.0
                    }
                elif ttl > 64 and ttl < 128:
                    return {
                        "method": "ttl",
                        "ttl": ttl,
                        "os_type": "Windows",
                        "confidence": 65.0
                    }
        except Exception:
            pass
        
        return None
    
    def _detect_from_ports(self, target: str, ports: List[int]) -> Optional[Dict]:
        """
        Detect OS from open ports
        
        Args:
            target: Target IP address
            ports: List of ports to check
        
        Returns:
            Dictionary containing port-based OS detection
        """
        try:
            open_ports = []
            for port in ports:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(0.5)
                result = sock.connect_ex((target, port))
                sock.close()
                if result == 0:
                    open_ports.append(port)
            
            # Windows-specific ports
            windows_ports = [135, 139, 445, 3389]
            linux_ports = [22, 111]
            
            windows_count = sum(1 for p in open_ports if p in windows_ports)
            linux_count = sum(1 for p in open_ports if p in linux_ports)
            
            if windows_count > linux_count and windows_count > 0:
                return {
                    "method": "ports",
                    "open_ports": open_ports,
                    "os_type": "Windows",
                    "confidence": 75.0
                }
            elif linux_count > windows_count and linux_count > 0:
                return {
                    "method": "ports",
                    "open_ports": open_ports,
                    "os_type": "Linux",
                    "confidence": 70.0
                }
        except Exception:
            pass
        
        return None
    
    def _check_smb(self, target: str) -> bool:
        """Check if SMB is available (Windows indicator)"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex((target, 445))
            sock.close()
            return result == 0
        except Exception:
            return False
    
    def _get_windows_version(self) -> Dict:
        """Get Windows version details"""
        version_info = {}
        
        try:
            result = subprocess.run(
                ['systeminfo'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                output = result.stdout
                
                # Extract OS name
                os_name_match = re.search(r'OS Name:\s*(.+)', output)
                if os_name_match:
                    version_info["os_name"] = os_name_match.group(1).strip()
                
                # Extract OS version
                os_version_match = re.search(r'OS Version:\s*(.+)', output)
                if os_version_match:
                    version_info["os_version_detail"] = os_version_match.group(1).strip()
                
                # Extract build number
                build_match = re.search(r'OS Build Number:\s*(\d+)', output)
                if build_match:
                    version_info["build_number"] = build_match.group(1)
        except Exception:
            pass
        
        return version_info
    
    def _get_linux_distro(self) -> Dict:
        """Get Linux distribution details"""
        distro_info = {}
        
        try:
            # Check /etc/os-release
            if os.path.exists("/etc/os-release"):
                with open("/etc/os-release", 'r') as f:
                    content = f.read()
                    
                    name_match = re.search(r'^NAME="?([^"]+)"?', content, re.MULTILINE)
                    if name_match:
                        distro_info["distro_name"] = name_match.group(1)
                    
                    version_match = re.search(r'^VERSION="?([^"]+)"?', content, re.MULTILINE)
                    if version_match:
                        distro_info["distro_version"] = version_match.group(1)
            
            # Fallback: uname
            result = subprocess.run(['uname', '-a'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                distro_info["kernel_info"] = result.stdout.strip()
        except Exception:
            pass
        
        return distro_info
    
    def get_os_specific_paths(self, os_type: str) -> Dict:
        """
        Get OS-specific paths for file operations
        
        Args:
            os_type: OS type (Windows, Linux, macOS)
        
        Returns:
            Dictionary containing OS-specific paths
        """
        paths = {
            "temp": [],
            "config": [],
            "logs": [],
            "user_data": []
        }
        
        if os_type == "Windows":
            paths["temp"] = [
                os.path.expanduser("~\\AppData\\Local\\Temp"),
                "C:\\Windows\\Temp",
                "C:\\Temp"
            ]
            paths["config"] = [
                os.path.expanduser("~\\AppData\\Roaming"),
                "C:\\ProgramData"
            ]
            paths["logs"] = [
                "C:\\Windows\\Logs",
                "C:\\Windows\\System32\\LogFiles"
            ]
            paths["user_data"] = [
                os.path.expanduser("~\\Documents"),
                os.path.expanduser("~\\Desktop"),
                os.path.expanduser("~\\Downloads")
            ]
        
        elif os_type in ["Linux", "macOS"]:
            paths["temp"] = ["/tmp", "/var/tmp"]
            paths["config"] = [
                os.path.expanduser("~/.config"),
                os.path.expanduser("~/.ssh"),
                os.path.expanduser("~/.aws")
            ]
            paths["logs"] = [
                "/var/log",
                "/var/log/auth.log",
                "/var/log/syslog"
            ]
            paths["user_data"] = [
                os.path.expanduser("~/Documents"),
                os.path.expanduser("~/Desktop")
            ]
        
        return paths
    
    def get_detection_results(self) -> Dict:
        """Get all detection results"""
        return self.detection_results

