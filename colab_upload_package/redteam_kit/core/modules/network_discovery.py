"""
Network Discovery Module - Stealthy Network Scanning
FOR AUTHORIZED SECURITY TESTING ONLY

Usage for Red Teaming:
---------------------
The Network Discovery module performs stealthy network scanning to discover active hosts
on the local network. It uses stealthy techniques like ICMP ping, random delays, and
minimal port scanning to avoid detection.

Example Usage:
    from utils.logger import FrameworkLogger
    from core.modules.network_discovery import NetworkDiscovery
    
    # Initialize logger and module
    logger = FrameworkLogger("network_discovery")
    discovery = NetworkDiscovery(logger)
    
    # Auto-detect local network and discover hosts (stealthy mode)
    results = discovery.discover_local_network(stealth_mode=True)
    print(f"Discovered {len(results['hosts'])} hosts")
    
    # Discover specific network range
    results = discovery.discover_network("192.168.1.0/24", stealth_mode=True)
    
    # Non-stealthy mode (faster but more detectable)
    results = discovery.discover_network("192.168.1.0/24", stealth_mode=False)

Red Team Use Cases:
- Stealthy network reconnaissance
- Host discovery before attack chain
- Network mapping with minimal footprint
- Identifying potential targets
- Avoiding IDS/IPS detection
"""

from typing import Dict, List, Optional
import time
import socket
import subprocess
import platform
import random
from ipaddress import ip_network
from utils.logger import FrameworkLogger


class NetworkDiscovery:
    """Stealthy network discovery module"""
    
    def __init__(self, logger: FrameworkLogger):
        """
        Initialize network discovery module
        
        Args:
            logger: Logger instance
        """
        self.logger = logger
        self.discovered_hosts = []
    
    def discover_local_network(self, stealth_mode: bool = True) -> Dict:
        """
        Auto-detect local network range and discover hosts stealthily
        
        Args:
            stealth_mode: Use slower, more stealthy scanning (default: True)
        
        Returns:
            Dictionary containing discovered hosts
        """
        self.logger.info("Auto-detecting local network range...")
        
        # Auto-detect network range
        network_range = self._get_local_network_range()
        if not network_range:
            return {
                "error": "Could not detect local network range",
                "hosts": [],
                "network_range": None
            }
        
        self.logger.info(f"Discovered network range: {network_range}")
        self.logger.info(f"Stealth mode: {'enabled' if stealth_mode else 'disabled'}")
        
        return self.discover_network(network_range, stealth_mode=stealth_mode)
    
    def discover_network(self, network_range: str, stealth_mode: bool = True) -> Dict:
        """
        Discover hosts on specified network range
        
        Args:
            network_range: CIDR notation (e.g., "192.168.1.0/24")
            stealth_mode: Use stealthy scanning techniques
        
        Returns:
            Dictionary with discovered hosts
        """
        self.logger.info(f"Scanning network: {network_range} (stealth_mode={stealth_mode})")
        
        results = {
            "network_range": network_range,
            "hosts": [],
            "total_scanned": 0,
            "stealth_mode": stealth_mode,
            "timestamp": time.time()
        }
        
        try:
            network = ip_network(network_range, strict=False)
            
            # Stealthy scanning: Use ICMP ping first (less detectable)
            for ip in network.hosts():
                ip_str = str(ip)
                results["total_scanned"] += 1
                
                # Stealthy host detection
                if self._stealthy_host_check(ip_str, stealth_mode):
                    host_info = {
                        "ip": ip_str,
                        "hostname": self._get_hostname(ip_str),
                        "open_ports": [],
                        "os_guess": None,
                        "status": "alive"
                    }
                    
                    # Quick port scan (only if host is alive, and not in stealth mode)
                    if not stealth_mode:
                        host_info["open_ports"] = self._quick_port_scan(ip_str)
                    
                    results["hosts"].append(host_info)
                    self.logger.info(f"Discovered host: {ip_str} ({host_info['hostname']})")
                
                # Stealthy delay between scans
                if stealth_mode:
                    delay = random.uniform(0.5, 2.0)  # Random delays between 0.5-2 seconds
                    time.sleep(delay)
                else:
                    time.sleep(0.1)  # Faster but more detectable
            
        except Exception as e:
            self.logger.error(f"Network discovery failed: {e}")
            results["error"] = str(e)
        
        self.discovered_hosts = results["hosts"]
        self.logger.info(f"Discovery complete: {len(results['hosts'])} hosts found")
        
        return results
    
    def _get_local_network_range(self) -> Optional[str]:
        """Auto-detect local network range"""
        try:
            hostname = socket.gethostname()
            local_ip = socket.gethostbyname(hostname)
            
            # Extract network prefix (assumes /24 subnet)
            network_prefix = '.'.join(local_ip.split('.')[:-1])
            network_range = f"{network_prefix}.0/24"
            
            self.logger.info(f"Detected local IP: {local_ip}")
            self.logger.info(f"Detected network range: {network_range}")
            
            return network_range
        except Exception as e:
            self.logger.error(f"Failed to detect local network range: {e}")
            return None
    
    def _stealthy_host_check(self, ip: str, stealth_mode: bool) -> bool:
        """Check if host is alive using stealthy methods"""
        if stealth_mode:
            # Use ICMP ping (more stealthy than port scans)
            return self._ping_host(ip)
        else:
            # Check common ports (faster but more detectable)
            ports = [22, 80, 135, 445, 3389]
            for port in ports[:2]:  # Only check 2 ports for speed
                if self._is_port_open(ip, port, timeout=0.5):
                    return True
            return False
    
    def _ping_host(self, ip: str) -> bool:
        """Ping host to check if alive"""
        try:
            if platform.system() == "Windows":
                cmd = f'ping -n 1 -w 1000 {ip}'
            else:
                cmd = f'ping -c 1 -W 1 {ip}'
            
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                timeout=2
            )
            return result.returncode == 0
        except Exception:
            return False
    
    def _is_port_open(self, ip: str, port: int, timeout: float = 0.5) -> bool:
        """Check if port is open"""
        sock = None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((ip, port))
            return result == 0
        except Exception:
            return False
        finally:
            if sock:
                try:
                    sock.close()
                except Exception:
                    pass
    
    def _quick_port_scan(self, ip: str) -> List[int]:
        """Quick port scan (non-stealthy, only for detected hosts)"""
        open_ports = []
        common_ports = [22, 23, 25, 53, 80, 135, 139, 443, 445, 3389]
        
        for port in common_ports[:5]:  # Limit to 5 ports
            if self._is_port_open(ip, port, timeout=0.3):
                open_ports.append(port)
            time.sleep(0.05)  # Small delay between port checks
        
        return open_ports
    
    def _get_hostname(self, ip: str) -> str:
        """Get hostname from IP"""
        try:
            hostname = socket.gethostbyaddr(ip)[0]
            return hostname
        except Exception:
            return "unknown"
    
    def get_discovered_hosts(self) -> List[Dict]:
        """Get list of discovered hosts"""
        return self.discovered_hosts

