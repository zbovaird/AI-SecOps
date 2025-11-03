"""
Reconnaissance Module
FOR AUTHORIZED SECURITY TESTING ONLY

Usage for Red Teaming:
---------------------
The Reconnaissance module is used to gather information about target systems during
the initial phase of a red team engagement. It collects system information, network
details, running services, vulnerabilities, and user enumeration data.

Example Usage:
    from utils.logger import FrameworkLogger
    from core.modules.recon import ReconModule
    
    # Initialize logger and module with target
    logger = FrameworkLogger("recon_test")
    recon = ReconModule(logger, target="192.168.1.100")
    
    # Perform reconnaissance on target
    recon_results = recon.perform_recon()
    # OR specify target when calling:
    # recon_results = recon.perform_recon(target="192.168.1.100")
    # OR scan specific ports:
    # recon_results = recon.perform_recon(target="192.168.1.100", ports=[22, 80, 443, 3389])
    
    # Access collected data
    system_info = recon_results["info_gathered"]["system_info"]
    network_info = recon_results["info_gathered"]["network_info"]
    services = recon_results["info_gathered"]["service_detection"]
    vulnerabilities = recon_results["info_gathered"]["vulnerability_scan"]
    
    # Retrieve stored recon data later
    stored_data = recon.get_recon_data()
    
    print(f"Target: {recon_results['target']}")
    print(f"IP Address: {recon_results['ip_address']}")
    print(f"Reconnaissance completed: {recon_results['status']}")
    print(f"Open ports: {len(network_info['open_ports'])}")
    print(f"Services detected: {len(services)}")
    print(f"Vulnerabilities found: {len(vulnerabilities)}")

Red Team Use Cases:
- Initial target assessment
- Information gathering phase
- Service discovery
- Vulnerability identification
- User enumeration
- Network mapping
"""

from typing import Dict, List, Optional
import time
import socket
import subprocess
import platform
import re
from urllib.parse import urlparse
from utils.logger import FrameworkLogger


class ReconModule:
    """Reconnaissance module for security testing"""
    
    def __init__(self, logger: FrameworkLogger, target: Optional[str] = None):
        """
        Initialize recon module
        
        Args:
            logger: Logger instance
            target: Target IP address, domain name, or URL (optional)
        """
        self.logger = logger
        self.target = target
        self.recon_data = {}
    
    def perform_recon(self, target: Optional[str] = None, ports: Optional[List[int]] = None) -> Dict:
        """
        Perform reconnaissance to gather information about target
        
        Args:
            target: Target IP address, domain, or URL (if not set in __init__)
            ports: List of ports to scan (defaults to common ports)
        
        Returns:
            Dictionary containing reconnaissance results
        """
        # Use target from parameter or instance variable
        scan_target = target or self.target
        
        if not scan_target:
            self.logger.error("No target specified for reconnaissance")
            return {
                "status": "failed",
                "error": "No target specified",
                "timestamp": time.time()
            }
        
        # Parse target to extract IP/hostname
        parsed_target = self._parse_target(scan_target)
        hostname = parsed_target["hostname"]
        ip_address = parsed_target["ip"]
        
        self.logger.info(f"Performing reconnaissance on {scan_target} ({ip_address})")
        
        # Default ports if not specified - limit to common ports to prevent resource exhaustion
        if ports is None:
            ports = [21, 22, 23, 25, 53, 80, 110, 111, 135, 139, 143, 443, 445, 993, 995, 1723, 3306, 3389, 5900, 8080]
        
        # Limit maximum ports to scan to prevent resource exhaustion
        MAX_PORTS = 50
        if len(ports) > MAX_PORTS:
            self.logger.warning(f"Port list truncated from {len(ports)} to {MAX_PORTS} ports to prevent resource exhaustion")
            ports = ports[:MAX_PORTS]
        
        # Gather info sequentially to reduce memory footprint
        # Store intermediate results to avoid rescanning
        network_info = self._gather_network_info(hostname, ip_address, ports)
        self.recon_data = {"info_gathered": {"network_info": network_info}}
        
        # Now detect services (will reuse network_info)
        service_detection = self._detect_services(hostname, ip_address, ports)
        self.recon_data["info_gathered"]["service_detection"] = service_detection
        
        # Gather remaining info
        results = {
            "status": "completed",
            "target": scan_target,
            "hostname": hostname,
            "ip_address": ip_address,
            "timestamp": time.time(),
            "info_gathered": {
                "system_info": self._gather_system_info(hostname, ip_address),
                "network_info": network_info,
                "service_detection": service_detection,
                "vulnerability_scan": self._scan_vulnerabilities(hostname, ip_address),
                "user_enumeration": self._enumerate_users(hostname, ip_address),
            }
        }
        
        self.recon_data = results
        return results
    
    def _parse_target(self, target: str) -> Dict[str, str]:
        """Parse target string to extract hostname and IP"""
        # Remove protocol if present
        if "://" in target:
            parsed = urlparse(target)
            hostname = parsed.netloc or parsed.path
        else:
            hostname = target
        
        # Remove port if present
        if ":" in hostname:
            hostname = hostname.split(":")[0]
        
        # Try to resolve hostname to IP
        ip_address = hostname
        try:
            if not self._is_ip_address(hostname):
                ip_address = socket.gethostbyname(hostname)
        except socket.gaierror:
            self.logger.warning(f"Could not resolve {hostname} to IP address")
            ip_address = "unknown"
        
        return {
            "hostname": hostname,
            "ip": ip_address
        }
    
    def _is_ip_address(self, address: str) -> bool:
        """Check if string is an IP address"""
        try:
            socket.inet_aton(address)
            return True
        except socket.error:
            return False
    
    def _gather_system_info(self, hostname: str, ip_address: str) -> Dict:
        """Gather system information about target"""
        info = {
            "hostname": hostname,
            "ip_address": ip_address,
            "platform": "unknown",
            "architecture": "unknown",
            "version": "unknown",
            "ttl": None,
            "os_guess": []
        }
        
        # Ping to get TTL (OS fingerprinting)
        try:
            if platform.system().lower() == "windows":
                result = subprocess.run(
                    ["ping", "-n", "1", "-w", "1000", ip_address],
                    capture_output=True,
                    timeout=5,
                    text=True
                )
            else:
                result = subprocess.run(
                    ["ping", "-c", "1", "-W", "1", ip_address],
                    capture_output=True,
                    timeout=5,
                    text=True
                )
            
            # Extract TTL from ping output
            ttl_match = re.search(r'ttl=(\d+)', result.stdout.lower())
            if ttl_match:
                ttl = int(ttl_match.group(1))
                info["ttl"] = ttl
                # OS guessing based on TTL
                if 64 <= ttl <= 128:
                    info["os_guess"].append("Linux/Unix")
                elif ttl <= 64:
                    info["os_guess"].append("Linux")
                elif ttl >= 128:
                    info["os_guess"].append("Windows")
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            pass
        
        return info
    
    def _gather_network_info(self, hostname: str, ip_address: str, ports: List[int]) -> Dict:
        """Gather network information"""
        open_ports = []
        active_connections = []
        
        # Scan ports with delays to prevent resource exhaustion
        self.logger.info(f"Scanning {len(ports)} ports on {ip_address}")
        for i, port in enumerate(ports):
            # Add small delay between scans to prevent overwhelming system
            if i > 0:
                time.sleep(0.1)  # 100ms delay between scans
            
            if self._scan_port(ip_address, port):
                open_ports.append({
                    "port": port,
                    "protocol": "tcp",
                    "status": "open"
                })
                self.logger.info(f"Port {port} is open")
        
        return {
            "open_ports": open_ports,
            "ports_scanned": len(ports),
            "active_connections": active_connections,
            "network_interfaces": []
        }
    
    def _scan_port(self, host: str, port: int, timeout: float = 1.0) -> bool:
        """Scan a single port - optimized to prevent resource exhaustion"""
        sock = None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            # Use connect_ex instead of connect to avoid exceptions
            result = sock.connect_ex((host, port))
            return result == 0
        except Exception:
            return False
        finally:
            # Always close socket to prevent resource leaks
            if sock:
                try:
                    sock.close()
                except Exception:
                    pass
    
    def _detect_services(self, hostname: str, ip_address: str, ports: List[int]) -> List[Dict]:
        """Detect running services on open ports - optimized to reuse port scan results"""
        services = []
        common_ports = {
            21: "ftp", 22: "ssh", 23: "telnet", 25: "smtp", 53: "dns",
            80: "http", 110: "pop3", 111: "rpcbind", 135: "msrpc", 139: "netbios-ssn",
            143: "imap", 443: "https", 445: "smb", 993: "imaps", 995: "pop3s",
            1723: "pptp", 3306: "mysql", 3389: "rdp", 5900: "vnc", 8080: "http-proxy"
        }
        
        # Reuse open ports from network_info if available to avoid rescanning
        network_info = self.recon_data.get("info_gathered", {}).get("network_info", {})
        open_ports_list = network_info.get("open_ports", [])
        
        if open_ports_list:
            # Use already scanned ports
            open_ports = [p["port"] for p in open_ports_list]
        else:
            # Fallback: scan ports (with delays)
            open_ports = []
            for i, port in enumerate(ports):
                if i > 0:
                    time.sleep(0.1)  # Delay between scans
                if self._scan_port(ip_address, port):
                    open_ports.append(port)
        
        # Identify services with delays
        for i, port in enumerate(open_ports):
            if i > 0:
                time.sleep(0.15)  # Delay between service detection
            
            service_name = common_ports.get(port, "unknown")
            version = self._get_service_version(ip_address, port, service_name)
            
            services.append({
                "service": service_name,
                "port": port,
                "version": version,
                "status": "active",
                "banner": self._get_banner(ip_address, port)
            })
        
        return services
    
    def _get_service_version(self, host: str, port: int, service: str) -> str:
        """Attempt to get service version - optimized with proper cleanup"""
        sock = None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2.0)
            sock.connect((host, port))
            
            # Try to get banner
            if service == "ssh":
                sock.send(b"\n")
                banner = sock.recv(1024).decode('utf-8', errors='ignore')
                if "SSH" in banner:
                    return banner.split("\n")[0].strip()
            elif service in ["http", "https"]:
                sock.send(b"HEAD / HTTP/1.1\r\nHost: " + host.encode() + b"\r\n\r\n")
                banner = sock.recv(1024).decode('utf-8', errors='ignore')
                if "Server:" in banner:
                    server_match = re.search(r'Server:\s*(.+)', banner, re.IGNORECASE)
                    if server_match:
                        return server_match.group(1).strip()
            
        except Exception:
            pass
        finally:
            if sock:
                try:
                    sock.close()
                except Exception:
                    pass
        
        return "detected"
    
    def _get_banner(self, host: str, port: int) -> str:
        """Get service banner - optimized with proper cleanup"""
        sock = None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2.0)
            sock.connect((host, port))
            sock.send(b"\n")
            banner = sock.recv(1024).decode('utf-8', errors='ignore')
            return banner.strip()[:200]  # Limit banner length
        except Exception:
            return ""
        finally:
            if sock:
                try:
                    sock.close()
                except Exception:
                    pass
    
    def _scan_vulnerabilities(self, hostname: str, ip_address: str) -> List[Dict]:
        """Scan for vulnerabilities"""
        vulnerabilities = []
        
        # Basic vulnerability checks based on open services
        # In production, this would integrate with vulnerability scanners
        services = self.recon_data.get("info_gathered", {}).get("service_detection", [])
        
        for service in services:
            service_name = service.get("service", "").lower()
            port = service.get("port", 0)
            
            # Check for common misconfigurations
            if service_name == "ftp" and port == 21:
                vulnerabilities.append({
                    "type": "ftp_anonymous",
                    "severity": "medium",
                    "description": f"FTP service detected on port {port} - check for anonymous login",
                    "port": port,
                    "service": service_name
                })
            
            if service_name == "telnet" and port == 23:
                vulnerabilities.append({
                    "type": "telnet_cleartext",
                    "severity": "high",
                    "description": f"Telnet service detected on port {port} - unencrypted protocol",
                    "port": port,
                    "service": service_name
                })
            
            if service_name == "smb" and port == 445:
                vulnerabilities.append({
                    "type": "smb_exposure",
                    "severity": "medium",
                    "description": f"SMB service detected on port {port} - check for SMB vulnerabilities",
                    "port": port,
                    "service": service_name
                })
        
        return vulnerabilities
    
    def _enumerate_users(self, hostname: str, ip_address: str) -> List[str]:
        """Enumerate users"""
        users = []
        
        # Basic user enumeration - in production would use tools like enum4linux, rpcclient, etc.
        # This is a placeholder for real enumeration techniques
        try:
            # Check for SMB/RPC
            if self._scan_port(ip_address, 445):
                self.logger.info("SMB service detected - user enumeration possible")
                # In production: use rpcclient or enum4linux
                # users.extend(self._enumerate_smb_users(ip_address))
        except Exception:
            pass
        
        return users
    
    def get_recon_data(self) -> Dict:
        """Get collected reconnaissance data"""
        return self.recon_data

