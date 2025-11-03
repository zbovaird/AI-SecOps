"""
Network Pivoting Module
FOR AUTHORIZED SECURITY TESTING ONLY

Usage for Red Teaming:
---------------------
The Network Pivoting module establishes network tunnels and pivots through compromised
systems to access internal networks. This module enables lateral movement and access to
restricted network segments.

Example Usage:
    from utils.logger import FrameworkLogger
    from core.modules.network_pivoting import NetworkPivoting
    
    # Initialize logger and module
    logger = FrameworkLogger("pivoting")
    pivoting = NetworkPivoting(logger, pivot_host="192.168.1.100")
    
    # Create SSH tunnel
    tunnel = pivoting.create_ssh_tunnel(
        local_port=8080,
        remote_host="10.0.0.5",
        remote_port=80
    )
    print(f"Tunnel created: {tunnel['status']}")
    
    # Create SOCKS proxy
    proxy = pivoting.create_socks_proxy(port=1080)
    print(f"SOCKS proxy: {proxy['status']}")
    
    # Port forwarding
    forward = pivoting.port_forward(
        local_port=3389,
        remote_host="10.0.0.10",
        remote_port=3389
    )
    print(f"Port forward: {forward['status']}")

Red Team Use Cases:
- SSH tunneling
- SOCKS proxy creation
- Port forwarding
- VPN tunneling
- DNS tunneling
- ICMP tunneling
- Proxy chains
- Network route manipulation
"""

from typing import Dict, List, Optional
import time
import socket
import subprocess
import threading
import platform
import os
from utils.logger import FrameworkLogger


class NetworkPivoting:
    """Network pivoting module"""
    
    def __init__(self, logger: FrameworkLogger, pivot_host: Optional[str] = None):
        """
        Initialize network pivoting module
        
        Args:
            logger: Logger instance
            pivot_host: Host to pivot through (optional)
        """
        self.logger = logger
        self.pivot_host = pivot_host
        self.active_tunnels = []
        self.active_proxies = []
    
    def create_ssh_tunnel(self, local_port: int, remote_host: str, remote_port: int,
                         username: Optional[str] = None, password: Optional[str] = None,
                         ssh_key: Optional[str] = None) -> Dict:
        """
        Create SSH tunnel
        
        Args:
            local_port: Local port to bind
            remote_host: Remote host to tunnel to
            remote_port: Remote port to tunnel to
            username: SSH username (optional)
            password: SSH password (optional)
            ssh_key: Path to SSH key (optional)
        
        Returns:
            Dictionary containing tunnel status
        """
        self.logger.info(f"Creating SSH tunnel: {local_port} -> {remote_host}:{remote_port}")
        
        results = {
            "type": "ssh_tunnel",
            "local_port": local_port,
            "remote_host": remote_host,
            "remote_port": remote_port,
            "status": "failed",
            "timestamp": time.time()
        }
        
        try:
            if platform.system() == "Windows":
                # Windows: Use OpenSSH client (if available) or netsh port forwarding
                try:
                    # Try OpenSSH client first (Windows 10+)
                    if self.pivot_host:
                        target = self.pivot_host
                    else:
                        target = remote_host
                    
                    cmd_parts = ["ssh", "-N", "-L"]
                    cmd_parts.append(f"{local_port}:{remote_host}:{remote_port}")
                    
                    if username:
                        cmd_parts.insert(1, f"{username}@{target}")
                    else:
                        cmd_parts.insert(1, target)
                    
                    if ssh_key:
                        cmd_parts.extend(["-i", ssh_key])
                    
                    # Start tunnel in background
                    process = subprocess.Popen(cmd_parts, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    
                    time.sleep(2)
                    
                    if process.poll() is None:
                        results["status"] = "active"
                        results["process_id"] = process.pid
                        results["method"] = "openssh"
                        self.active_tunnels.append(results)
                    else:
                        # Fallback to netsh port forwarding (local only, no SSH)
                        results["error"] = "SSH tunnel failed, using netsh port forwarding"
                        results["note"] = "netsh port forwarding is local only"
                except FileNotFoundError:
                    # OpenSSH not available, use netsh port forwarding (local only)
                    try:
                        # netsh port forwarding (requires admin privileges)
                        cmd = f'netsh interface portproxy add v4tov4 listenport={local_port} listenaddress=127.0.0.1 connectport={remote_port} connectaddress={remote_host}'
                        result = subprocess.run(cmd, shell=True, capture_output=True, timeout=5)
                        if result.returncode == 0:
                            results["status"] = "active"
                            results["method"] = "netsh"
                            results["note"] = "Local port forwarding via netsh"
                            self.active_tunnels.append(results)
                        else:
                            results["error"] = "netsh port forwarding failed (may require admin)"
                    except Exception as e:
                        results["error"] = f"Both SSH and netsh failed: {e}"
            else:
                # Linux/macOS: Use SSH
                if self.pivot_host:
                    target = self.pivot_host
                else:
                    target = remote_host
                
                # Build SSH command
                cmd_parts = ["ssh", "-N", "-L"]
                cmd_parts.append(f"{local_port}:{remote_host}:{remote_port}")
                
                if username:
                    cmd_parts.insert(1, f"{username}@{target}")
                else:
                    cmd_parts.insert(1, target)
                
                if ssh_key:
                    cmd_parts.extend(["-i", ssh_key])
                
                if password:
                    # Use sshpass for password authentication
                    cmd = f'sshpass -p "{password}" ' + " ".join(cmd_parts)
                else:
                    cmd = " ".join(cmd_parts)
                
                # Start tunnel in background
                process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                # Give it a moment to establish
                time.sleep(2)
                
                if process.poll() is None:
                    results["status"] = "active"
                    results["process_id"] = process.pid
                    results["method"] = "ssh"
                    self.active_tunnels.append(results)
                else:
                    results["error"] = "Tunnel failed to establish"
                
        except Exception as e:
            self.logger.error(f"SSH tunnel creation failed: {e}")
            results["error"] = str(e)
        
        return results
    
    def create_socks_proxy(self, port: int = 1080, username: Optional[str] = None,
                          password: Optional[str] = None) -> Dict:
        """
        Create SOCKS proxy
        
        Args:
            port: Port for SOCKS proxy
            username: Username for authentication (optional)
            password: Password for authentication (optional)
        
        Returns:
            Dictionary containing proxy status
        """
        self.logger.info(f"Creating SOCKS proxy on port {port}")
        
        results = {
            "type": "socks_proxy",
            "port": port,
            "status": "failed",
            "timestamp": time.time()
        }
        
        try:
            if not self.pivot_host:
                results["error"] = "No pivot host specified"
                return results
            
            if platform.system() == "Windows":
                # Windows: Use OpenSSH client for SOCKS proxy
                try:
                    cmd_parts = ["ssh", "-N", "-D"]
                    cmd_parts.append(str(port))
                    
                    if username:
                        cmd_parts.append(f"{username}@{self.pivot_host}")
                    else:
                        cmd_parts.append(self.pivot_host)
                    
                    process = subprocess.Popen(cmd_parts, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    
                    time.sleep(2)
                    
                    if process.poll() is None:
                        results["status"] = "active"
                        results["process_id"] = process.pid
                        results["method"] = "openssh"
                        self.active_proxies.append(results)
                    else:
                        results["error"] = "SOCKS proxy failed to establish"
                except FileNotFoundError:
                    results["error"] = "OpenSSH client not available (install OpenSSH client for Windows)"
            else:
                # Linux/macOS: Use SSH dynamic port forwarding
                cmd_parts = ["ssh", "-N", "-D"]
                cmd_parts.append(str(port))
                
                if username:
                    cmd_parts.append(f"{username}@{self.pivot_host}")
                else:
                    cmd_parts.append(self.pivot_host)
                
                if password:
                    cmd = f'sshpass -p "{password}" ' + " ".join(cmd_parts)
                else:
                    cmd = " ".join(cmd_parts)
                
                process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                time.sleep(2)
                
                if process.poll() is None:
                    results["status"] = "active"
                    results["process_id"] = process.pid
                    results["method"] = "ssh"
                    self.active_proxies.append(results)
                else:
                    results["error"] = "SOCKS proxy failed to establish"
                
        except Exception as e:
            self.logger.error(f"SOCKS proxy creation failed: {e}")
            results["error"] = str(e)
        
        return results
    
    def port_forward(self, local_port: int, remote_host: str, remote_port: int,
                    protocol: str = "tcp") -> Dict:
        """
        Create port forward
        
        Args:
            local_port: Local port
            remote_host: Remote host
            remote_port: Remote port
            protocol: Protocol (tcp/udp)
        
        Returns:
            Dictionary containing port forward status
        """
        self.logger.info(f"Creating port forward: {local_port} -> {remote_host}:{remote_port}")
        
        results = {
            "type": "port_forward",
            "local_port": local_port,
            "remote_host": remote_host,
            "remote_port": remote_port,
            "protocol": protocol,
            "status": "failed",
            "timestamp": time.time()
        }
        
        try:
            # Create simple TCP port forward
            def forward_connection(client_sock, remote_host, remote_port):
                try:
                    remote_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    remote_sock.connect((remote_host, remote_port))
                    
                    # Forward data
                    while True:
                        data = client_sock.recv(4096)
                        if not data:
                            break
                        remote_sock.send(data)
                        
                        response = remote_sock.recv(4096)
                        if not response:
                            break
                        client_sock.send(response)
                        
                except Exception:
                    pass
                finally:
                    client_sock.close()
                    remote_sock.close()
            
            # Start forwarding server
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind(("127.0.0.1", local_port))
            server.listen(5)
            
            results["status"] = "active"
            results["server_socket"] = True
            
            # Start forwarding in background thread
            def accept_connections():
                while True:
                    client_sock, addr = server.accept()
                    thread = threading.Thread(
                        target=forward_connection,
                        args=(client_sock, remote_host, remote_port)
                    )
                    thread.daemon = True
                    thread.start()
            
            thread = threading.Thread(target=accept_connections)
            thread.daemon = True
            thread.start()
            
            self.active_tunnels.append(results)
            
        except Exception as e:
            self.logger.error(f"Port forward creation failed: {e}")
            results["error"] = str(e)
        
        return results
    
    def create_dns_tunnel(self, domain: str, local_port: int = 53) -> Dict:
        """
        Create DNS tunnel
        
        Args:
            domain: Domain for DNS queries
            local_port: Local DNS port
        
        Returns:
            Dictionary containing DNS tunnel status
        """
        self.logger.info(f"Creating DNS tunnel using {domain}")
        
        results = {
            "type": "dns_tunnel",
            "domain": domain,
            "local_port": local_port,
            "status": "failed",
            "timestamp": time.time()
        }
        
        # DNS tunneling requires specialized tools
        results["note"] = "DNS tunneling requires specialized tools (dnscat2, etc.)"
        
        return results
    
    def create_icmp_tunnel(self, remote_host: str) -> Dict:
        """
        Create ICMP tunnel
        
        Args:
            remote_host: Remote host for tunnel
        
        Returns:
            Dictionary containing ICMP tunnel status
        """
        self.logger.info(f"Creating ICMP tunnel to {remote_host}")
        
        results = {
            "type": "icmp_tunnel",
            "remote_host": remote_host,
            "status": "failed",
            "timestamp": time.time()
        }
        
        # ICMP tunneling requires specialized tools
        results["note"] = "ICMP tunneling requires specialized tools (ptunnel, etc.)"
        
        return results
    
    def create_proxy_chain(self, proxies: List[Dict]) -> Dict:
        """
        Create proxy chain
        
        Args:
            proxies: List of proxy configurations
        
        Returns:
            Dictionary containing proxy chain status
        """
        self.logger.info(f"Creating proxy chain with {len(proxies)} proxies")
        
        results = {
            "type": "proxy_chain",
            "proxies": len(proxies),
            "status": "failed",
            "timestamp": time.time()
        }
        
        # Proxy chaining requires specialized configuration
        results["note"] = "Proxy chaining requires specialized configuration"
        
        return results
    
    def get_active_tunnels(self) -> List[Dict]:
        """Get all active tunnels"""
        return self.active_tunnels
    
    def get_active_proxies(self) -> List[Dict]:
        """Get all active proxies"""
        return self.active_proxies
    
    def stop_tunnel(self, tunnel_id: int) -> Dict:
        """Stop a tunnel"""
        try:
            tunnel = self.active_tunnels[tunnel_id]
            
            if "process_id" in tunnel:
                # Stop process
                if platform.system() == "Windows":
                    subprocess.run(["taskkill", "/F", "/PID", str(tunnel["process_id"])], 
                                 capture_output=True, timeout=5)
                else:
                    subprocess.run(["kill", str(tunnel["process_id"])], timeout=5)
                tunnel["status"] = "stopped"
                return {"status": "stopped", "tunnel_id": tunnel_id}
            
            elif tunnel.get("method") == "netsh":
                # Remove netsh port forwarding
                try:
                    cmd = f'netsh interface portproxy delete v4tov4 listenport={tunnel["local_port"]} listenaddress=127.0.0.1'
                    subprocess.run(cmd, shell=True, capture_output=True, timeout=5)
                    tunnel["status"] = "stopped"
                    return {"status": "stopped", "tunnel_id": tunnel_id}
                except Exception:
                    pass
                    
        except Exception as e:
            return {"status": "failed", "error": str(e)}
        
        return {"status": "not_found"}

