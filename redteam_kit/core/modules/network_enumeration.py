"""
Network Enumeration Module
FOR AUTHORIZED SECURITY TESTING ONLY

Usage for Red Teaming:
---------------------
The Network Enumeration module performs comprehensive network reconnaissance including DNS
enumeration, subdomain discovery, network mapping, and service enumeration. This module
extends basic reconnaissance with advanced enumeration techniques.

Example Usage:
    from utils.logger import FrameworkLogger
    from core.modules.network_enumeration import NetworkEnumeration
    
    # Initialize logger and module
    logger = FrameworkLogger("net_enum")
    net_enum = NetworkEnumeration(logger)
    
    # DNS enumeration
    dns_result = net_enum.enumerate_dns("example.com")
    print(f"DNS records found: {len(dns_result['records'])}")
    
    # Subdomain discovery
    subdomains = net_enum.discover_subdomains("example.com", wordlist=["www", "mail", "ftp"])
    print(f"Subdomains found: {subdomains['subdomains']}")
    
    # Network mapping
    network_map = net_enum.map_network("192.168.1.0/24")
    print(f"Hosts discovered: {len(network_map['hosts'])}")
    
    # Service enumeration
    services = net_enum.enumerate_services("192.168.1.100", ports=[80, 443, 22])
    print(f"Services found: {len(services['services'])}")

Red Team Use Cases:
- DNS enumeration and zone transfers
- Subdomain discovery
- Network topology mapping
- Service enumeration
- SNMP enumeration
- LDAP enumeration
- SMB enumeration
- Network segment discovery
"""

from typing import Dict, List, Optional
import time
import socket
import subprocess
import re
import random
from ipaddress import ip_network, ip_address
from utils.logger import FrameworkLogger

# DNS operations (optional dependency)
try:
    import dns.resolver
    import dns.reversename
    DNS_AVAILABLE = True
except ImportError:
    DNS_AVAILABLE = False


class NetworkEnumeration:
    """Network enumeration module"""
    
    def __init__(self, logger: FrameworkLogger):
        """
        Initialize network enumeration module
        
        Args:
            logger: Logger instance
        """
        self.logger = logger
        self.enumeration_results = {}
    
    def enumerate_dns(self, domain: str, record_types: Optional[List[str]] = None) -> Dict:
        """
        Enumerate DNS records for domain
        
        Args:
            domain: Domain to enumerate
            record_types: List of record types to query (A, AAAA, MX, NS, TXT, SOA, etc.)
        
        Returns:
            Dictionary containing DNS enumeration results
        """
        self.logger.info(f"Enumerating DNS records for {domain}")
        
        if record_types is None:
            record_types = ["A", "AAAA", "MX", "NS", "TXT", "SOA", "CNAME"]
        
        results = {
            "domain": domain,
            "records": {},
            "timestamp": time.time()
        }
        
        try:
            if not DNS_AVAILABLE:
                results["note"] = "DNS enumeration requires dnspython library (pip install dnspython)"
                return results
            
            for record_type in record_types:
                try:
                    answers = dns.resolver.resolve(domain, record_type)
                    records = []
                    for answer in answers:
                        records.append(str(answer))
                    results["records"][record_type] = records
                except Exception as e:
                    self.logger.debug(f"Failed to query {record_type} record: {e}")
                    results["records"][record_type] = []
            
            # Try zone transfer
            zone_transfer = self._attempt_zone_transfer(domain)
            if zone_transfer:
                results["zone_transfer"] = zone_transfer
                results["zone_transfer_successful"] = True
            else:
                results["zone_transfer_successful"] = False
            
        except Exception as e:
            self.logger.error(f"DNS enumeration failed: {e}")
            results["error"] = str(e)
        
        return results
    
    def discover_subdomains(self, domain: str, wordlist: Optional[List[str]] = None,
                           dns_servers: Optional[List[str]] = None) -> Dict:
        """
        Discover subdomains using wordlist
        
        Args:
            domain: Base domain
            wordlist: List of subdomain prefixes to try
            dns_servers: Custom DNS servers (optional)
        
        Returns:
            Dictionary containing discovered subdomains
        """
        self.logger.info(f"Discovering subdomains for {domain}")
        
        if wordlist is None:
            wordlist = [
                "www", "mail", "ftp", "admin", "test", "dev", "staging",
                "api", "app", "web", "portal", "secure", "vpn", "remote",
                "server", "ns1", "ns2", "mx", "mx1", "mx2", "blog", "shop"
            ]
        
        results = {
            "domain": domain,
            "subdomains": [],
            "total_tested": len(wordlist),
            "timestamp": time.time()
        }
        
        if not DNS_AVAILABLE:
            results["note"] = "Subdomain discovery requires dnspython library"
            return results
        
        resolver = dns.resolver.Resolver()
        if dns_servers:
            resolver.nameservers = dns_servers
        
        for subdomain in wordlist:
            try:
                full_domain = f"{subdomain}.{domain}"
                answers = resolver.resolve(full_domain, "A")
                
                for answer in answers:
                    results["subdomains"].append({
                        "subdomain": full_domain,
                        "ip": str(answer),
                        "record_type": "A"
                    })
                    self.logger.info(f"Found subdomain: {full_domain} -> {answer}")
                
                time.sleep(0.1)  # Rate limiting
                
            except Exception:
                pass  # Subdomain doesn't exist
        
        return results
    
    def map_network(self, network_range: str, ports: Optional[List[int]] = None) -> Dict:
        """
        Map network and discover hosts
        
        Args:
            network_range: Network range in CIDR notation (e.g., "192.168.1.0/24")
            ports: Ports to scan on discovered hosts
        
        Returns:
            Dictionary containing network map
        """
        self.logger.info(f"Mapping network {network_range}")
        
        if ports is None:
            ports = [22, 23, 25, 53, 80, 135, 139, 443, 445, 3389]
        
        results = {
            "network": network_range,
            "hosts": [],
            "total_scanned": 0,
            "timestamp": time.time()
        }
        
        try:
            network = ip_network(network_range, strict=False)
            
            for ip in network.hosts():
                ip_str = str(ip)
                results["total_scanned"] += 1
                
                # Check if host is alive
                if self._is_host_alive(ip_str):
                    host_info = {
                        "ip": ip_str,
                        "hostname": self._get_hostname(ip_str),
                        "open_ports": [],
                        "os_guess": None
                    }
                    
                    # Scan ports
                    for port in ports[:10]:  # Limit ports to prevent resource exhaustion
                        if self._is_port_open(ip_str, port):
                            host_info["open_ports"].append(port)
                    
                    results["hosts"].append(host_info)
                    self.logger.info(f"Discovered host: {ip_str} ({len(host_info['open_ports'])} open ports)")
                
                # Stealthy rate limiting - random delays to avoid detection
                delay = random.uniform(0.05, 0.2)  # Random delay between 50-200ms
                time.sleep(delay)
                
        except Exception as e:
            self.logger.error(f"Network mapping failed: {e}")
            results["error"] = str(e)
        
        return results
    
    def enumerate_services(self, target: str, ports: Optional[List[int]] = None) -> Dict:
        """
        Enumerate services on target
        
        Args:
            target: Target IP or hostname
            ports: Ports to enumerate
        
        Returns:
            Dictionary containing service enumeration results
        """
        self.logger.info(f"Enumerating services on {target}")
        
        if ports is None:
            ports = [21, 22, 23, 25, 53, 80, 110, 135, 139, 143, 443, 445, 993, 995, 3306, 3389, 5432]
        
        results = {
            "target": target,
            "services": [],
            "timestamp": time.time()
        }
        
        for port in ports:
            try:
                service_info = self._enumerate_service(target, port)
                if service_info:
                    results["services"].append(service_info)
                    # Stealthy delay between service enumeration
                    delay = random.uniform(0.1, 0.3)  # Random delay 100-300ms
                    time.sleep(delay)
            except Exception as e:
                self.logger.debug(f"Failed to enumerate port {port}: {e}")
        
        return results
    
    def enumerate_smb(self, target: str) -> Dict:
        """
        Enumerate SMB shares and services
        
        Args:
            target: Target IP or hostname
        
        Returns:
            Dictionary containing SMB enumeration results
        """
        self.logger.info(f"Enumerating SMB on {target}")
        
        results = {
            "target": target,
            "shares": [],
            "users": [],
            "groups": [],
            "timestamp": time.time()
        }
        
        try:
            # Try to list shares
            cmd = f'smbclient -L //{target} -N'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                # Parse shares from output
                for line in result.stdout.split('\n'):
                    if 'Disk' in line or 'Printer' in line:
                        share_match = re.search(r'(\S+)\s+(Disk|Printer)', line)
                        if share_match:
                            results["shares"].append({
                                "name": share_match.group(1),
                                "type": share_match.group(2)
                            })
        except Exception as e:
            self.logger.error(f"SMB enumeration failed: {e}")
            results["error"] = str(e)
        
        return results
    
    def enumerate_ldap(self, target: str, base_dn: Optional[str] = None) -> Dict:
        """
        Enumerate LDAP directory
        
        Args:
            target: LDAP server IP or hostname
            base_dn: Base DN for enumeration
        
        Returns:
            Dictionary containing LDAP enumeration results
        """
        self.logger.info(f"Enumerating LDAP on {target}")
        
        results = {
            "target": target,
            "users": [],
            "groups": [],
            "computers": [],
            "timestamp": time.time()
        }
        
        # LDAP enumeration requires specialized library (ldap3)
        # This is a placeholder for real implementation
        results["note"] = "LDAP enumeration requires ldap3 library"
        
        return results
    
    def enumerate_snmp(self, target: str, community: str = "public") -> Dict:
        """
        Enumerate SNMP information
        
        Args:
            target: Target IP
            community: SNMP community string
        
        Returns:
            Dictionary containing SNMP enumeration results
        """
        self.logger.info(f"Enumerating SNMP on {target}")
        
        results = {
            "target": target,
            "community": community,
            "oids": {},
            "timestamp": time.time()
        }
        
        try:
            # Common SNMP OIDs
            oids = {
                "sysName": "1.3.6.1.2.1.1.5.0",
                "sysDescr": "1.3.6.1.2.1.1.1.0",
                "sysUpTime": "1.3.6.1.2.1.1.3.0"
            }
            
            for name, oid in oids.items():
                cmd = f'snmpget -v2c -c {community} {target} {oid}'
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
                
                if result.returncode == 0:
                    results["oids"][name] = result.stdout.strip()
                    
        except Exception as e:
            self.logger.error(f"SNMP enumeration failed: {e}")
            results["error"] = str(e)
        
        return results
    
    def _attempt_zone_transfer(self, domain: str) -> Optional[List[str]]:
        """Attempt DNS zone transfer"""
        if not DNS_AVAILABLE:
            return None
        
        try:
            import dns.query
            import dns.zone
            
            # Get nameservers
            ns_answers = dns.resolver.resolve(domain, "NS")
            nameservers = [str(answer) for answer in ns_answers]
            
            for ns in nameservers:
                try:
                    zone = dns.zone.from_xfr(dns.query.xfr(ns, domain))
                    records = []
                    for name, node in zone.nodes.items():
                        for rdataset in node.rdatasets:
                            for rdata in rdataset:
                                records.append(f"{name} {rdataset.rdtype} {rdata}")
                    return records
                except Exception:
                    continue
        except Exception:
            pass
        
        return None
    
    def _is_host_alive(self, ip: str) -> bool:
        """Check if host is alive"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.5)
            result = sock.connect_ex((ip, 22))  # Try SSH port
            sock.close()
            return result == 0
        except Exception:
            return False
    
    def _get_hostname(self, ip: str) -> Optional[str]:
        """Get hostname for IP"""
        try:
            return socket.gethostbyaddr(ip)[0]
        except Exception:
            return None
    
    def _is_port_open(self, ip: str, port: int) -> bool:
        """Check if port is open"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.3)
            result = sock.connect_ex((ip, port))
            sock.close()
            return result == 0
        except Exception:
            return False
    
    def _enumerate_service(self, target: str, port: int) -> Optional[Dict]:
        """Enumerate specific service"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            sock.connect((target, port))
            
            # Try to get banner
            banner = None
            try:
                sock.send(b"\r\n")
                banner = sock.recv(1024).decode('utf-8', errors='ignore').strip()
            except Exception:
                pass
            
            sock.close()
            
            if banner:
                return {
                    "port": port,
                    "service": self._guess_service(port),
                    "banner": banner[:200]
                }
            
            return {
                "port": port,
                "service": self._guess_service(port),
                "status": "open"
            }
            
        except Exception:
            return None
    
    def _guess_service(self, port: int) -> str:
        """Guess service from port"""
        services = {
            21: "ftp", 22: "ssh", 23: "telnet", 25: "smtp", 53: "dns",
            80: "http", 110: "pop3", 135: "msrpc", 139: "netbios",
            143: "imap", 443: "https", 445: "smb", 993: "imaps",
            995: "pop3s", 3306: "mysql", 3389: "rdp", 5432: "postgresql"
        }
        return services.get(port, "unknown")
    
    def get_enumeration_results(self) -> Dict:
        """Get all enumeration results"""
        return self.enumeration_results

