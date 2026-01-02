"""
Target Selection Utility
FOR AUTHORIZED SECURITY TESTING ONLY

Usage for Red Teaming:
---------------------
The Target Selector utility provides an interactive interface for selecting targets
from discovered network hosts. It displays hosts in a simple numbered list and allows
selection by number, range, or 'all'.

Example Usage:
    from utils.logger import FrameworkLogger
    from core.modules.network_discovery import NetworkDiscovery
    from core.modules.target_selector import TargetSelector
    
    # Discover hosts
    logger = FrameworkLogger("engagement")
    discovery = NetworkDiscovery(logger)
    results = discovery.discover_local_network()
    
    # Display and select targets
    selector = TargetSelector(logger)
    selected_ips = selector.interactive_select(results['hosts'])
    print(f"Selected targets: {selected_ips}")

Red Team Use Cases:
- Selecting targets from discovered hosts
- Interactive target selection
- Filtering hosts by criteria
- Managing target lists
"""

from typing import Dict, List, Optional
from utils.logger import FrameworkLogger


class TargetSelector:
    """Target selection utility"""
    
    def __init__(self, logger: Optional[FrameworkLogger] = None):
        """
        Initialize target selector
        
        Args:
            logger: Logger instance (optional)
        """
        self.logger = logger
        self.selected_targets = []
    
    def display_hosts(self, hosts: List[Dict]) -> None:
        """
        Display discovered hosts in simple numbered list
        
        Args:
            hosts: List of discovered host dictionaries
        """
        if not hosts:
            print("\n[!] No hosts discovered")
            return
        
        print("\n" + "="*70)
        print("DISCOVERED NETWORK HOSTS")
        print("="*70)
        print(f"{'#':<4} {'IP Address':<18} {'Hostname':<25} {'Open Ports':<20}")
        print("-"*70)
        
        for i, host in enumerate(hosts, 1):
            ip = host.get('ip', 'N/A')
            hostname = host.get('hostname', 'unknown')
            if len(hostname) > 23:
                hostname = hostname[:20] + "..."
            
            ports = host.get('open_ports', [])
            if ports:
                port_str = ", ".join(map(str, ports[:5]))
                if len(ports) > 5:
                    port_str += f" (+{len(ports)-5} more)"
            else:
                port_str = "none"
            
            print(f"{i:<4} {ip:<18} {hostname:<25} {port_str:<20}")
        
        print("="*70)
        print(f"Total: {len(hosts)} hosts discovered\n")
    
    def interactive_select(self, hosts: List[Dict]) -> List[str]:
        """
        Interactive selection of targets
        
        Args:
            hosts: List of discovered hosts
        
        Returns:
            List of selected IP addresses
        
        Examples:
            "1,3,5" - Select hosts 1, 3, and 5
            "1-5" - Select hosts 1 through 5
            "all" - Select all hosts
            "1,3-5,7" - Mixed selection
        """
        if not hosts:
            if self.logger:
                self.logger.warning("No hosts provided for selection")
            return []
        
        self.display_hosts(hosts)
        
        while True:
            try:
                selection = input("Select targets (numbers, ranges like '1-5', or 'all'): ").strip()
                
                if not selection:
                    print("[!] Please enter a selection")
                    continue
                
                if selection.lower() == 'all':
                    selected_ips = [h['ip'] for h in hosts]
                    break
                
                selected_ips = self._parse_selection(selection, hosts)
                if selected_ips:
                    break
                else:
                    print("[!] Invalid selection. Please try again.")
                    
            except KeyboardInterrupt:
                print("\n[!] Selection cancelled")
                return []
            except Exception as e:
                print(f"[!] Error: {e}. Please try again.")
        
        self.selected_targets = selected_ips
        
        if self.logger:
            self.logger.info(f"Selected {len(selected_ips)} target(s): {', '.join(selected_ips)}")
        
        print(f"\n[+] Selected {len(selected_ips)} target(s):")
        for ip in selected_ips:
            print(f"    - {ip}")
        print()
        
        return selected_ips
    
    def _parse_selection(self, selection: str, hosts: List[Dict]) -> List[str]:
        """
        Parse selection string like '1,3,5' or '1-5' or '1,3-5,7'
        
        Args:
            selection: Selection string
            hosts: List of hosts
        
        Returns:
            List of selected IP addresses
        """
        selected_ips = []
        parts = selection.split(',')
        
        for part in parts:
            part = part.strip()
            
            if not part:
                continue
            
            if '-' in part:
                # Range selection (e.g., "1-5")
                range_parts = part.split('-')
                if len(range_parts) != 2:
                    raise ValueError(f"Invalid range format: {part}")
                
                try:
                    start_idx = int(range_parts[0].strip()) - 1
                    end_idx = int(range_parts[1].strip())
                    
                    if start_idx < 0 or end_idx > len(hosts):
                        raise ValueError(f"Range {part} out of bounds (1-{len(hosts)})")
                    
                    if start_idx >= end_idx:
                        raise ValueError(f"Invalid range: start must be < end")
                    
                    # Add all hosts in range
                    for i in range(start_idx, end_idx):
                        selected_ips.append(hosts[i]['ip'])
                        
                except ValueError as e:
                    raise ValueError(f"Invalid range '{part}': {e}")
            else:
                # Single selection
                try:
                    idx = int(part) - 1
                    if idx < 0 or idx >= len(hosts):
                        raise ValueError(f"Index {part} out of bounds (1-{len(hosts)})")
                    selected_ips.append(hosts[idx]['ip'])
                except ValueError as e:
                    raise ValueError(f"Invalid index '{part}': {e}")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_ips = []
        for ip in selected_ips:
            if ip not in seen:
                seen.add(ip)
                unique_ips.append(ip)
        
        return unique_ips
    
    def filter_by_ports(self, hosts: List[Dict], ports: List[int]) -> List[str]:
        """
        Filter hosts by open ports
        
        Args:
            hosts: List of discovered hosts
            ports: List of ports to filter by
        
        Returns:
            List of IP addresses that have any of the specified ports open
        """
        filtered = []
        for host in hosts:
            open_ports = host.get('open_ports', [])
            if any(port in open_ports for port in ports):
                filtered.append(host['ip'])
        
        if self.logger:
            self.logger.info(f"Filtered {len(filtered)} hosts by ports {ports}")
        
        return filtered
    
    def filter_by_hostname(self, hosts: List[Dict], pattern: str) -> List[str]:
        """
        Filter hosts by hostname pattern
        
        Args:
            hosts: List of discovered hosts
            pattern: Pattern to match in hostname (case-insensitive)
        
        Returns:
            List of IP addresses matching the pattern
        """
        filtered = []
        pattern_lower = pattern.lower()
        
        for host in hosts:
            hostname = host.get('hostname', '').lower()
            if pattern_lower in hostname:
                filtered.append(host['ip'])
        
        if self.logger:
            self.logger.info(f"Filtered {len(filtered)} hosts by hostname pattern '{pattern}'")
        
        return filtered
    
    def get_selected_targets(self) -> List[str]:
        """Get list of currently selected targets"""
        return self.selected_targets

