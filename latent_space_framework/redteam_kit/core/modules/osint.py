"""
OSINT (Open Source Intelligence) Module
FOR AUTHORIZED SECURITY TESTING ONLY

Usage for Red Teaming:
---------------------
The OSINT module gathers open source intelligence about targets including domain information,
email addresses, social media profiles, employee information, and technology stack details.
This module aids in reconnaissance and social engineering phases.

Example Usage:
    from utils.logger import FrameworkLogger
    from core.modules.osint import OSINTModule
    
    # Initialize logger and module
    logger = FrameworkLogger("osint")
    osint = OSINTModule(logger)
    
    # Gather domain information
    domain_info = osint.gather_domain_info("example.com")
    print(f"Subdomains: {domain_info['subdomains']}")
    
    # Find email addresses
    emails = osint.find_emails("example.com")
    print(f"Emails found: {len(emails['emails'])}")
    
    # Gather employee information
    employees = osint.gather_employee_info("Example Corp")
    print(f"Employees found: {len(employees['employees'])}")
    
    # Technology stack detection
    tech_stack = osint.detect_tech_stack("https://example.com")
    print(f"Technologies: {tech_stack['technologies']}")

Red Team Use Cases:
- Domain and subdomain enumeration
- Email address discovery
- Employee information gathering
- Social media reconnaissance
- Technology stack identification
- DNS information gathering
- WHOIS information retrieval
- Certificate transparency logs
"""

from typing import Dict, List, Optional
import time
import re
import socket
import requests
import subprocess
import random
from urllib.parse import urlparse
from utils.logger import FrameworkLogger

# DNS operations (optional dependency)
try:
    import dns.resolver
    DNS_AVAILABLE = True
except ImportError:
    DNS_AVAILABLE = False


class OSINTModule:
    """OSINT (Open Source Intelligence) module"""
    
    def __init__(self, logger: FrameworkLogger):
        """
        Initialize OSINT module
        
        Args:
            logger: Logger instance
        """
        self.logger = logger
        self.intel_data = {}
    
    def gather_domain_info(self, domain: str) -> Dict:
        """
        Gather comprehensive domain information
        
        Args:
            domain: Domain name to investigate
        
        Returns:
            Dictionary containing domain information
        """
        self.logger.info(f"Gathering domain information for {domain}")
        
        results = {
            "domain": domain,
            "subdomains": [],
            "dns_records": {},
            "whois_info": {},
            "ip_addresses": [],
            "timestamp": time.time()
        }
        
        try:
            if not DNS_AVAILABLE:
                results["note"] = "DNS enumeration requires dnspython library"
                return results
            
            # DNS records
            record_types = ["A", "AAAA", "MX", "NS", "TXT", "SOA"]
            
            for i, record_type in enumerate(record_types):
                # Stealthy delay between DNS queries
                if i > 0:
                    delay = random.uniform(0.5, 1.5)  # Random delay 0.5-1.5 seconds
                    time.sleep(delay)
                
                try:
                    answers = dns.resolver.resolve(domain, record_type)
                    records = [str(answer) for answer in answers]
                    results["dns_records"][record_type] = records
                    
                    if record_type == "A":
                        results["ip_addresses"].extend(records)
                except Exception:
                    pass
            
            # WHOIS information
            try:
                whois_info = self._get_whois(domain)
                results["whois_info"] = whois_info
            except Exception as e:
                self.logger.debug(f"WHOIS lookup failed: {e}")
            
            # Subdomain discovery
            subdomains = self._discover_subdomains(domain)
            results["subdomains"] = subdomains
            
        except Exception as e:
            self.logger.error(f"Domain info gathering failed: {e}")
            results["error"] = str(e)
        
        self.intel_data[domain] = results
        return results
    
    def find_emails(self, domain: str, max_results: int = 100) -> Dict:
        """
        Find email addresses associated with domain
        
        Args:
            domain: Domain to search
            max_results: Maximum results to return
        
        Returns:
            Dictionary containing found email addresses
        """
        self.logger.info(f"Finding emails for {domain}")
        
        results = {
            "domain": domain,
            "emails": [],
            "sources": [],
            "timestamp": time.time()
        }
        
        try:
            # Search domain website
            try:
                # Stealthy delay before HTTP request
                time.sleep(random.uniform(0.5, 1.0))
                response = requests.get(f"https://{domain}", timeout=5, verify=False)
                emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', response.text)
                results["emails"].extend(emails[:max_results])
                results["sources"].append("website")
            except Exception:
                pass
            
            # Common email patterns
            common_patterns = [
                f"admin@{domain}",
                f"contact@{domain}",
                f"info@{domain}",
                f"support@{domain}",
                f"sales@{domain}",
                f"security@{domain}"
            ]
            
            # Test which emails exist (would need actual email verification)
            for email in common_patterns:
                results["emails"].append(email)
            
            # Remove duplicates
            results["emails"] = list(set(results["emails"]))[:max_results]
            
        except Exception as e:
            self.logger.error(f"Email finding failed: {e}")
            results["error"] = str(e)
        
        return results
    
    def gather_employee_info(self, company_name: str) -> Dict:
        """
        Gather employee information
        
        Args:
            company_name: Company name to search
        
        Returns:
            Dictionary containing employee information
        """
        self.logger.info(f"Gathering employee information for {company_name}")
        
        results = {
            "company": company_name,
            "employees": [],
            "sources": [],
            "timestamp": time.time()
        }
        
        # Note: This would typically require API access to LinkedIn, etc.
        # This is a placeholder for real implementation
        results["note"] = "Employee information gathering requires API access to social media platforms"
        
        return results
    
    def detect_tech_stack(self, url: str) -> Dict:
        """
        Detect technology stack used by website
        
        Args:
            url: Website URL
        
        Returns:
            Dictionary containing detected technologies
        """
        self.logger.info(f"Detecting technology stack for {url}")
        
        results = {
            "url": url,
            "technologies": [],
            "headers": {},
            "timestamp": time.time()
        }
        
        try:
            # Stealthy delay before HTTP request
            time.sleep(random.uniform(0.5, 1.5))
            response = requests.get(url, timeout=5, verify=False)
            headers = response.headers
            
            results["headers"] = dict(headers)
            
            # Detect technologies from headers
            tech_indicators = {
                "Server": headers.get("Server", ""),
                "X-Powered-By": headers.get("X-Powered-By", ""),
                "X-AspNet-Version": headers.get("X-AspNet-Version", ""),
                "X-Framework": headers.get("X-Framework", "")
            }
            
            for header_name, header_value in tech_indicators.items():
                if header_value:
                    results["technologies"].append({
                        "name": header_value,
                        "source": header_name
                    })
            
            # Detect from HTML content
            html_content = response.text.lower()
            
            if "wordpress" in html_content:
                results["technologies"].append({"name": "WordPress", "source": "HTML"})
            if "drupal" in html_content:
                results["technologies"].append({"name": "Drupal", "source": "HTML"})
            if "joomla" in html_content:
                results["technologies"].append({"name": "Joomla", "source": "HTML"})
            if "react" in html_content or "reactjs" in html_content:
                results["technologies"].append({"name": "React", "source": "HTML"})
            if "jquery" in html_content:
                results["technologies"].append({"name": "jQuery", "source": "HTML"})
            
        except Exception as e:
            self.logger.error(f"Tech stack detection failed: {e}")
            results["error"] = str(e)
        
        return results
    
    def search_certificate_transparency(self, domain: str) -> Dict:
        """
        Search certificate transparency logs for subdomains
        
        Args:
            domain: Domain to search
        
        Returns:
            Dictionary containing discovered subdomains from CT logs
        """
        self.logger.info(f"Searching certificate transparency logs for {domain}")
        
        results = {
            "domain": domain,
            "subdomains": [],
            "certificates": [],
            "timestamp": time.time()
        }
        
        # Note: This would require API access to CT log services
        # This is a placeholder for real implementation
        results["note"] = "Certificate transparency search requires API access to CT log services"
        
        return results
    
    def gather_social_media_info(self, company_name: str) -> Dict:
        """
        Gather social media information
        
        Args:
            company_name: Company name to search
        
        Returns:
            Dictionary containing social media profiles
        """
        self.logger.info(f"Gathering social media info for {company_name}")
        
        results = {
            "company": company_name,
            "profiles": {},
            "timestamp": time.time()
        }
        
        # Note: This would require API access to social media platforms
        # This is a placeholder for real implementation
        results["note"] = "Social media information gathering requires API access"
        
        return results
    
    def _get_whois(self, domain: str) -> Dict:
        """Get WHOIS information"""
        try:
            # Use whois command
            result = subprocess.run(
                ["whois", domain],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            whois_data = {}
            for line in result.stdout.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower()
                    value = value.strip()
                    if key and value:
                        if key not in whois_data:
                            whois_data[key] = []
                        whois_data[key].append(value)
            
            return whois_data
            
        except Exception as e:
            self.logger.debug(f"WHOIS lookup failed: {e}")
            return {}
    
    def _discover_subdomains(self, domain: str) -> List[str]:
        """Discover subdomains"""
        subdomains = []
        
        # Common subdomain prefixes
        prefixes = ["www", "mail", "ftp", "admin", "test", "dev", "staging", "api", "app"]
        
        for i, prefix in enumerate(prefixes):
            # Stealthy delay between DNS queries
            if i > 0:
                delay = random.uniform(0.3, 1.0)  # Random delay 0.3-1 seconds
                time.sleep(delay)
            
            subdomain = f"{prefix}.{domain}"
            try:
                socket.gethostbyname(subdomain)
                subdomains.append(subdomain)
            except Exception:
                pass
        
        return subdomains
    
    def get_intel_data(self) -> Dict:
        """Get all gathered intelligence data"""
        return self.intel_data

