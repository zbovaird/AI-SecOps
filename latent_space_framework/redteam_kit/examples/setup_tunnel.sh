#!/usr/bin/env python3
"""
Quick setup script for public tunneling options
"""

import subprocess
import sys

def check_and_install_ngrok():
    """Check for ngrok and provide installation instructions"""
    print("="*80)
    print("Setting up Public Tunnel for Exfiltration Logger")
    print("="*80)
    print("\nOption 1: ngrok (Recommended - Free with signup)")
    print("  1. Sign up at: https://ngrok.com/signup")
    print("  2. Install: brew install ngrok (or download from ngrok.com)")
    print("  3. Authenticate: ngrok config add-authtoken YOUR_TOKEN")
    print("  4. Run: ngrok http 5001")
    print("\nOption 2: Cloudflare Tunnel (Free, no signup needed)")
    print("  1. Install: brew install cloudflare/cloudflare/cloudflared")
    print("  2. Run: cloudflared tunnel --url http://localhost:5001")
    print("\nOption 3: localhost.run (SSH tunnel - Free)")
    print("  1. Run: ssh -R 80:localhost:5001 localhost.run")
    print("\nOption 4: serveo.net (SSH tunnel - Free)")
    print("  1. Run: ssh -R 80:localhost:5001 serveo.net")
    print("="*80)

if __name__ == "__main__":
    check_and_install_ngrok()


