"""
EcoSense AI Server - Public Cloudflare HTTPS Tunnel Launcher.

Downloads portable cloudflared.exe (100% Free, No Account Required)
and exposes local FastAPI server (http://localhost:8000) over public HTTPS.
"""
import os
import sys
import time
import urllib.request
import subprocess
from pathlib import Path

CLOUDFLARED_URL = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
EXE_PATH = Path("cloudflared.exe")

def ensure_cloudflared():
    if not EXE_PATH.exists():
        print("Downloading free Cloudflare Tunnel client (cloudflared.exe)...")
        urllib.request.urlretrieve(CLOUDFLARED_URL, EXE_PATH)
        print("Download complete.")

def run_tunnel(port: int = 8000):
    ensure_cloudflared()
    print(f"Exposing http://localhost:{port} via Cloudflare HTTPS Tunnel...")
    
    cmd = [str(EXE_PATH.resolve()), "tunnel", "--url", f"http://localhost:{port}"]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    tunnel_url = None
    for line in process.stdout:
        print(line, end="")
        if "trycloudflare.com" in line:
            parts = line.split()
            for part in parts:
                if "trycloudflare.com" in part:
                    tunnel_url = part
                    print("\n=======================================================")
                    print(f"LIVE PUBLIC HTTPS URL: {tunnel_url}")
                    print(f"DOCS URL: {tunnel_url}/docs")
                    print("=======================================================\n")
                    break

if __name__ == "__main__":
    run_tunnel(8000)
