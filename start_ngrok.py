"""
EcoSense AI Server - Ngrok Public HTTPS Tunnel Launcher.

Provides instant, rock-solid public HTTPS URL for http://127.0.0.1:8000.
Get free authtoken (takes 20 sec): https://dashboard.ngrok.com/get-started/your-authtoken
"""
import sys
import time
from pyngrok import ngrok, conf

def main():
    if len(sys.argv) > 1:
        token = sys.argv[1].strip()
        ngrok.set_auth_token(token)
    
    try:
        # Kill any old lingering tunnels
        ngrok.kill()
        
        # Connect tunnel to local port 8000
        tunnel = ngrok.connect("127.0.0.1:8000")
        public_url = tunnel.public_url.replace("http://", "https://")
        
        print("\n=======================================================")
        print(f"LIVE PUBLIC HTTPS URL: {public_url}")
        print(f"DOCS URL: {public_url}/docs")
        print("=======================================================")
        print("🟢 TUNNEL IS LIVE & RUNNING! Keep this window open.\n")
        
        # Keep process alive so tunnel stays active
        ngrok_process = ngrok.get_ngrok_process()
        ngrok_process.proc.wait()
    except KeyboardInterrupt:
        print("\nTunnel stopped by user.")
    except Exception as err:
        print(f"\n[ngrok error] {err}")
        print("\nTo fix: Get your free authtoken from https://dashboard.ngrok.com/get-started/your-authtoken")
        print("Then run: python start_ngrok.py YOUR_AUTHTOKEN\n")

if __name__ == "__main__":
    main()
