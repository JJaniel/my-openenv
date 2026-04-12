import uvicorn
from email_triage_env.server.app import app

def main():
    print("Starting Email Triage Environment Server...")
    print("Local access: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
