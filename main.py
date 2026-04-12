import uvicorn
from server.app import app

def main():
    print("Starting Email Triage Environment Server...")
    print("Local access: http://localhost:7860")
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
