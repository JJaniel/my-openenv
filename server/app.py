from openenv.core.env_server import create_fastapi_app
from server.environment import EmailTriageEnv
from models import EmailAction, EmailObservation
import uvicorn

# Initialize the FastAPI app
app = create_fastapi_app(
    env=EmailTriageEnv,
    action_cls=EmailAction,
    observation_cls=EmailObservation,
)

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Email Triage OpenEnv is running"}

def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
