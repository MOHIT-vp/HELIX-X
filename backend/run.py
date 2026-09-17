import uvicorn

from app.config import Settings

if __name__ == "__main__":
    settings = Settings.from_env()
    settings.validate()
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=True, log_level=settings.log_level.lower())
