from api.main import app
import uvicorn


def main():
    """Entry point for OpenEnv server"""
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()