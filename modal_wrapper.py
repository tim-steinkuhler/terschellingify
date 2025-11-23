"""
Modal deployment wrapper for the PDF query Gradio app.
"""
from fastapi import FastAPI
from gradio.routes import mount_gradio_app
import modal

from app_frontend import app as blocks  # Import Gradio app (from app_frontend.py)

# Create a lightweight Modal image (Debian-based) with required dependencies
image = (
    modal.Image.debian_slim().pip_install(
        "gradio<6",  # Use Gradio version below 6 (v6 may break compatibility)
        "google-genai"
    )
    .add_local_python_source("app_frontend")
    .add_local_dir("./images", remote_path="/root/images")
    .add_local_file("./prompt.txt", remote_path="/root/prompt.txt")
)



# Define the Modal app container
app = modal.App("terschellingify", image=image)


@app.function(
    max_containers=1,  # Only one instance (Gradio uses local file storage, preventing multiple replicas)
    secrets=[modal.Secret.from_name("gemini")]  # Fetch Gemini API key from Modal secrets
)
@modal.concurrent(max_inputs=1000) # Async handling for up to 1000 concurrent requests within a single instance
@modal.asgi_app()  # Register this as an ASGI app (compatible with FastAPI)
def serve() -> FastAPI:
    """
    Main server function: 
    - Wraps Gradio inside FastAPI 
    - Deploys the API through Modal with a single instance for session consistency
    """
    api = FastAPI(docs=True)  # Enable Swagger documentation at /docs
    return mount_gradio_app(app=api, blocks=blocks, path="/")  # Mount Gradio app at root path


@app.local_entrypoint()
def main():
    """
    Local development entry point: 
    - Allows running the app locally for testing
    - Prints the type of Gradio app to confirm readiness
    """
    print(f"{type(blocks)} is ready to go!")