import os
from google import genai
from google.genai import types
import gradio as gr


# Import environment variables (only relevant locally)
from dotenv import load_dotenv
# load .env file with API keys
load_dotenv(dotenv_path="./.env")

MODEL = "gemini-3-pro-image-preview"
CONTEXT_FOLDER = "./images/context"

client = genai.Client()


def load_local_image(file_path: str) -> types.Part:
    """Load a local image file and return it as a types.Part object."""
    # TODO: fail on other file types
    mime_type = (
        "image/jpeg" if file_path.lower().endswith((".jpg", ".jpeg"))
        else "image/png"
    )

    with open(file_path, "rb") as f:
        image_bytes = f.read()
    return types.Part.from_bytes(
        data=image_bytes,
        mime_type=mime_type,
    )


def generate_image(prompt_text: str, input_image: types.Part, context_images: list[types.Part]) -> types.GenerateContentResponse:
    return client.models.generate_content(
        model=MODEL,
        contents=[prompt_text, input_image, *context_images],
        config=types.GenerateContentConfig(
            response_modalities=['IMAGE'],
            image_config=types.ImageConfig(
                aspect_ratio="16:9",
                image_size="2K"
            ),
        )
    )

def load_prompt() -> str:
    with open("./prompt.txt", "r") as file:
        prompt_text = file.read()
    return prompt_text

def store_image(response: types.GenerateContentResponse, output_path: str):
    """Store the generated image from the response to the specified output path."""
    for part in response.parts:
        if part.text is not None:
            print(part.text)
        elif part.inline_data is not None:
            image = part.as_image()
            image.save(output_path)


PROMPT_TEXT = load_prompt()

CONTEXT_IMAGES = [
    load_local_image(f"{CONTEXT_FOLDER}/{filename}")
    for filename in os.listdir(CONTEXT_FOLDER)
    if filename.endswith((".jpg", ".jpeg", ".png"))
]


def process_image(input_image_gr: gr.Image) -> gr.Image:
    if input_image_gr is None:
        return "Please upload an image."

    print("Loading image...")
    input_image = load_local_image(input_image_gr)

    print("Generating image...")
    response = generate_image(PROMPT_TEXT, input_image, CONTEXT_IMAGES)

    print("Storing generated image...")
    store_image(response, "./generated_image.png")
    
    return gr.Image("./generated_image.png")


# Gradio interface setup
with gr.Blocks() as app:
    
    with gr.Row(height="60px"):
        
        gr.Text("Ben je in Baarn, maar ligt je hart nog in Terschelling? <br />Upload je foto en waan je even op het eiland!")
        map = gr.Image(label="Map Image", value="./images/frontend/map.jpg", height="50px", interactive=False)
        query_button = gr.Button("Terschellingify")

    with gr.Row(height="fit"):
        image_upload = gr.Image(sources=["upload"], label="Upload image", height="80%", type="filepath")
        # image_upload = gr.File(file_types=["image"], file_count="single", label="Upload image")
        output = gr.Image(label="Terschellingified")

    query_button.click(
        process_image, 
        inputs=[image_upload], 
        outputs=[output]
    )


if __name__ == "__main__":
    app.launch()
