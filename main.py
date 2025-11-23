import os
from google import genai
from google.genai import types


# Import environment variables (only relevant locally)
from dotenv import load_dotenv
# load .env file with API keys
load_dotenv(dotenv_path="./.env")

MODEL = "gemini-3-pro-image-preview"
CONTEXT_FOLDER = "./images/context"

client = genai.Client()


def local_image(file_path: str) -> types.Part:
    """Load a local image file and return it as a types.Part object."""
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

CONTEXT_IMAGES = [
    local_image(f"{CONTEXT_FOLDER}/{filename}")
    for filename in os.listdir(CONTEXT_FOLDER)
    if filename.endswith((".jpg", ".jpeg", ".png"))
]

def generate_image(prompt_text: str, input_image: types.Part, context_images: list[types.Part]) -> types.GenerateContentResponse:
    return client.models.generate_content(
        model=MODEL,
        contents=[prompt_text, input_image, *context_images],
        config=types.GenerateContentConfig(
            response_modalities=['IMAGE'],
        #     image_config=types.ImageConfig(
        #         aspect_ratio="1:1",
        #         image_size="1K"
        #     ),
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

def main():
    print("Loading prompt and images...")
    prompt_text = load_prompt()
    input_image = local_image("./images/input_image.jpg")
    print("Generating new image...")
    response = generate_image(prompt_text, input_image, CONTEXT_IMAGES)
    print("Storing generated image...")
    store_image(response, "./generated_image.png")
    print("Done. Generated image saved to './generated_image.png'.")

if __name__ == "__main__":
    main()
