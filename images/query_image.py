"""
This script loads an image, sends it to Google's Gemini Pro Vision model 
along with a user question, and prints the model's response.
"""

import os
from dotenv import load_dotenv

load_dotenv()


os.environ["GOOGLE_API_KEY"]=os.getenv("GOOGLE_API_KEY")

import google.generativeai as genai
import os
import PIL.Image

def call_LMM(image_path: str, prompt: str) -> str:
    # Load the image
    img = PIL.Image.open(image_path)

    # Call generative model
    model = genai.GenerativeModel('gemini-pro-vision')
    response = model.generate_content([prompt, img], stream=True)
    response.resolve()

    return response.text

image_path = "AAPL.png"
question = """Explain what you see in this image. 
              What do the indicators D & E at the bottom of the chart refer to?
              Does this explain VWAP? """

response = call_LMM(image_path, 
                    question)
print(response)