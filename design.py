import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2
import numpy as np
import requests
import io
import os
import pandas as pd
import logging
from PIL import Image
import json
import base64
from dotenv import load_dotenv
import time

load_dotenv()

imgbb_api_key = os.getenv("IMG_BB_API_KEY")

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Log to file
file_handler = logging.FileHandler('api_log.log')
logger.addHandler(file_handler)

# Log to console
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)

def upload_to_imgbb(image: Image.Image, api_key: str) -> str:
    """
    Uploads an image to imgBB.

    Parameters:
        image (PIL.Image.Image): The image to upload.
        api_key (str): The API key for imgBB.

    Returns:
        str: The URL of the uploaded image.
    """
    # Convert the image to base64
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    # Set up the API request
    url = "https://api.imgbb.com/1/upload"
    payload = {
        "key": api_key,
        "image": img_str,
    }

    # Send the request
    response = requests.post(url, payload)

    # Check the response
    if response.status_code == 200:
        data = response.json()
        return data["data"]["url"]
    else:
        raise Exception(f"Failed to upload image: {response.content}")

def send_request(image_url, mask_image_url, prompt):
    with st.spinner('Processing...'):
        url = "https://stablediffusionapi.com/api/v5/interior"
        stable_diffusion_api_key = os.getenv("STABLE_DIFFUSION_API_KEY")
        payload = json.dumps({
        "key": stable_diffusion_api_key,
        "init_image" : image_url,
        "mask_image" : mask_image_url,
        "prompt" : prompt,
        "steps" : 50,
        "guidance_scale" : 7
        })
        headers = {'Content-Type': 'application/json'}  # Define headers
        
        try:
            response = requests.request("POST", url, headers=headers, data=payload)
            response.raise_for_status()
            response_data = response.json()
        except Exception as e:
            st.write("Error when sending the request: ", str(e))
            return None, None

        # Log the response
        logging.info('API Response: %s', response_data)

        if response_data.get('status') == 'processing':
            return response_data, response_data.get('eta')

        return response_data, None

def fetch_result(fetch_url):
    while True:
        try:
            response = requests.get(fetch_url)
            response.raise_for_status()
            data = response.json()

            if data['status'] == 'success':
                return data

            # If the status is not 'success', sleep for a while before retrying
            time.sleep(5)
        except Exception as e:
            st.write("Error when fetching the result: ", str(e))
            return None

st.title('Room Interior Generator')

prompt = st.text_input('What do you want to change?')

uploaded_image = st.file_uploader("Upload an image...", type=['png', 'jpg', 'jpeg'])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    imgbb_api_key = "43c34513420917713ce63e5dba7b2da6" 
    image_url = upload_to_imgbb(image, imgbb_api_key)

    uploaded_image_width, uploaded_image_height = image.size

    drawing_mode = st.sidebar.selectbox(
        "Drawing tool:",
        ("freedraw", "line", "rect", "circle", "transform", "polygon", "point"),
    )
    stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
    if drawing_mode == "point":
        point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3)
    stroke_color = st.sidebar.color_picker("Stroke color hex: ")
    bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
    realtime_update = st.sidebar.checkbox("Update in realtime", True)

    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        background_image=image,
        update_streamlit=realtime_update,
        width=uploaded_image_width,
        height=uploaded_image_height,
        drawing_mode=drawing_mode,
        display_toolbar=st.sidebar.checkbox("Display toolbar", True),
        key="canvas",
    )

    if canvas_result.image_data is not None:
        st.image(canvas_result.image_data)

        mask_image_data = np.where(canvas_result.image_data[..., 3] != 0, 255, 0).astype(np.uint8)
        mask_image = Image.fromarray(mask_image_data)

        mask_image_io = io.BytesIO()
        mask_image.save(mask_image_io, format='PNG')
        mask_image_io.seek(0)

        st.download_button(
            label="Download mask image",
            data=mask_image_io,
            file_name='mask.png',
            mime='image/png',
        )

    if canvas_result.json_data is not None:
        objects = pd.json_normalize(canvas_result.json_data["objects"])
        for col in objects.select_dtypes(include=["object"]).columns:
            objects[col] = objects[col].astype("str")
        st.dataframe(objects)

if st.button('I am ready'):
    if canvas_result.json_data is not None:
        mask_image_url = 'https://raw.githubusercontent.com/walter-grace/hosting-files/main/mask7.png'
        response, eta = send_request(image_url, mask_image_url, prompt)

        if eta is not None:
            st.write(f'Processing image in the background. Estimated time to completion: {eta} seconds.')

        if isinstance(response, dict) and response['status'] == 'processing':
            fetch_url = response['fetch_result']
            response = fetch_result(fetch_url)
            if response is None:
                st.write('Error fetching the result.')
                
        if isinstance(response, dict) and response['status'] == 'success':
            generated_image_url = response['output'][0]
            st.image(generated_image_url, caption='Generated Image', use_column_width=True)
                    
            # Download the generated image
            response = requests.get(generated_image_url)
            generated_image = Image.open(io.BytesIO(response.content))

            generated_image_io = io.BytesIO()
            generated_image.save(generated_image_io, format='PNG')
            generated_image_io.seek(0)

            st.download_button(
                label="Download generated image",
                data=generated_image_io,
                file_name='generated_image.png',
                mime='image/png',
            )
        else:
            st.write('Error generating image. Response:', response)
