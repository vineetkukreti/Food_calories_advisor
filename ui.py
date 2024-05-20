import streamlit as st
from roboflow import Roboflow
from PIL import Image
from openai import OpenAI
import requests
from io import BytesIO
import base64
import json

api_key = "bhai api key"

def encode_image(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def listme(image):
    base64_image = encode_image(image)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "As an AI assistant, I specialize in identifying edible items present in images. Your query will be processed with precision, listing only the edible items that are actually present in the image. I won't include anything that isn't visible in the image."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    data = response.json()
    content = data['choices'][0]['message']['content']
    return content


rf = Roboflow(api_key="xhuU24wOpb8s0EuD1Azy")
project = rf.workspace().project("ok-rve3e")
model = project.version("1").model


client = OpenAI(api_key=api_key)  # Replace with your OpenAI API key


st.set_page_config(page_title="Calories finder", page_icon="🤖", layout="centered")
st.title("Claories mapper")
st.markdown("### Upload an image to get started")
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    st.markdown("#### Uploaded Image")
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)

    l = listme(img)

    img_path = "temp_image.jpg"
    img.save(img_path)
    

    with st.spinner('Predicting...'):
        data = model.predict(img_path, confidence=40, overlap=30).json()
        item_names = [prediction['class'] for prediction in data['predictions']]
        

        model.predict(img_path, confidence=40, overlap=30).save("prediction.jpg")
        pred_img = Image.open("prediction.jpg")

    st.markdown("#### Prediction Image")
    st.image(pred_img, caption='Prediction Image', use_column_width=True)
    

    st.markdown("#### Predictions")
    st.write("The model detected the following items in the image:")
    for name in item_names:
        st.markdown(f"- **{name}**")
    

    prompt = f"Provide the calorie information for the following food items:\n{', '.join(item_names)}"
    prompt+=l
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a nutrition expert."},
            {"role": "user", "content": prompt}
        ]
    )
    calorie_info = completion.choices[0].message.content.strip()

    st.markdown("#### Calorie Information")
    st.write("Here is the calorie information for the detected food items:")
    st.write(calorie_info)

else:
    st.info("Please upload an image to see predictions.")
