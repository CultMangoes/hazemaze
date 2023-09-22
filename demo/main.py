import os

import streamlit as st
from PIL import Image

# hero = Image.open('images/hero/hero.jpeg')

st.set_page_config(page_title="HazeMaze - De-Hazing Tool", page_icon="🔥")

st.title("HazeMaze")
# st.subheader("AI-ML De-Smoking and De-Hazing for Indoor Fire Rescue")
st.subheader("AI-ML based intelligent de-smoking/de-hazing algorithm")
st.write(
    "Design and Development of AI-ML-based intelligent de-smoking/de-hazing algorithm for reproducing the real-time video of the area under fire specifically for indoor fire hazards to aid the rescue operation.")


@st.cache_resource()
def get_dehazer():
    from model import dehazer
    return dehazer


dehaze = get_dehazer()

# st.image(hero, use_column_width=True)

tab1, tab2 = st.tabs(["Image  Upload", "Video Upload"])

with tab1:
    st.camera_input(label="Upload an image")
    st.header("Upload an image")
    uploaded_file_image = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"],
                                           accept_multiple_files=True)
with tab2:
    st.header("Video Upload")
    st.write("Coming Soon!")
    uploaded_file_video = st.file_uploader("Choose a video file", type=["mp4"], accept_multiple_files=False)

submit_button = st.button("Dehaze it!")
if submit_button:
    try:
        if uploaded_file_image is not None:
            payload = {"image": uploaded_file_image}
            for file in uploaded_file_image:
                img, t = dehaze(Image.open(file))
                st.image(img, use_column_width=True)
                st.info(f"Time taken: {t:.2f} seconds")
        elif uploaded_file_video is not None:
            payload = {"video": uploaded_file_video}
        else:
            st.error("Please upload an image or video file.")
            raise Exception("No file uploaded.")
    except Exception as e:
        st.error(f"Error Uploading: {str(e)}")

hide_streamlit_style = """
                    <style>
                    # MainMenu {visibility: hidden;}
                    footer {visibility: hidden;}
                    footer:after {
                    content:'Made by Team CultMangoes'; 
                    visibility: visible;
    	            display: block;
    	            position: relative;
    	            # background-color: red;
    	            padding: 15px;
    	            top: 2px;
    	            }
                    </style>
                    """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
