import asyncio

import streamlit as st
from PIL import Image

st.set_page_config(page_title="HazeMaze - De-Hazing Tool", page_icon="🔥")

st.title("HazeMaze")
st.subheader("AI-ML based intelligent de-smoking/de-hazing algorithm")
st.write(
    "Design and Development of AI-ML-based intelligent de-smoking/de-hazing algorithm "
    "for reproducing the real-time video of the area under fire specifically "
    "for indoor fire hazards to aid the rescue operation."
)


# @st.cache_resource()
def get_dehazers():
    from model import image_dehazer, video_dehazer
    return image_dehazer, video_dehazer


image_dehazer, video_dehazer = get_dehazers()

input_type = st.radio("Select the type of input", ("Image", "Video"), horizontal=True)

if input_type == "Image":
    st.header("Upload Images")
    uploaded_file_image = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"],
                                           accept_multiple_files=True)
    if st.button("Dehaze it!", key="image"):
        if uploaded_file_image is not None:
            for file in uploaded_file_image:
                img, t = image_dehazer(Image.open(file))
                st.image(img, use_column_width=True)
                st.info(f"Time taken: {t:.2f} seconds")
        else:
            st.error("Please upload image files")

if input_type == "Video":
    st.header("Upload Video")
    uploaded_file_video = st.file_uploader("Choose a video file", type=["mp4"], accept_multiple_files=False)
    if st.button("Dehaze it!", key="video"):
        if uploaded_file_video is not None:
            with open("temp.mp4", "wb") as f:
                f.write(uploaded_file_video.read())
            stream = video_dehazer("temp.mp4", w=100)
            widget = st.image([])
            for frame in stream:
                widget.image(frame, use_column_width=True)
        else:
            st.error("Please upload a video file")

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
