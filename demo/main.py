import cv2
import numpy as np
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
def get_image_dehazer():
    from model import image_dehazer
    return image_dehazer


@st.cache_resource()
def get_video_dehazer():
    from model import video_dehazer
    return video_dehazer


image_dehazer = get_image_dehazer()
video_dehazer = get_video_dehazer()

tab1, tab2 = st.tabs(["Image  Upload", "Video Upload"])

with tab1:
    st.header("Upload an image")
    uploaded_file_image = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"],
                                           accept_multiple_files=True)
with tab2:
    st.header("Video Upload")
    uploaded_file_video = st.file_uploader("Choose a video file", type=["mp4"], accept_multiple_files=False)

submit_button = st.button("Dehaze it!")
if submit_button:
    try:
        if uploaded_file_image is not None:
            for file in uploaded_file_image:
                img, t = image_dehazer(Image.open(file))
                st.image(img, use_column_width=True)
                st.info(f"Time taken: {t:.2f} seconds")
        # if uploaded_file_video is not None:
        #     vid = uploaded_file_video.name
            # with open(vid, mode='wb') as f:
            #     f.write(uploaded_file_video.read())
            # vid_cap = cv2.VideoCapture(vid)
            #
            # images = []
            # for i in range(24 * 6):
            #     if i % 2 != 0: continue
            #     ret, frame = vid_cap.read()
            #     if ret:
            #         images.append(Image.fromarray(frame))
            #     else:
            #         break
            # vid = video_dehazer(images)
            # vid = np.array([np.array(img) for img in vid[0]])
            # # save video
            # height, width, layers = vid[0].shape
            # size = (width, height)
            # out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 24, size)
            # for i in range(len(vid)):
            #     out.write(vid[i])
            # out.release()
            # st.video('output.mp4')
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
