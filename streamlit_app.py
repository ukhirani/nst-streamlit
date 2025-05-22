import streamlit as st
from PIL import Image
import numpy as np

st.set_page_config(
    page_title="Neural Style Transfer by Umang Hirani",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("# Neural Style Transfer")
st.markdown("###### _by Umang Hirani_")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("### Content Image")
    content_image = st.file_uploader("Upload Content Image", type=['png', 'jpg'], key='content_image')
    if content_image is not None:
        image = Image.open(content_image)
        st.image(image, use_container_width=True, output_format='auto')

with col2:
    st.markdown("### Style Image")
    style_image = st.file_uploader("Upload Style Image", type=['png', 'jpg'], key='style_image')
    if style_image is not None:
        image = Image.open(style_image)
        st.image(image, use_container_width=True, output_format='auto')

with col3:
    st.markdown("### Parameters")
    content_weight = st.slider("Content/Style Tradeoff", min_value=0.0, max_value=1.0, value=0.5, step=0.01, key='content_weight', format="%f")
    tv_loss = st.checkbox("TV Loss", value=False, key='tv_loss')


with col4:
    st.markdown("### Output Image")
    output_image = st.image([], use_container_width=True, output_format='auto')



