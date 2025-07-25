import streamlit as st
import os
from utils import load_example_index, load_model, summarize, parse_example_file
from PIL import Image

st.markdown("""
<style>
            
    [data-testid="stHorizontalBlock"], .block-container {
        max-width: 750px;
        margin-left: auto;
        margin-right: auto;
        padding: 32px 22px 48px 22px;
        background: #F7D7DE;
        border-radius: 14px;
        box-shadow: 0 6px 36px #d8697a33;
    }
            
    body, .stApp {
        background-color: #484445 !important;
        color: #3D1B24 !important;
        font-family: 'Inter', 'Fira Mono', monospace;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #B6364A !important;
        text-align: center !important;
        letter-spacing: 0.5px;
    }

    .stButton > button {
        background: #B6364A !important;
        color: #FFF5F8 !important;
        border-radius: 5px;
        border: 1.5px solid #8A2A3A !important;
        box-shadow: 0 2px 12px #ff8c6630;
        transition: background 0.18s;
    }
    .stButton > button:hover {
        background: #8A2A3A !important;
    }
    .stTextInput>label, .stTextArea>label {
        color: #B6364A !important;
        font-weight: bold;
    }

    /* Streamlit default container overrides */
    [data-testid="stHorizontalBlock"], .stContainer {
        max-width: 750px;
        margin: 0 auto;
    }
    .stImage img {
        border: 1.5px solid #B6364A;
        background: #FFF0F3;
        border-radius: 4px;
        box-shadow: 0 2px 10px #d8697aaa;
        height: 100px !important;
        width: auto !important;
        object-fit: contain;
        display: block;
        margin: 16px auto 6px auto;
    }

    /* Remove Streamlit branding */
    #MainMenu, footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ---- Paths and Config ----
DATA_DIR = "data"
IMAGES_DIR = os.path.join(DATA_DIR, "thumbnails")
TEXTS_DIR = os.path.join(DATA_DIR, "texts")
EXAMPLES_JSON = os.path.join(DATA_DIR, "examples.json")  
MODEL_DIR = "models/model.pt"

st.set_page_config(page_title="Reddit post summarizer", layout="wide")
st.markdown("<h1>Reddit post summarizer</h1>", unsafe_allow_html=True)

@st.cache_resource
def get_model():
    return load_model(MODEL_DIR)

if 'selected_post_idx' not in st.session_state:
    st.session_state.selected_post_idx = 0

#st.markdown('<div class="main-container">', unsafe_allow_html=True)

# ---- Load posts metadata and model ----
examples = load_example_index(EXAMPLES_JSON)
tokenizer, model = get_model()

# ---- example selection gallery ----
st.write("Load an example:")
cols = st.columns(len(examples))
for idx, (example, col) in enumerate(zip(examples, cols)):
    with col:
        img_path = os.path.join(IMAGES_DIR, example['image'])
        with Image.open(img_path) as img:
            orig_width, orig_height = img.size
            new_height = 150
            new_width = int(orig_width * (new_height / orig_height))
            resized_img = img.resize((new_width, new_height))
            st.image(resized_img, caption=example['title'], use_container_width=False)

        if st.button("Choose", key=f"choose_{idx}"):
            st.session_state.selected_post_idx = idx

# ---- Main interface ----
selected_example = examples[st.session_state.selected_post_idx]
text_file_path = os.path.join(TEXTS_DIR, selected_example['text'])

post_info = parse_example_file(text_file_path)

st.subheader("Post to summarize")

topic_input = st.text_input(
    label="Subreddit name - r/",
    value=post_info['topic'],
    max_chars=50,
    help="Enter topic or subreddit"
)

title_input = st.text_input(
    label="Title",
    value=post_info['title'],
    max_chars=200,
    help="Enter the title of the post"
)

content_input = st.text_area(
    label="Post",
    value=post_info['content'],
    height=300,
    help="Edit the content text before summarizing"
)

col1, col2, col3 = st.columns([2, 2, 1])
with col2:
    summarize_button = st.button("Summarize!")

if summarize_button:
    with st.spinner("Summarizing..."):
        summary = summarize(
            content_input,
            tokenizer,
            model,
            title=title_input,
            topic=topic_input
        )
    st.subheader("Summary")
    st.markdown(
        f"""
        <div style="
            background: #fff;
            color: #3D1B24;
            border-left: 6px solid #B6364A;
            padding: 16px 18px;
            border-radius: 8px;
            margin: 24px 0 12px 0;
            font-size: 1.14rem;
            ">
            {summary}
        </div>
        """,
        unsafe_allow_html=True
    )

