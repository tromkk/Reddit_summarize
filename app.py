import gradio as gr
import os
from utils import load_example_index, load_model, summarize, parse_example_file

DATA_DIR = "data"
IMAGES_DIR = os.path.join(DATA_DIR, "thumbnails")
TEXTS_DIR = os.path.join(DATA_DIR, "texts")
EXAMPLES_JSON = os.path.join(DATA_DIR, "examples.json")
MODEL_DIR = os.path.join("models", "model.pt")

# Preload model and example data
tokenizer, model = load_model(MODEL_DIR)
examples = load_example_index(EXAMPLES_JSON)

def get_example_info(idx):
    ex = examples[idx]
    img_path = os.path.join(IMAGES_DIR, ex['image'])
    text_path = os.path.join(TEXTS_DIR, ex['text'])
    post_info = parse_example_file(text_path)
    return img_path, post_info['topic'], post_info['title'], post_info['content']

def load_example(idx):
    _, topic, title, content = get_example_info(idx)
    return [topic, title, content, idx]

def do_summarize(idx, topic, title, content):
    return summarize(content, tokenizer, model, title=title, topic=topic)

css = """
.gradio-container {
    max-width: 750px;
    margin-left: auto;
    margin-right: auto;
    padding: 32px 22px 48px 22px;
    background: #F7D7DE;
    border-radius: 14px;
    box-shadow: 0 6px 36px #d8697a33;
}

body, html {
    background-color: #484445 !important;
    color: #3D1B24 !important;
    font-family: 'Inter', 'Fira Mono', monospace;
}

.gr-image img {
    width: 150px !important;
    height: 150px !important;
    object-fit: cover !important;
    border: 1.5px solid #B6364A !important;
    border-radius: 4px !important;
    box-shadow: 0 2px 10px #d8697aaa !important;
    margin: 16px auto 6px auto !important;
}

/* Button styling */
.gr-button {
    background: #B6364A !important;
    color: #FFF5F8 !important;
    border-radius: 5px;
    border: 1.5px solid #8A2A3A !important;
    box-shadow: 0 2px 12px #ff8c6630;
}
.gr-button:hover {
    background: #8A2A3A !important;
}
.gr-text-input label, .gr-text-area label {
    color: #B6364A !important;
    font-weight: bold;
}

/* Headings */
h1, h2, h3, h4, h5, h6 {
    color: #B6364A !important;
    text-align: center !important;
    letter-spacing: 0.5px;
}

#summary_output {
    color: black !important;
    background-color: #90EE90 !important;
 !important; 
    padding: 10px;
    border-radius: 8px;
}
"""

with gr.Blocks(css=css, fill_width=False) as demo:

    gr.Markdown("<h1>Reddit post summarizer</h1>")
    gr.Markdown("Load an example:")

    # Examples
    with gr.Row():
        img_buttons = []
        for i, example in enumerate(examples):
            with gr.Column():
                img_path = os.path.join(IMAGES_DIR, example['image'])
                image = gr.Image(value=img_path, label=example['title'], show_label=True, elem_classes="gr-image")
                btn = gr.Button("Choose", elem_id=f"choose_{i}")
                img_buttons.append((image, btn))

    chosen_idx = gr.State(0)

    # Main
    topic_input = gr.Textbox(label="Subreddit name - r/", max_lines=1)
    title_input = gr.Textbox(label="Title", max_lines=1)
    content_input = gr.Textbox(label="Post", lines=10)
    summarize_btn = gr.Button("Summarize!")
    summary_output = gr.HTML(elem_id="summary_output")

    # Hook up buttons to load examples
    for i, (image, btn) in enumerate(img_buttons):
        btn.click(fn=load_example, inputs=[gr.State(i)], outputs=[topic_input, title_input, content_input, chosen_idx])

    summarize_btn.click(
        fn=lambda idx, t, ti, co: do_summarize(idx, t, ti, co),
        inputs=[chosen_idx, topic_input, title_input, content_input],
        outputs=[summary_output]
    )

    # Load first example initially
    demo.load(fn=load_example, inputs=[gr.State(0)], outputs=[topic_input, title_input, content_input, chosen_idx])

if __name__ == "__main__":
    demo.launch()
