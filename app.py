import json
import os
import shutil
import requests

import gradio as gr
from huggingface_hub import Repository
from text_generation import Client

from share_btn import community_icon_html, loading_icon_html, share_js, share_btn_css

HF_TOKEN = os.environ.get("HF_TOKEN", None)

API_URL = "https://api-inference.huggingface.co/models/bigcode/starcoder"
API_URL_BASE ="https://api-inference.huggingface.co/models/bigcode/starcoderbase"
API_URL_PLUS = "https://api-inference.huggingface.co/models/bigcode/starcoderplus"

FIM_PREFIX = "<fim_prefix>"
FIM_MIDDLE = "<fim_middle>"
FIM_SUFFIX = "<fim_suffix>"

FIM_INDICATOR = "<FILL_HERE>"

FORMATS = """## Model Formats

The model is pretrained on code and is formatted with special tokens in addition to the pure code data,\
such as prefixes specifying the source of the file or tokens separating code from a commit message.\
Use these templates to explore the model's capacities:

### 1. Prefixes 🏷️
For pure code files, use any combination of the following prefixes:

```
<reponame>REPONAME<filename>FILENAME<gh_stars>STARS\ncode<|endoftext|>
```
STARS can be one of: 0, 1-10, 10-100, 100-1000, 1000+

### 2. Commits 💾
The commits data is formatted as follows:

```
<commit_before>code<commit_msg>text<commit_after>code<|endoftext|>
```

### 3. Jupyter Notebooks 📓
The model is trained on Jupyter notebooks as Python scripts and structured formats like:

```
<start_jupyter><jupyter_text>text<jupyter_code>code<jupyter_output>output<jupyter_text>
```

### 4. Issues 🐛
We also trained on GitHub issues using the following formatting:
```
<issue_start><issue_comment>text<issue_comment>...<issue_closed>
```

### 5. Fill-in-the-middle 🧩
Fill in the middle requires rearranging the model inputs. The playground handles this for you - all you need is to specify where to fill:
```
code before<FILL_HERE>code after
```
"""

theme = gr.themes.Monochrome(
    primary_hue="indigo",
    secondary_hue="blue",
    neutral_hue="slate",
    radius_size=gr.themes.sizes.radius_sm,
    font=[
        gr.themes.GoogleFont("Open Sans"),
        "ui-sans-serif",
        "system-ui",
        "sans-serif",
    ],
)

client = Client(
    API_URL,
    headers={"Authorization": f"Bearer {HF_TOKEN}"},
)
client_base = Client(
    API_URL_BASE, headers={"Authorization": f"Bearer {HF_TOKEN}"},
)
client_plus = Client(
    API_URL_PLUS, headers={"Authorization": f"Bearer {HF_TOKEN}"},
)

def generate(
    prompt, temperature=0.9, max_new_tokens=256, top_p=0.95, repetition_penalty=1.0, version="StarCoder",
):

    temperature = float(temperature)
    if temperature < 1e-2:
        temperature = 1e-2
    top_p = float(top_p)
    fim_mode = False

    generate_kwargs = dict(
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        do_sample=True,
        seed=42,
    )

    if FIM_INDICATOR in prompt:
        fim_mode = True
        try:
            prefix, suffix = prompt.split(FIM_INDICATOR)
        except:
            raise ValueError(f"Only one {FIM_INDICATOR} allowed in prompt!")
        prompt = f"{FIM_PREFIX}{prefix}{FIM_SUFFIX}{suffix}{FIM_MIDDLE}"

    if version == "StarCoder":
        stream = client.generate_stream(prompt, **generate_kwargs)
    elif version == "StarCoderPlus":
        stream = client_plus.generate_stream(prompt, **generate_kwargs)
    else:
        stream = client_base.generate_stream(prompt, **generate_kwargs)

    if fim_mode:
        output = prefix
    else:
        output = prompt

    previous_token = ""
    for response in stream:
        if response.token.text == "<|endoftext|>":
            if fim_mode:
                output += suffix
            else:
                return output
        else:
            output += response.token.text
        previous_token = response.token.text
        yield output
    return output


examples = [
    "X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.1)\n\n# Train a logistic regression model, predict the labels on the test set and compute the accuracy score",
    "// Returns every other value in the array as a new array.\nfunction everyOther(arr) {",
    "Poor English: She no went to the market. Corrected English:",
    "def alternating(list1, list2):\n   results = []\n   for i in range(min(len(list1), len(list2))):\n       results.append(list1[i])\n       results.append(list2[i])\n   if len(list1) > len(list2):\n       <FILL_HERE>\n   else:\n       results.extend(list2[i+1:])\n   return results",
]


def process_example(args):
    for x in generate(args):
        pass
    return x


css = ".generating {visibility: hidden}"

monospace_css = """
#q-input textarea {
    font-family: monospace, 'Consolas', Courier, monospace;
}
"""


css += share_btn_css + monospace_css + ".gradio-container {color: black}"


description = """
<div style="text-align: center;">
    <h1> ⭐ StarCoder <span style='color: #e6b800;'>Models</span> Playground</h1>
</div>
<div style="text-align: left;">
    <p>This is a demo to generate text and code with the following StarCoder models:</p>
    <ul>
        <li><a href="https://huggingface.co/bigcode/starcoderplus" style='color: #e6b800;'>StarCoderPlus</a>: A finetuned version of StarCoderBase on English web data, making it strong in both English text and code generation.</li>
        <li><a href="https://huggingface.co/bigcode/starcoderbase" style='color: #e6b800;'>StarCoderBase</a>: A code generation model trained on 80+ programming languages, providing broad language coverage for code generation tasks.</li>
        <li><a href="https://huggingface.co/bigcode/starcoder" style='color: #e6b800;'>StarCoder</a>: A finetuned version of StarCoderBase specifically focused on Python, while also maintaining strong performance on other programming languages.</li>
    </ul>
    <p><b>Please note:</b> These models are not designed for instruction purposes. If you're looking for instruction or want to chat with a fine-tuned model, you can visit the <a href="https://huggingface.co/spaces/HuggingFaceH4/starchat-playground">StarChat Playground</a>.</p>
</div>
"""
disclaimer = """⚠️<b>Any use or sharing of this demo constitues your acceptance of the BigCode [OpenRAIL-M](https://huggingface.co/spaces/bigcode/bigcode-model-license-agreement) License Agreement and the use restrictions included within.</b>\
 <br>**Intended Use**: this app and its [supporting model](https://huggingface.co/bigcode) are provided for demonstration purposes; not to serve as replacement for human expertise. For more details on the model's limitations in terms of factuality and biases, see the [model card.](hf.co/bigcode)"""

with gr.Blocks(theme=theme, analytics_enabled=False, css=css) as demo:
    with gr.Column():
        gr.Markdown(description)
        with gr.Row():
            version = gr.Dropdown(
                        ["StarCoderPlus", "StarCoderBase", "StarCoder"],
                        value="StarCoder",
                        label="Model",
                        info="Choose a model from the list",
                        )
        with gr.Row():
            with gr.Column():
                instruction = gr.Textbox(
                    placeholder="Enter your code here",
                    lines=5,
                    label="Input",
                    elem_id="q-input",
                )
                submit = gr.Button("Generate", variant="primary")
                output = gr.Code(elem_id="q-output", lines=30, label="Output")
                with gr.Row():
                    with gr.Column():
                        with gr.Accordion("Advanced settings", open=False):
                            with gr.Row():
                                column_1, column_2 = gr.Column(), gr.Column()
                                with column_1:
                                    temperature = gr.Slider(
                                        label="Temperature",
                                        value=0.2,
                                        minimum=0.0,
                                        maximum=1.0,
                                        step=0.05,
                                        interactive=True,
                                        info="Higher values produce more diverse outputs",
                                    )
                                    max_new_tokens = gr.Slider(
                                        label="Max new tokens",
                                        value=256,
                                        minimum=0,
                                        maximum=8192,
                                        step=64,
                                        interactive=True,
                                        info="The maximum numbers of new tokens",
                                    )
                                with column_2:
                                    top_p = gr.Slider(
                                        label="Top-p (nucleus sampling)",
                                        value=0.90,
                                        minimum=0.0,
                                        maximum=1,
                                        step=0.05,
                                        interactive=True,
                                        info="Higher values sample more low-probability tokens",
                                    )
                                    repetition_penalty = gr.Slider(
                                        label="Repetition penalty",
                                        value=1.2,
                                        minimum=1.0,
                                        maximum=2.0,
                                        step=0.05,
                                        interactive=True,
                                        info="Penalize repeated tokens",
                                    )
                                    
                gr.Markdown(disclaimer)
                with gr.Group(elem_id="share-btn-container"):
                    community_icon = gr.HTML(community_icon_html, visible=True)
                    loading_icon = gr.HTML(loading_icon_html, visible=True)
                    share_button = gr.Button(
                        "Share to community", elem_id="share-btn", visible=True
                    )
                gr.Examples(
                    examples=examples,
                    inputs=[instruction],
                    cache_examples=False,
                    fn=process_example,
                    outputs=[output],
                )
                gr.Markdown(FORMATS)

    submit.click(
        generate,
        inputs=[instruction, temperature, max_new_tokens, top_p, repetition_penalty, version],
        outputs=[output],
    )
    share_button.click(None, [], [], _js=share_js)
demo.queue(concurrency_count=16).launch(debug=True)