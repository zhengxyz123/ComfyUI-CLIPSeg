# ComfyUI-CLIPSeg

Using [CLIPSeg model](https://huggingface.co/docs/transformers/main/en/model_doc/clipseg) to generate masks for image inpainting tasks based on text prompts.

![Usage of ComfyUI-CLIPSeg](example.png)

## Installation

1. Navigate to your ComfyUI custom nodes directory:

   ```bash
   cd /path/to/your/ComfyUI/custom_nodes
   ```

2. Clone this repository:

   ```bash
   git clone git@github.com:zhengxyz123/ComfyUI-CLIPSeg.git --depth=1
   ```

3. Restart ComfyUI.

The plugin will automatically download the [CIDAS/clipseg-rd64-refined](https://huggingface.co/CIDAS/clipseg-rd64-refined) model from Hugging Face the first time you use it.

If you have installed ComfyUI properly then there is no need to install any additional dependencies.

## Usage

Once installed, you'll find a new node called `CLIPSeg` under the `mask` category in ComfyUI.

First, drag and drop the `CLIPSeg` node into your workflow.

In the text input field, enter your desired prompts. You can specify multiple elements separated by commas.

Then connect the node to your image input (e.g., Load Image, Inpainting Pipeline).

Run the workflow and a binary mask based on your text description will be generated.
