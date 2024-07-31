# Image Retrieval using BLIP Model

This repository contains a Jupyter Notebook demonstrating the usage of the BLIP (Bootstrapped Language-Image Pretraining) model for image-text matching. The notebook leverages the `transformers` library from Hugging Face to load and use the BLIP model.

## Notebook Overview

The notebook performs the following steps:
1. **Loading the Model**: Imports and loads the BLIP model and processor.
2. **Image Loading**: Loads an image from a specified URL.
3. **Text Description**: Defines a text description for the image.
4. **Processing Inputs**: Prepares the image and text for the model.
5. **Model Inference**: Runs the model to get matching scores between the image and text.
6. **Result Interpretation**: Processes and prints the matching probability.

## Prerequisites

To run the notebook, ensure you have the following dependencies installed:
- `transformers`
- `PIL` (Python Imaging Library)
- `requests`
- `torch`

You can install the necessary libraries using:
```bash
pip install transformers pillow requests torch
```

## Running the Notebook

1. Clone the repository:
```bash
git clone https://github.com/saadtariq001s/image-retrieval.git
```

2. Navigate to the repository directory:
```bash
cd image-retrieval
```

3. Open the Jupyter Notebook:
```bash
jupyter notebook image_retrieval.ipynb
```

4. Execute the cells sequentially to see the image-text matching in action.

## Example Code

Here's a brief overview of the key sections in the notebook:

### Loading the Model
```python
from transformers import BlipForImageTextRetrieval, AutoProcessor

model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco")
processor = AutoProcessor.from_pretrained("Salesforce/blip-itm-base-coco")
```

### Loading the Image
```python
img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'
from PIL import Image
import requests

raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
raw_image
```

### Defining Text and Processing Inputs
```python
text = "an image of a woman and a dog on the beach"
inputs = processor(images=raw_image, text=text, return_tensors="pt")
```

### Model Inference and Result Interpretation
```python
itm_scores = model(**inputs)[0]
import torch

itm_score = torch.nn.functional.softmax(itm_scores, dim=1)
print(f"The image and text are matched with a probability of {itm_score[0][1]:.4f}")
```

## License

This project is licensed under the MIT License.
