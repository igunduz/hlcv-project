from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from PIL import Image
import numpy as np
from segformer_trainer_updated import SegformerFinetuner  # Import your SegformerFinetuner class

# Load BLIP processor and model
processor = BlipProcessor.from_pretrained("prasanna2003/blip-image-captioning")
if processor.tokenizer.eos_token is None:
    processor.tokenizer.eos_token = ''
model = BlipForConditionalGeneration.from_pretrained("prasanna2003/blip-image-captioning")

# Load your fine-tuned Segformer model
id2label = {...}  # Define your class id to label mapping
your_model = SegformerFinetuner(id2label)  # Initialize your SegformerFinetuner model
your_model.load_state_dict(torch.load("path_to_your_model_checkpoint"))  # Load your model checkpoint
your_model.eval()  # Set the model to evaluation mode

# Load and preprocess the image
image = Image.open('file_name.jpg').convert('RGB')

# Process the input image with your Segformer model
with torch.no_grad():
    input_image = your_model.prepare_image(image)
    segmentation_mask = your_model.segment_image(input_image)
    segmentation_mask = np.argmax(segmentation_mask, axis=0)  # Convert to class indices

# Create a prompt using the segmentation mask
prompt = f"Instruction: Generate a caption based on the image affordance.\nImage Affordance: {segmentation_mask}\noutput: "

# Process the input for BLIP
inputs = processor(image, prompt, return_tensors="pt")

# Generate captions
output = model.generate(**inputs, max_length=100)

# Decode and print the generated caption
generated_caption = processor.tokenizer.decode(output[0])
print("Generated Caption:", generated_caption)
