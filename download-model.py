# import required libraries
from transformers import pipeline
from transformers import AutoTokenizer, VisionEncoderDecoderModel, ViTImageProcessor
from PIL import Image

# define the model name
model_name = "nlpconnect/vit-gpt2-image-captioning"

# define the model path
model_path = 'models/transformers/' # will be created automatically if not exists

# download the model
print("Downloading the model...")
model = VisionEncoderDecoderModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
print("Model downloaded successfully!")

# save the model to local directory
print("Saving the model to local directory...")
classifier = pipeline('image-to-text', model=model, tokenizer=tokenizer, feature_extractor=feature_extractor)
classifier.save_pretrained(model_path)
print("Model saved successfully!")
