from transformers import AutoTokenizer, VisionEncoderDecoderModel, ViTImageProcessor
from PIL import Image


model_path = 'models/transformers/'

def load_model():
  # Load the model
  model = VisionEncoderDecoderModel.from_pretrained(model_path, local_files_only=True)
  tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
  feature_extractor = ViTImageProcessor.from_pretrained(model_path, local_files_only=True)
  return model, tokenizer, feature_extractor

def generate_caption(image_path, model, tokenizer, feature_extractor):
  image = Image.open(image_path)
  pixel_values = feature_extractor(images=[image], return_tensors="pt").pixel_values
  pixel_values = pixel_values.to("cpu")
  output_ids = model.generate(pixel_values)
  preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
  captions = [pred.strip() for pred in preds]
  return captions[0].capitalize() + "."
