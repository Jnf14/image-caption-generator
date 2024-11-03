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

# test the model
print("Testing the model...")
device = "cpu"
model.to(device)
max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
image = Image.open("images/cat.png")
pixel_values = feature_extractor(images=[image], return_tensors="pt").pixel_values
pixel_values = pixel_values.to(device)

output_ids = model.generate(pixel_values, **gen_kwargs)

preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
captions = [pred.strip() for pred in preds]

print("The given caption is: ", captions)
print("Model tested successfully!")

# save the model to local directory
print("Saving the model to local directory...")
classifier = pipeline('image-to-text', model=model, tokenizer=tokenizer, feature_extractor=feature_extractor)
classifier.save_pretrained(model_path)
print("Model saved successfully!")


# # load model from local directory if it works
# model = VisionEncoderDecoderModel.from_pretrained(model_path, local_files_only=True)
# print("-----------  model loaded from local dir ------------")
# tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
# print("-----------  tokenizer loaded from local dir ------------")
# classifier = pipeline('image-to-text' ,model=model, tokenizer=tokenizer)

# classifier(["good"])
