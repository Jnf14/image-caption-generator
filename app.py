from flask import Flask, request, render_template
from model import load_model, generate_caption

# Create the application instance
app = Flask(__name__)

# Load the caption model
model, tokenizer, feature_extractor = load_model()

# Create a URL route in our application for "/"
@app.route('/', methods=['GET'])
def home():
  print("GET request received")
  return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
  print("POST request received")
  # Get the image from the POST request
  image = request.files['imagefile']
  # Save the image to ./uploads
  file_path = "./images/" + image.filename
  image.save(file_path)
  # Generate the caption
  caption = generate_caption(file_path, model, tokenizer, feature_extractor)
  # Return the caption
  return render_template('index.html', caption=caption)

if __name__ == '__main__':
  # Run the application
  app.run(debug=True, port=3000)
