from flask import Flask, request, render_template
from model import load_model, generate_caption

UPLOAD_FOLDER = './images'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Create the application instance
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit

# File extension validation
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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

  # Check if the post request has the file part
  if 'imagefile' not in request.files:
    return render_template('index.html', error='No image found.')


  # Get the image from the POST request
  image = request.files['imagefile']
  if not allowed_file(image.filename):
    return render_template('index.html', error='File type not supported.')

  # Save the image to ./uploads
  file_path = "./images/" + image.filename
  image.save(file_path)
  # Generate the caption
  caption = generate_caption(file_path, model, tokenizer, feature_extractor)
  # Return the caption
  return render_template('index.html', caption=caption)

if __name__ == '__main__':
  # Run the application
  app.run(debug=False, host='0.0.0.0', port=3000)
