from flask import Flask, render_template, request
from modules.deep_learning_OCR_func import OCR

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        upload_file = request.files['image_name']
        filename = upload_file.filename
        path_save = f'./static/uploaded/{filename}' 
        upload_file.save(path_save) # es la imagen que se sube
        text = OCR(path_save, filename)
        return render_template('index.html', upload=True, upload_image=filename, text=text)

    return render_template('index.html', upload=False)

if __name__ == '__main__':
    app.run(debug=True)