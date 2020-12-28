from flask import Flask, flash, request, redirect, render_template
from werkzeug.utils import secure_filename
from os import system

from wtforms import SelectMultipleField, SubmitField, SelectField

from radtorch.settings import *
import psutil
from flask_wtf import FlaskForm
from pprint import pprint
from json import dumps

selected_part = ""
path_upload = './uploads/'
model_names = [""]
save_path = "output.png"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '3db73ffc7894de38d4d24a342ef8dc765d821becf623e3be'
app.config['UPLOAD_FOLDER'] = path_upload


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


class IndexForm(FlaskForm):
    parts = [('ELBOW', 'ELBOW'), ('FINGER', 'FINGER'), ('FOREARM', 'FOREARM'), ('HAND', 'HAND'),
                   ('HUMERUS', 'HUMERUS'), ('SHOULDER', 'SHOULDER'), ('WRIST', 'WRIST')]
    part = SelectField('Part', choices=parts, render_kw={'onchange': "myFunction()"})
    submit = SubmitField('Submit')


Parts = {"1": "ELBOW",
         "2": "FINGER",
         "3": "FOREARM",
         "4": "HAND",
         "5": "HUMERUS",
         "6": "SHOULDER",
         "7": "WRIST"}

file = None

@app.route('/res', methods=['GET', 'POST'])
def res():
    global selected_part
    selected_part = request.data.decode("utf-8")
    return app.response_class(
			    response=dumps("Success!"),
			    status=200,
			    mimetype='application/json'
			)


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    global file, selected_part
    print("selected_part:", selected_part)
    form = IndexForm()
    form.part.choices = ['ELBOW', 'FINGER', 'FOREARM', 'HAND', 'HUMERUS', 'SHOULDER', 'WRIST']
    classifiers = ["NN", "LR", "RF"]
    scores = {}
    # if form.validate_on_submit():
    #     form = form.part.data
    if request.method == 'GET':
        return render_template('bs.html', form=form)
    elif request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            if not os.path.exists(path_upload):
                os.makedirs(path_upload)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            for classifier in classifiers:
                path_model = f"models/{selected_part}/XR_{selected_part}_{classifier}.pkl"
                model = pickle.load(open(path_model, 'rb'))
                if classifier == "NN":
                    idx, prob, encoded = display_class_activation_map(model, target_image_path=file_path)
                else:
                    idx, prob = predict(model, file_path)
                result_class = "Positive" if idx == 1 else "Negative"
                prob = round(prob, 6)
                scores.update({classifier: {"result_class": result_class, "probability": prob}})

            # result_dict = {"Neural Network": {"id": idx}}
            return render_template("result.html",
                                    scores=scores,
                                    result_class=result_class,
                                    prob=prob,
                                    encoded=encoded,
                                    selected_part=None)

def predict(model, file_path):
    df = model.classifier.predict(file_path, True)
    prob = df['PREDICTION_ACCURACY'].max()
    idx = df.loc[df['PREDICTION_ACCURACY'] == prob]['LABEL_IDX'].item()
    return idx, float(prob)

def display_class_activation_map(model, target_image_path):
    """class activation maps from a specific layer of the trained model"""
    global save_path
    try:
        target_layer=model.trained_model.layer4[2].conv3
    except:
        target_layer = None
    return model.cam(target_image_path=target_image_path,
                     target_layer=target_layer,
                     save_path=save_path,
                     type='gradcam',
                     figure_size=(20, 7),
                     cmap='jet',
                     alpha=0.2)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000)
