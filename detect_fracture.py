from flask import Flask, flash, request, redirect, render_template
from werkzeug.utils import secure_filename
from os import system

from wtforms import SelectMultipleField, SubmitField, SelectField

from radtorch.settings import *
import psutil
from flask_wtf import FlaskForm

path_model = "models/ELBOW/XR_ELBOW.pkl"
path_upload = './uploads/'
model_names = [""]
# path_output = '/output/'
# save_path = path_output+datetime.now().strftime("%d%m%Y-%H%M%S")+'.png'
# save_path = "output_"+datetime.now().strftime("%d%m%Y-%H%M%S")+'.png'
save_path = "output.png"
# system("mkdir -p output")
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
    # choices = SelectMultipleField(
    #     u'Select Model', choices=models_list)
    # model = SelectField('Model', choices=models_list, default='Neural Network')
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

@app.route('/<body_part_no>', methods=['GET', 'POST'])
def elbow(body_part_no):
    print(f"{Parts[body_part_no]} Selected")
    return f"<h1>{Parts[body_part_no]} Selected!</h1>"


# @app.route('/1', methods=['GET', 'POST'])
# def elbow():
#     print("NN Selected")
#     return "<h1>Neural Network Selected!</h1>"
#
#
# @app.route('/2', methods=['GET', 'POST'])
# def logistic_regression():
#     print("NN Selected")
#     return "<h1>Logistic Regression Selected!</h1>"
#
#
# @app.route('/3', methods=['GET', 'POST'])
# def random_forest():
#     print("NN Selected")
#     return "<h1>Random Forest Selected!</h1>"


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    global file
    form = IndexForm()
    form.part.choices = ['ELBOW', 'FINGER', 'FOREARM', 'HAND', 'HUMERUS', 'SHOULDER', 'WRIST']
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

            idx, prob, encoded = detect(path_image=file_path, path_model=path_model)
            # result_dict = {"Neural Network": {"id": idx}}
            result_class = "Positive" if idx == 1 else "Negative"

            return render_template("result.html",
                                   result_class=result_class,
                                   prob=prob,
                                   encoded=encoded,
                                   selected_part=None)


def detect(path_image, path_model):
    model = import_radtorch_model(path_model)
    return display_class_activation_map(model, path_image)


def display_class_activation_map(model, target_image_path):
    """class activation maps from a specific layer of the trained model"""
    global save_path
    return model.cam(target_image_path=target_image_path,
                     target_layer=model.trained_model.layer4[2].conv3,
                     save_path=save_path,
                     type='gradcam',
                     figure_size=(20, 7),
                     cmap='jet',
                     alpha=0.2)


def import_radtorch_model(path_model):
    return pickle.load(open(path_model, 'rb'))


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8174)
