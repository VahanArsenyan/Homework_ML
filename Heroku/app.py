from flask import Flask, request, render_template
from data_prepare.data_preprocessing import DataPreprocessing
from datetime import datetime
from model.model_train import Model
import pandas as pd

app = Flask(__name__, template_folder="templates")
model = Model()
model.load()

data_temp = DataPreprocessing()
_, _, cats = data_temp.recover_stat()


@app.route('/', methods=["GET", "POST"])
def hello_world():
    if request.method == "POST":
        params = validate_input(request.form)
        print(params)
        if params is not None:
            res = model.model.predict(pd.DataFrame([list(params.values())], columns=list(params.keys())))
            return str(res[0])
        else:
            return "Incorrect argument, please check fields"
    elif request.method == "GET":
            return render_template("index.html", areas=cats["sub_area"], types=cats["product_type"])
    return 'Hello, World!'


def validate_input(data):
    try:
        new_data = {}

        new_data["full_sq"] = float(data["full_sq"])
        new_data["life_sq"] = float(data["life_sq"])
        p_dt = datetime.strptime(data["timestamp"], '%Y-%m-%d')
        if p_dt.year > 2017 or p_dt.year < 1930:
            raise ValueError
        new_data["timestamp"] = int((p_dt.year*12 + p_dt.month) / 4)
        new_data["floor"] = int(data["floor"])
        new_data["max_floor"] = int(data["max_floor"])
        new_data['build_year'] = float(data['build_year'])
        new_data['num_room'] = float(data['num_room'])
        if data['product_type'] not in cats['product_type']:
            raise ValueError
        new_data['product_type'] = {k: v for v, k in enumerate(cats['product_type'])}[data['product_type']]
        if data['sub_area'] not in cats['sub_area']:
            raise ValueError
        new_data['sub_area'] = {k: v for v, k in enumerate(cats['sub_area'])}[data['sub_area']]
        return new_data
    except Exception as E:
        return None


if __name__ == "__main__":
    app.run()
