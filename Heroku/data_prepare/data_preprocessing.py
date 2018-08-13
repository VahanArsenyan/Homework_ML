import pandas as pd
import os
import pickle as serial


class DataPreprocessing:

    def __init__(self):
        self.file_paths = os.listdir("data")
        self.places = {}
        self.data: pd.DataFrame = None
        self.features = ["full_sq", "life_sq", "timestamp", "floor",
                         "max_floor", "build_year", "num_room", "product_type", "sub_area", "price_doc"]
        self.categorical = ["product_type", "sub_area"]
        self.categories = {}
        self.recovery_file = "data_characteristics/data_state.d"

    def load_data(self):
        parts = list(map(lambda x: pd.read_csv(os.path.join('../data', x), header=0, index_col=None), self.file_paths))
        self.data = pd.concat(parts, axis=0, sort=False)

    def get_useful_features(self):
        self.data = self.data[self.features]

    def filter_nones(self):
        self.data.dropna(how="any", inplace=True)

    def filter_outliers(self):
        self.data = self.data.loc[(self.data['build_year'] < 2017) & (self.data['build_year'] > 1930)]

    def date_preprocessing(self):
        self.data["timestamp"] = pd.to_datetime(self.data["timestamp"], infer_datetime_format=True)
        self.data['timestamp'] = (self.data['timestamp'].dt.year * 12 + self.data['timestamp'].dt.month) / 4

    def store_categories(self):
        for category in self.categorical:
            self.categories[category] = self.data[category].dropna().unique()

    def write_stat(self):
        with open(self.recovery_file, mode="wb") as file:
            serial.dump(self.features, file)
            serial.dump(self.categorical, file)
            serial.dump(self.categories, file)

    def recover_stat(self):
        with open(self.recovery_file, mode="rb") as file:
            return serial.load(file), serial.load(file), serial.load(file)

    def cat_to_int(self):
        for category in self.categorical:
            rep_map = {k: v for v, k in enumerate(self.categories[category])}
            self.data.replace(rep_map, inplace=True)

    def type_correction(self):
        self.data["timestamp"] = self.data["timestamp"].astype("int")
        self.data["floor"] = self.data["floor"].astype("int")
        self.data["max_floor"] = self.data["max_floor"].astype("int")

    def total_setup(self):
        self.load_data()
        self.store_categories()
        self.get_useful_features()
        self.filter_nones()
        self.filter_outliers()
        self.cat_to_int()
        self.date_preprocessing()
        self.type_correction()


if __name__ == "__main__":
    test = DataPreprocessing()
    test.total_setup()
    test.write_stat()
