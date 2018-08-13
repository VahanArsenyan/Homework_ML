import lightgbm as lgb
from data_prepare.data_preprocessing import DataPreprocessing


class Model:

    def __init__(self, num_trees=150, num_leaves=15, max_depth=7):
        self.params = {"num_trees": num_trees, "num_threads": 8, 'num_leaves': num_leaves,
                       'objective': 'regression', 'device': "cpu",
                       'metric': 'lr_root', "verbosity": 3, "tree_learner": "voting",
                       "max_depth": max_depth}
        self.model = None
        self.model_path = "pred.model"

    def train(self):
        pre_pro = DataPreprocessing()
        pre_pro.total_setup()
        train_data = lgb.Dataset(pre_pro.data.iloc[:, :-1], label=pre_pro.data.iloc[:, -1])
        self.model = lgb.train(self.params, train_data, 100,
                               categorical_feature=pre_pro.categorical + ["floor", "timestamp", "max_floor"])
        return self

    def save(self):
        self.model.save_model(self.model_path)
        return self

    def load(self):
        self.model = lgb.Booster(model_file='model/pred.model')


if __name__ == "__main__":
    Model().train().save()