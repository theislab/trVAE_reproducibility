from random import shuffle

import numpy as np
import scanpy.api as sc
import seaborn as sns

sc.settings.verbosity = 1  # show logging output
sns.set_style("white")


class data_reader():

    def __init__(self, train_data, valid_data, conditions, tr_ct_list=None, ho_ct_list=None,
                 cell_type_key="cell_label"):

        self.conditions = conditions
        self.cell_type_key = cell_type_key
        if tr_ct_list and ho_ct_list:
            self.t_in = tr_ct_list
            self.t_out = ho_ct_list
            self.train_real = self.data_remover(train_data)
            self.train_real_adata = self.train_real
            ind_list = [i for i in range(self.train_real.shape[0])]
            shuffle(ind_list)
            self.train_real = self.train_real[ind_list, :].X
            self.valid_real_adata = self.data_remover(valid_data)
            self.valid_real = self.valid_real_adata.X

        else:
            self.train_real = train_data
            self.train_real_adata = self.train_real
            ind_list = [i for i in range(self.train_real.shape[0])]
            shuffle(ind_list)
            self.train_real = self.train_real[ind_list, :].X
            self.valid_real_adata = valid_data
            self.valid_real = valid_data.X

    def data_remover(self, adata):
        source_data = []
        for i in self.t_in:
            source_data.append(self.extractor(adata, i)[3])
        target_data = []
        for i in self.t_out:
            target_data.append(self.extractor(adata, i)[1])
        mearged_data = self.training_data_provider(source_data, target_data)
        mearged_data.var_names = adata.var_names
        return mearged_data

    def extractor(self, data, cell_type):
        cell_with_both_condition = data[data.obs[self.cell_type_key] == cell_type]
        condtion_1 = data[
            (data.obs[self.cell_type_key] == cell_type) & (data.obs["condition"].isin(self.conditions["ctrl"]))]
        condtion_2 = data[
            (data.obs[self.cell_type_key] == cell_type) & (data.obs["condition"].isin(self.conditions["stim"]))]
        training = data[
            ~((data.obs[self.cell_type_key] == cell_type) & (data.obs["condition"].isin(self.conditions["stim"])))]
        return [training, condtion_1, condtion_2, cell_with_both_condition]

    def training_data_provider(self, train_s, train_t):
        train_s_X = []
        train_s_diet = []
        train_s_groups = []
        for i in train_s:
            train_s_X.append(i.X)
            train_s_diet.append(i.obs["condition"].tolist())
            train_s_groups.append(i.obs[self.cell_type_key].tolist())
        train_s_X = np.concatenate(train_s_X)
        temp = []
        for i in train_s_diet:
            temp = temp + i
        train_s_diet = temp
        temp = []
        for i in train_s_groups:
            temp = temp + i
        train_s_groups = temp
        train_t_X = []
        train_t_diet = []
        train_t_groups = []
        for i in train_t:
            train_t_X.append(i.X)
            train_t_diet.append(i.obs["condition"].tolist())
            train_t_groups.append(i.obs[self.cell_type_key].tolist())
        temp = []
        for i in train_t_diet:
            temp = temp + i
        train_t_diet = temp
        temp = []
        for i in train_t_groups:
            temp = temp + i
        train_t_groups = temp
        train_t_X = np.concatenate(train_t_X)
        # concat_all
        train_real = np.concatenate([train_s_X, train_t_X])
        train_real = sc.AnnData(train_real)
        train_real.obs["condition"] = train_s_diet + train_t_diet
        train_real.obs[self.cell_type_key] = train_s_groups + train_t_groups
        return train_real

    def balancer(self, data):
        class_names = np.unique(data.obs[self.cell_type_key])
        class_pop = {}
        for cls in class_names:
            class_pop[cls] = len(data[data.obs[self.cell_type_key] == cls])

        max_number = np.max(list(class_pop.values()))

        all_data_x = []
        all_data_label = []
        all_data_condition = []

        for cls in class_names:
            temp = data[data.obs[self.cell_type_key] == cls]
            index = np.random.choice(range(len(temp)), max_number)
            temp_x = temp.X[index]
            all_data_x.append(temp_x)
            temp_ct = np.repeat(cls, max_number)
            all_data_label.append(temp_ct)
            temp_cc = np.repeat(np.unique(temp.obs["condition"]), max_number)
            all_data_condition.append(temp_cc)

        balanced_data = sc.AnnData(np.concatenate(all_data_x))
        balanced_data.obs[self.cell_type_key] = np.concatenate(all_data_label)
        balanced_data.obs["condition"] = np.concatenate(all_data_label)

        class_names = np.unique(balanced_data.obs[self.cell_type_key])
        class_pop = {}
        for cls in class_names:
            class_pop[cls] = len(balanced_data[balanced_data.obs[self.cell_type_key] == cls])
        # print(class_pop)
        return balanced_data
