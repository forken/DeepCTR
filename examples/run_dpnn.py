import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from deepctr.models import DPNN
from deepctr.models import DIN
from deepctr.inputs import SparseFeat,VarLenSparseFeat,DenseFeat,get_feature_names


def get_dpnn_xy_fd(data):
    feature_columns = [SparseFeat('item', 50000, embedding_dim=16,use_hash=True)]
    feature_columns += [VarLenSparseFeat(SparseFeat('hist_p_item', 50000, embedding_dim=16, use_hash=True,embedding_name='item'), maxlen=20)]
    feature_columns += [VarLenSparseFeat(SparseFeat('hist_n_item', 50000, embedding_dim=16, use_hash=True,embedding_name='item'), maxlen=20)]
    behavior_feature_list = ["item"]

    feature_dict = {'item': data['itemid'], 'hist_p_item': np.stack(data['itemid_pre_liked'].values),'hist_n_item': np.stack(data['itemid_pre_nliked'].values)}
    x = {name:feature_dict[name] for name in get_feature_names(feature_columns)}
    y = data['liked'].values
    return x, y, feature_columns, behavior_feature_list

def get_din_xy_fd(data):
    feature_columns = [SparseFeat('item', 50000, embedding_dim=16,use_hash=True)]
    feature_columns += [VarLenSparseFeat(SparseFeat('hist_item', 50000, embedding_dim=16,use_hash=True,embedding_name='item'),maxlen=20)]
    behavior_feature_list = ["item"]
    feature_dict = {'item': data['itemid'], 'hist_item': np.stack(data['itemid_pre_liked'].values)}

    x = {name:feature_dict[name] for name in get_feature_names(feature_columns)}
    y = data['liked'].values
    return x, y, feature_columns, behavior_feature_list


if __name__ == "__main__":
    data = pd.read_pickle('/Users/forken/anaconda3/notebook/DeepCtr/DataProcessing/ecommerce_cvr.pkl')
    train, test = train_test_split(data, test_size=0.1)

    train_x, train_y, feature_columns, behavior_feature_list = get_dpnn_xy_fd(train)
    test_x, test_y, _, _ = get_dpnn_xy_fd(test)
    model = DPNN(feature_columns, behavior_feature_list,l2_reg_embedding=1e-6)
    model.compile('adam', 'binary_crossentropy', metrics=['binary_crossentropy'])
    history = model.fit(train_x, train_y, verbose=1, epochs=2, validation_split=0.1)
    pred_ans = model.predict(test_x, batch_size=256)
    print("DPNN w/ posw test LogLoss", round(log_loss(test_y, pred_ans), 4))
    print("DPNN w/ posw test AUC", round(roc_auc_score(test_y, pred_ans), 4))

    model2 = DPNN(feature_columns, behavior_feature_list, l2_reg_embedding=1e-6, pos_weighted=False)
    model2.compile('adam', 'binary_crossentropy', metrics=['binary_crossentropy'])
    history = model2.fit(train_x, train_y, verbose=1, epochs=2, validation_split=0.1)
    pred_ans = model2.predict(test_x, batch_size=256)
    print("DPNN w/o posw test LogLoss", round(log_loss(test_y, pred_ans), 4))
    print("DPNN w/o posw test AUC", round(roc_auc_score(test_y, pred_ans), 4))

    din_train_x, din_train_y, din_feature_columns, din_behavior_feature_list = get_din_xy_fd(train)
    din_test_x, din_test_y, _, _ = get_din_xy_fd(test)
    din = DIN(din_feature_columns, din_behavior_feature_list,l2_reg_embedding=1e-6)
    din.compile('adam', 'binary_crossentropy', metrics=['binary_crossentropy'])
    din_history = din.fit(din_train_x, din_train_y, verbose=1, epochs=2, validation_split=0.1)
    din_pred_ans = din.predict(din_test_x, batch_size=256)
    print("DIN test LogLoss", round(log_loss(din_test_y, din_pred_ans), 4))
    print("DIN test AUC", round(roc_auc_score(din_test_y, din_pred_ans), 4))