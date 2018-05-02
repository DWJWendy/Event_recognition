#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 18-4-29 下午9:20
@Author  : 毛毛虫_wendy
@E-mail  : dengwenjun818@gmail.com
@blog    : mmcwendy.info
@File    : predict.py
"""

from Program.Preprocess_2 import datapreprocess
import pandas as pd
from Program.Model_2 import model
from sklearn.metrics.pairwise import cosine_similarity

event_map = {1: "Recruiting", 2: "Cooperation", 3: "Research", 4: "Promotion", 5: "Sale", 6: "Null"}


def compute_event_cosine(item_x, event_vec, shreshold=0.8):
    """
    计算每条微博与每类事件的相似度,选择相似度最大,如果相似度最大且对于阈值,则加入对应的企业事件,否则为null事事件
    :param item_x: np.array类型, 每一条微博的向量
    :param event_vec: 每一个事件的中心向量
    :param shreshold: 阈值（默认0.7）
    :return:
    """
    cosine_simi = []
    for e in sorted(event_vec.keys()):
        cosine_simi.append(cosine_similarity(model.do_format(item_x), event_vec[e].reshape(1, -1))[0][0])
    max_value = max(cosine_simi)
    if max_value >= shreshold:
        return cosine_simi.index(max_value) + 1
    else:
        return 6


def predict_label(item, event_vec):
    """
    预测语料库中所有微博的事件
    :param item: 语料库表征数据
    :param event_vec: 事件向量
    :return:
    """
    test_x = item.copy()
    predict_result = test_x.copy()
    predict_result["predict_label"] = test_x["Sentence_vec"]. \
        apply(lambda x: compute_event_cosine(x, event_vec))
    predict_result = predict_result[["Weibo_id", "predict_label"]]
    datapreprocess.write_file("../Data_2/predict/corpora_predict_result.csv", item_data=predict_result)
    return predict_result


def map_predict_data(raw_data, predict):
    raw_data.index = list(raw_data["Weibo_id"])
    raw_data = raw_data[["Company", "Time", "Comment", "Like", "Transfer"]]
    predict.index = list([predict["Weibo_id"]])
    predict = predict[["predict_label"]]
    with open("../Data_2/predict/final_predict_result.txt", "w", encoding="utf-8", ) as f:
        for i in predict.index:
            f.write(str(i[0]) + "\t" + raw_data.loc[i[0], "Company"] + "\t" + raw_data.loc[i[0], "Time"] +
                    "\t" + event_map[int(predict.loc[i[0], "predict_label"])] + "\n")
    #
    #     weibo_id = i[0]
    #     predict.loc[weibo_id, "Company"] = raw_data.loc[weibo_id, "Company"]
    #     predict.loc[weibo_id, "Time"] = raw_data.loc[weibo_id, "Time"]
    #     predict.loc[weibo_id, "Comment"] = raw_data.loc[weibo_id, "Comment"]
    #     predict.loc[weibo_id, "Like"] = raw_data.loc[weibo_id, "Like"]
    #     predict.loc[weibo_id, "Transfer"] = raw_data.loc[weibo_id, "Transfer"]
    # datapreprocess.write_file("../Data_2/predict/final_predict_result.csv", item_data=predict)


if __name__ == "__main__":
    # train_path = "../Data_2/pre_process/5_150_1_target_represent.csv"
    # corpora_split_path = "../Data_2/pre_process/5_150_1_sentence_represent.csv"
    # model_path = "../Data_2/pre_process/window_5_size_150_sg_1.vec.model"
    # test_data_path = "../Data_2/pre_process/test_data.csv"
    # test_label_path = "../Data_2/pre_process/test_data_label.csv"
    # trigger = model.Trigger(train_path=train_path, corpora_split_path=corpora_split_path,
    #                   model_path=model_path, test_data_path=test_data_path)
    # event_vec = trigger.run(weight_1=True, weight_2=True, threshold=0.7, top_1=400, top_2=1000,
    #                         enrich=True, source_from=1, new=False)
    # item_data = datapreprocess.read_file("../Data_2/pre_process/5_150_1_sentence_represent.csv")
    # predict_label(item=item_data, event_vec=event_vec)
    raw_data = datapreprocess.read_file(path="../Data_2/raw/clean_company_corpora.csv")
    predict_data = datapreprocess.read_file(path="../Data_2/predict/corpora_predict_result.csv")
    map_predict_data(raw_data, predict_data)
