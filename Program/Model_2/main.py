#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 18-4-29 下午4:39
@Author  : 毛毛虫_wendy
@E-mail  : dengwenjun818@gmail.com
@blog    : mmcwendy.info
@File    : main.py
"""

from Program.Preprocess_2 import datapreprocess
from Program.Model_2 import model
from Program.Model_2 import other_model

if __name__ == "__main__":
    # 数据预处理部分
    raw_path = "../Data_2/raw/clean_company_corpora.csv"
    target_path = "../Data_2/raw/target.csv"
    stopwords_path = "../Data_2/raw/Stopwords.txt"
    # 数据预处理参数
    window, size, sg = 5, 150, 1
    # data_process = datapreprocess.DataProcess(raw_path=raw_path, target_path=target_path,
    #                                           stop_words_path=stopwords_path)
    # data_process.run(window=window, size=size, sg=sg)

    # 事件识别
    train_path = "../Data_2/pre_process/5_150_1_target_represent.csv"
    corpora_split_path = "../Data_2/pre_process/5_150_1_sentence_represent.csv"
    model_path = "../Data_2/pre_process/window_5_size_150_sg_1.vec.model"
    test_data_path = "../Data_2/pre_process/test_data.csv"
    test_label_path = "../Data_2/pre_process/test_data_label.csv"
    trigger = model.Trigger(train_path=train_path, corpora_split_path=corpora_split_path,
                      model_path=model_path, test_data_path=test_data_path)
    weight_1, weight_2, threshold, top_1, top_2, enrich, source_from, new = True, True, 0.7, 400, 1000, False, 1, False
    classifier = model.Trigger_Classifer(train_path, corpora_split_path, test_data_path, test_label_path, model_path,
                                   weight_1,
                                   weight_2, threshold, top_1, top_2, enrich, source_from, new)
    print(classifier.run())

    # 其他模型对比
    # event_vec_path = "../Data/trigger/event_vec.csv"
    # contrast_model = other_model.Contrast_Classifer(train_x_path, train_y_path, test_x_path, test_y_path)
    # contrast_model.run()
