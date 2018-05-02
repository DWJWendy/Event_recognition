#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 18-4-26 下午4:09
@Author  : 毛毛虫_wendy
@E-mail  : dengwenjun818@gmail.com
@blog    : mmcwendy.info
@File    : main.py
"""

from Program.Preprocess import datapreprocess
from Program.Model import model
from Program.Model import other_model

if __name__ == "__main__":
    # 数据预处理部分
    raw_path = "../Data/raw/Corpora.csv"
    label_path = "../Data/raw/Corpora_label.csv"
    stop_words_path = "../Data/raw/Stopwords.txt"
    target_train_path = "../Data/pre_process/target_train_x.csv"
    corpora_split_path = "../Data/pre_process/corpora_split_content.csv"
    train_x_path = "../Data/pre_process/target_train_x.csv"
    train_y_path = "../Data/pre_process/target_train_y.csv"
    test_x_path = "../Data/pre_process/target_test_x.csv"
    test_y_path = "../Data/pre_process/target_test_y.csv"

    # 数据预处理参数
    window, size, sg = 5, 150, 1
    model_path = '../Data/pre_process/window_' + str(window) + '_size_' + str(size) + '_sg_' + str(sg) + '.vec.model'

    # 模型参数
    weight_1, weight_2, method, threshold, top, enrich, source_from, new = True, True, 1, 0.8, 2000, True, 2, False
    # 数据预处理程序
    # raw_data = datapreprocess.DataProcess(raw_path=raw_path, label_path=label_path, stop_words_path=stop_words_path)
    # raw_data.run(window=window, size=size, sg=sg)
    # Trigger 识别事件模型
    # 触发器加权和不加权对比实验结果
    trigger_classfier = model.Trigger_Classifer(train_path=target_train_path, corpora_split_path=corpora_split_path,
                                                model_path=model_path,
                                                tain_x_path=train_x_path, train_y_path=train_y_path,
                                                test_x_path=test_x_path,
                                                test_y_path=test_y_path, weight_1=weight_1, weight_2=weight_2,
                                                method=method,
                                                threshold=threshold,
                                                top=top, enrich=enrich, source_from=source_from, new=new)
    print(trigger_classfier.run())

    # 其他模型对比
    # event_vec_path = "../Data/trigger/event_vec.csv"
    # contrast_model = other_model.Contrast_Classifer(train_x_path, train_y_path, test_x_path, test_y_path)
    # contrast_model.run()
