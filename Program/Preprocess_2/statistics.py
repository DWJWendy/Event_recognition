#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 18-4-28 上午9:10
@Author  : 毛毛虫_wendy
@E-mail  : dengwenjun818@gmail.com
@blog    : mmcwendy.info
@File    : statistics.py
"""

from Program.Preprocess import datapreprocess
import random


def form_clean_corpora(raw, company):
    """
    :param company_list: 将原始预料raw_data中抽取company_list的新语料
    :return:
    """
    indexs = []
    for i in range(len(raw.index)):
        if raw.ix[i, "Company"] not in company:
            indexs.append(i)
    clean_data = raw.drop(indexs, axis=0)
    datapreprocess.write_file(path="../Data_2/raw/clean_company_corpora.csv", item_data=clean_data)
    return clean_data


def form_target_data(item, label_map, company):
    indexs = []
    for i in range(len(item.index)):
        if item.ix[i, "Company"] not in company:
            indexs.append(i)
    clean_data = item.drop(indexs, axis=0)
    clean_data = clean_data.dropna()
    clean_data["Label"] = clean_data["Label"].apply(lambda x: label_map[x])
    for i in range(len(company)):
        print(company[i], len(clean_data[clean_data["Company"] == company[i]].index))
    for i in range(1, 7):
        print(i, len(clean_data[clean_data["Label"] == i].index))
    print(len(clean_data.index))
    print(clean_data["Company"].unique())
    datapreprocess.write_file(path="../Data_2/raw/target.csv", item_data=clean_data)
    return clean_data


def form_test_data(item):
    test_data = item.drop("Sentence_vec", axis=1)
    datapreprocess.write_file(path="../Data_2/pre_process/test_data_label.csv",item_data=test_data)




if __name__ == "__main__":
    company_list = ['中兴通讯', '华为中国区', 'vivo智能手机', 'coolpad官方微博', 'OPPO', '联想', '小米公司', '金立智能手机',
                    "小辣椒", "360手机", 'HTC官方微博', '魅族科技', '天语手机', '超级手机官微', '朵唯女性手机',
                    '锤子科技', 'TCL通讯中国', 'nubia智能手机']
    label_num = {"招募": 1, "合作": 2, "研发": 3, "推广": 4, "销售": 5, "无": 6}
    # raw_data = datapreprocess.read_file(path="../Data/raw/Corpora.csv")
    # form_clean_corpora(raw_data, company_list)
    # item = datapreprocess.read_file(path="../Data_2/raw/Corpora_label.csv")
    # form_target_data(item=item, label_map=label_num, company=company_list)
    data = datapreprocess.read_file(path="../Data_2/pre_process/test_data.csv")
    form_test_data(data)
