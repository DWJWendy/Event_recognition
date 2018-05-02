#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 18-4-21 下午4:37
@Author  : 毛毛虫_wendy
@E-mail  : dengwenjun818@gmail.com
@blog    : mmcwendy.info
@File    : statistics.py
"""

from Program.Preprocess import datapreprocess

company_list = ['中兴通讯', '华为中国区', 'vivo智能手机', 'coolpad官方微博', 'OPPO', '联想', '小米公司', '金立智能手机',
                 "小辣椒", "360手机", 'HTC官方微博', '魅族科技', '天语手机', '超级手机官微', '一加手机', '朵唯女性手机', '锤子科技',
                'TCL通讯中国', 'nubia智能手机']

if __name__ == "__main__":
    corpora = datapreprocess.read_file(path="../Data/pre_process/corpora_clean_data.csv")


    company__num = {}
    with open("predict_corpora_data.txt", "w", encoding="utf-8") as f:
        f.write("Weibo_id" + '\t' + "Company" + '\t' + "Content" + '\t' + "Time"'\t' +
                "Comment" + '\t' + 'Like' + '\t' + "Transfer" + '\n')
        i = 1
        for key in company_list:
            data = corpora[corpora["Company"] == key]
            print(data)
            company__num[key] = len(data.index)
    print(company__num)



