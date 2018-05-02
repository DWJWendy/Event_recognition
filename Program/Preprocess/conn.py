#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 18-4-22 上午10:52
@Author  : 毛毛虫_wendy
@E-mail  : dengwenjun818@gmail.com
@blog    : mmcwendy.info
@File    : conn.py
"""
import pymongo
from Program.Preprocess import datapreprocess

class MongoDB(object):
    def __init__(self):
        # -*- 链接数据库 -*-
        client = pymongo.MongoClient("localhost", 27017)
        db = client["Weibo20180206"]
        self.data = db["data"]

    def find_item(self):
        return self.data.find()

    def write_file(self, item_data):
        with open("../Data/raw/Corpora.csv", "w", encoding="utf-8") as f:
            f.write("Weibo_id"+'\t'+"Company"+'\t'+"Content"+'\t'+"Time"'\t'+"Comment"+'\t'+'Like'+'\t'+"Transfer"+'\n')
            i = 1
            for item in item_data:
                f.write("weibo_"+str(i)+'\t'+item["nickname"]+'\t'+item["Post"]+'\t'+item["Pubtime"] +
                        '\t'+str(item["Comment_num"])+'\t'+str(item["Like_num"])+'\t'+str(item["Transfer_num"])+'\n')
                i += 1
        return i


if __name__ == "__main__":
    conn = MongoDB()
    items = conn.find_item()
    conn.write_file(item_data=items)
    corpora = datapreprocess.read_file("../Data/pre_process/corpora_clean_data.csv")
    print(corpora["Company"].unique())
    company_list = ['中兴通讯', '华为中国区', 'vivo智能手机', 'coolpad官方微博', 'OPPO', '联想', '小米公司', '金立智能手机', "小辣椒",
            "360手机", 'HTC官方微博', '魅族科技', '天语手机', '超级手机官微', '一加手机', '朵唯女性手机', '锤子科技',
            'TCL通讯中国', 'nubia智能手机', '天语手机']

