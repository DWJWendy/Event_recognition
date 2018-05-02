#!-usr-bin-env python
# -*- encoding:utf-8 -*-
"""
@author:毛毛虫_Wendy
@license:(C) Copyright 2017- 
@contact:dengwenjun@gmail.com
@file:evolution.py
@time:10-15-17 10:37 PM
"""

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt


class Evolution(object):
    def __init__(self):
        self.event = ["Sale", "Cooperation", "Promotion", "Research", "Recruiting", "Null"]
        self.time = ["2016-01", "2016-02", "2016-03", "2016-04", "2016-05", "2016-06", "2016-07", "2016-08", "2016-09",
                     "2016-10", "2016-11", "2016-12", "2017-01", "2017-02", "2017-03", "2017-04", "2017-05", "2017-06"]
        self.company = ["Huawei", "Lenovo", "ZTE", "OPPO", "MI", "Coolpad", "Meizu", "Vivo", "HTC", "TCL"]
        # self.company = ["Huawei", "Lenovo", "ZTE", "OPPO", "MI", "Coolpad", "Meizu", "Vivo", "HTC", "TCL", "Gionee",
        #                 "Smartisan", "LE", "Nubia", "K-touch", "Duowei", "Xiaolajiao", "360"]
        self.datafile = "../Data_2/predict/final_predict_result.txt"
        self.marker = ["D", "h", "H", "p", "8", "p", "+", "h", "o", "D", "H", "p", "8", "p", "+", "h", "o", "D"]
        self.font = {"family": "Times New Roman", "color": "black", "weight": "normal", "size": 10}
        self.path = "../Results/10_picture_2/event/"

    def read_company(self, company_name="LE"):
        """
        :param datafile: 读取分类后的结果文件
        :param company_name: 读取某一公司关于5类事件的行为
        :return:
        """
        """
        先构造一个18*10维的矩阵，再读取输入结果，更新数据矩阵
        """

        data = pd.DataFrame(np.zeros(108).reshape(18, 6), columns=self.event, index=self.time)
        with open(self.datafile, "r") as f:
            datalist = f.readlines()
        for item in datalist:
            item = item.split("\t")
            event_type = item[3].replace("\n", "")
            if item[1] == company_name and re.match(r"(2016-|2017-)\d{1,2}", item[2]) and event_type in self.event:
                date = re.match(r"(2016-|2017-)\d{1,2}", item[2]).group()
                if date in self.time:
                    data.ix[date, event_type] += 1
            else:
                pass
        """
        数据归一化，并乘上100，算对应占比(%)
        """
        for t in self.time:
            datasum = sum(data.ix[t])
            if datasum != 0:
                data.ix[t] = (data.ix[t] / sum(data.ix[t])) * 100
            else:
                data.ix[t] = 0.0
        return data

    def read_event(self, event_name="Sale"):
        """
        :param datafile: 读取分类后的结果文件
        :param event_name: 画出对应的十个企业对于同一个事件的演化
        :return:
        """
        """
        先构造一个18*10维的矩阵，再读取输入结果，更新数据矩阵
        """
        data = pd.DataFrame(np.zeros(180).reshape(18, 10), columns=self.company, index=self.time)

        with open(self.datafile, "r") as f:
            datalist = f.readlines()
        for item in datalist:
            item = item.split("\t")
            company_name = item[1]
            if item[3].replace("\n", "") == event_name and re.match(r"(2016-|2017-)\d{1,2}",
                                                                    item[2]) and company_name in self.company:
                date = re.match(r"(2016-|2017-)\d{1,2}", item[2]).group()
                if date in self.time:
                    data.ix[date, company_name] += 1
            else:
                pass
        """
        数据归一化，并乘上100，算对应占比(%)
        """
        for t in self.time:
            datasum = sum(data.ix[t])
            if datasum != 0:
                data.ix[t] = (data.ix[t] / datasum) * 100
            else:
                data.ix[t] = 0.0
        return data

    def read_company_event(self, company, event_name):
        data = pd.DataFrame(np.zeros(36).reshape(18, 2), columns=[event_name, "Average"], index=self.time)
        with open(self.datafile, "r") as f:
            datalist = f.readlines()
        for item in datalist:
            item = item.split("\t")
            company_name = item[1]
            if item[3].replace("\n", "") == event_name and re.match(r"(2016-|2017-)\d{1,2}",
                                                                    item[2]) and company_name==company:
                date = re.match(r"(2016-|2017-)\d{1,2}", item[2]).group()
                if date in self.time:
                    data.ix[date, event_name] += 1
            else:
                pass
        data["Average"] = sum(data[event_name])/len(data.index)

        return data


    def picture(self, data, param, type):
        """
        :param data: 输入构造数据
        :param param: 输入当前操作比如：Sale 或者 Huawei
        :return:
        """
        df = pd.DataFrame(data)
        for i in range(len(type)):
            df[type[i]].plot(kind="line", style=self.marker[i], linestyle=":", ylim=[0, 80], label=type[i])
        plt.xlabel("(Year/Month)", fontdict=self.font)
        plt.ylabel("Ratio(%)", fontdict=self.font)
        plt.legend(type)
        plt.savefig(self.path + param + ".png", dpi=1000)
        plt.show()

    def all_picture(self, company):
        fig = plt.figure()
        ax1 = fig.add_subplot(3, 2, 1)
        ax2 = fig.add_subplot(3, 2, 2)
        ax3 = fig.add_subplot(3, 2, 3)
        ax4 = fig.add_subplot(3, 2, 4)
        ax5 = fig.add_subplot(3, 2, 5)
        ax6 = fig.add_subplot(3, 2, 6)
        ax = [ax1, ax2, ax3, ax4, ax5, ax6]
        xlabel = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
        for i in range(len(self.event)):
            data = self.read_company_event(company=company, event_name=self.event[i])
            ax[i].plot(data[self.event[i]], "k^--", color="b", alpha=0.6)
            ax[i].plot(data["Average"], color="r")
           # ax[i].legend([self.event[i], "Average"])
            ax[i].set_xlabel(xlabel[i]+self.event[i])
            ax[i].set_xticks([])

        # fig, axes = plt.subplots(3, 2, sharex=True)
        # e = {"0_0":0,"0_1":1,"1_0":2,"1_1":3,"2_0":4,"2_1":5}
        # for i in range(3):
        #     for j in range(2):
        #         i_j = str(i)+"_" + str(j)
        #         data = self.read_company_event(company=company, event_name=self.event[e[i_j]])
        #         axes[i, j].plot(data[self.event[e[i_j]]], "k^--", color="b", alpha=0.6)
        #         axes[i, j].plot(data["Average"], color="r")
        #         axes[i, j].legend([self.event[e[i_j]], "Average"])
        #         axes[i, j].set_xticks([])
        plt.subplots_adjust(wspace=0, hspace=0.2)
        plt.savefig(self.path + company + ".png", dpi=1000)
        plt.show()


def PRF(data, xlabel="Method", ylabel="PRF_value", path="../Results/picture/PRF.png"):
    """
    :param data: 输入数据集
    :return: 画准确率、召回率、F值 柱状图
    """
    """
    输入类似的数据集
     data ={"WEED":[0.53165,0.563641667,0.547178621],
          "BOW":[0.21255,0.25143,0.230360992],
         "TF-IDF":[0.221485,0.244865,0.232588933],
          "TF-IDF_W":[0.225738333,0.245026667,0.234987356],
          "LDA":[0.024728333,0.10846,0.040274324]}
    """

    df = pd.DataFrame(data, index=["Precision", "Recall", "F_value"]).T
    df.plot(kind="bar", ylim=[0, 0.8], color=["b", "r", "c"], alpha=0.4)  # ylim设置y轴刻度
    plt.xticks(rotation=360, fontsize=8)
    plt.xlabel(xlabel, fontsize=10)
    plt.ylabel(ylabel, fontsize=10)
    plt.savefig(path, dpi=1000)
    plt.show()


def line_top(data, xlabel="Top_1", ylabel="F_value", path="../Results/picture/",
             title="Top high frequency words in train_data"):
    data.plot(kind="line", style="^", linestyle=":", ylim=[0.5, 0.8], c="r")
    plt.xticks(list(data.index), fontsize=7)
    plt.yticks(fontsize=7)
    plt.ylabel(ylabel, fontsize=9)
    plt.xlabel(xlabel, fontsize=9)
    plt.legend([xlabel], fontsize=8)
    plt.title(title, fontsize=9)
    plt.savefig(path + xlabel + ".png", dpi=1000)
    plt.show()


def line_threshold(data, xlabel="Threshold", ylabel="F_value", path="../Results/picture/"):
    data.plot(kind="line", style=["H", "^"], linestyle=":", ylim=[0.5, 0.8], color=["r", "c"], alpha=0.6)
    plt.xticks(list(data.index), fontsize=7)
    plt.yticks(fontsize=7)
    plt.ylabel(ylabel, fontsize=9)
    plt.xlabel(xlabel, fontsize=9)
    plt.legend(["Trigger", "classifier"], fontsize=8)
    plt.title("The cosine threshold in trigger and classifier", fontsize=9)
    plt.savefig(path + xlabel + ".png", dpi=1000)
    plt.show()


def main(param=0):
    """
    :param param: 主函数用于打印演化图像
    :return:
    """
    evolution_process = Evolution()
    event = evolution_process.event
    company = evolution_process.company
    if param == 0:
        for e in event:
            data = evolution_process.read_event(event_name=e)
            evolution_process.picture(data=data, type=company, param=e)
    elif param == 1:
        for c in company:
            data = evolution_process.read_company(company_name=c)
            evolution_process.picture(data=data, type=event, param=c)

    elif param == 2:
        for c in company:
            evolution_process.all_picture(company=c)



if __name__ == "__main__":
    # 方法对比
    # data = {"SVM+BOW": [0.6365886331, 0.6483333333, 0.6735703769],
    #         "Bayes+BOW": [0.470971739, 0.4383333333, 0.6124026231],
    #         "NN+BOW": [0.5171112291, 0.5066666667, 0.5413555238],
    #         "W2V+Trigger": [0.670078444, 0.6666666667, 0.699839807]}
    # PRF(data=data, ylabel="FPR_value")

    # Enrich
    data = {"False": [0.5744285375, 0.54, 0.629034448],
            "True": [0.670078444, 0.6666666667, 0.699839807]}
    PRF(data=data, xlabel="Enrich", ylabel="FPR_value", path="../Results/param/Enrich.png")

    # Weighted
    data = {"No_Weighted": [0.584394133, 0.54, 0.6503920368],
            "Weighted": [0.670078444, 0.6666666667, 0.699839807]}
    PRF(data=data, xlabel="Weighted", ylabel="FPR_value", path="../Results/param/Weighted.png")

    # TOP_1
    # data = {"F_value": [0.6784796315, 0.6676826045, 0.6673490888, 0.6627985845, 0.6643484657, 0.6643484657]}
    # data = pd.DataFrame(data, index=[200, 300, 400, 500, 600, 700])
    # line_top(data)

    # Top_2
    # data = {"F_value": [0.6705333221, 0.6673490888, 0.6676826045, 0.6705127059, 0.665960273, 0.6690327368]}
    # data = pd.DataFrame(data, index=[1000, 2000, 3000, 4000, 5000, 6000])
    # line_top(data, xlabel="Top_2", title="Top high frequency words in corpora")

    # threshold_1 & threshold_2
    # data = {"Threshold_1": [0.6672245882, 0.6690327368, 0.6690472535, 0.6690327368, 0.6690327368],
    #         "Threshold_2": [0.627702875, 0.6363378931, 0.6446815143, 0.667366422, 0.5716282908]}
    # data = pd.DataFrame(data, index=[0.5, 0.6, 0.7, 0.8, 0.9])
    # line_threshold(data, xlabel="Threshold")

    # main(param=0)

