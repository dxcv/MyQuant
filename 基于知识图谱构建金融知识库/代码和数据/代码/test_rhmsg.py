# encoding=utf-8

'''
Created on 2017-7-28

@author: Zhangquan Zhou
'''

from rhkg import *

# 加载消息
news_file = open(r'C:\Users\Administrator\Desktop\消息面策略\news1.txt','r', encoding='UTF-8')
new_text = news_file.readlines()[0]
print("news:{0}".format(new_text))
print("\n")

# 加载知识图谱，并构建词典
word_list, knowledge_graph = kgraph.load_csv_dict(r'C:\Users\Administrator\Desktop\消息面策略\rhkg.csv')
print("词典:{}".format(word_list))
print("\n")
for subject in knowledge_graph:
    print("主语:{}".format(subject))
    print("\t属性和值:{}".format(knowledge_graph[subject]))
print("\n")

# 实例化消息处理对象
handler = news_handler.NewsHandler(word_list, knowledge_graph)

topK = 50
tags = handler.extract_tags(new_text, topK)
print("关键词{0}:{1}".format(topK,tags))
print("\n")

related_securities = handler.trace(new_text)  
for key in related_securities:
    print("相关股票:"+key)
    paths = related_securities[key]
    i = 0
    for path in paths:
        print("\t关联路径{0}:{1}".format(i,path))
        i += 1
        
print("\n")
print("根据关联路径长度筛选之后：")
related_securities = handler.trace(new_text,path_len=2)  
for key in related_securities:
    print("相关股票:"+key)
    paths = related_securities[key]
    i = 0
    for path in paths:
        print("\t关联路径{0}:{1}".format(i,path))
        i += 1       


print("\n")
print("根据关联路径长度以及语义关联筛选之后：")
related_securities = handler.trace(new_text,path_len=4,multi_tags=-1,root_tag=True)  
for key in related_securities:
    print("相关股票:"+key)
    paths = related_securities[key]
    i = 0
    for path in paths:
        print("\t关联路径{0}:{1}".format(i,path))
        i += 1  










        