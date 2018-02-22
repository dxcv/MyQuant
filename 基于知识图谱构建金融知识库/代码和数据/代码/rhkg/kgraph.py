# encoding=utf-8

'''
Created on 2017-7-21

@author: Zhangquan Zhou
'''

import csv

# 内部定义关键词
subject = 'subject'
predicate = 'predicate'
objject = 'object'
# 内部定义类
SCode = 'SCode'                   # 证券代码类
SName = 'SName'                   # 证券名称类
Sector = 'Sector'                 # 证券部门类
# 内部定义谓词
scode = 'scode'                   # 指示证券代码
is_code_of = 'is_code_of'         # 指示代码所属对象
sname = 'sname'                   # 指示证券名称
is_name_of = 'is_name_of'         # 指示名称所属对象
sector = 'sector'                 # 指示行业板块
is_sector_of = 'is_sector_of'     # 指示行业板块所属对象
theme = 'theme'                   # 指示题材
is_theme_of = 'is_theme_of'       # 指示题材所属对象
is_a = 'is_a'                     # 指示所属类别
has = 'has'                       # 指示从属成员
subclassof = 'subclassof'         # 指示子类
superclassof = 'superclassof'     # 指示父类
# 对称谓词
same_as = 'same_as'               # 指示同义对象
relate_to = 'relate_to'           # 指示关联对象

defined_classes          = [SName,SCode,Sector]
defined_predicates       = [is_a,sname,scode,sector,theme,same_as,relate_to,subclassof, 
                            has,is_name_of,is_code_of,is_sector_of,is_theme_of,superclassof]

corresponding_predicates = [is_a,sname,scode,sector,theme,same_as,relate_to,subclassof]
inverse_predicates       = [has, is_name_of,is_code_of,is_sector_of,is_theme_of,same_as,
                            relate_to,superclassof]


def load_csv_dict(file_path):
    """
    从csv文件中读取知识图谱，并且构建词典。
    """
    
    knowledge_graph = {}
    word_list = []
    
    with open(file_path) as csvFile:
        reader = csv.DictReader(csvFile)
        
        for row in reader:
            
            s = row[subject]
            p = row[predicate]
            o = row[objject]
            
            insert_triple(knowledge_graph, s, p, o)          
            inverse_p = get_inverse_predicate(p)
            # 加入互反关系
            insert_triple(knowledge_graph, o, inverse_p, s)
            
            # 将非关键词的术语加入词典
            if is_defined(s) == False:
                word_list.append(s)
            # 将非关键词的术语加入词典
            if is_defined(o) == False:
                word_list.append(o)

    
    word_list = list(set(word_list)) # 去重
    
    return word_list, knowledge_graph


def insert_triple(knowledge_graph, s, p, o):
    
    if s in knowledge_graph:
        values = knowledge_graph[s]
        if p not in values:
            values[p] = [o]
        else:
            values[p].append(o)
    else:
        knowledge_graph[s] = {}
        knowledge_graph[s][p] = [o]
        
    

def get_inverse_predicate(predicate):
    
    if predicate in corresponding_predicates:
        i = corresponding_predicates.index(predicate)
        return inverse_predicates[i]
    else:
        i = inverse_predicates.index(predicate)
        return corresponding_predicates[i]
            
    
def is_defined(word):
    if word in defined_classes or \
        word in defined_predicates:
        return True
    else:
        return False    
    
    
def get_travel_predicates():
    return defined_predicates
    
    
def show(knowledge_graph):
    for key in  knowledge_graph:
        print(key+":")
        print(knowledge_graph[key])
    
    
    
    
    