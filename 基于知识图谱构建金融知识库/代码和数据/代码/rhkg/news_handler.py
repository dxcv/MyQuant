# encoding=utf-8

'''
Created on 2017-7-21

@author: Zhangquan Zhou
'''

import sys
import jieba
import jieba.analyse
from rhkg import kgraph
from collections import deque

class NewsHandler(object):
    
    def __init__(self, words_list, knowledge_graph):
        '''
                         接受两个参数：
        words_list: 术语列表，用于导入分词器，是list类型；
        knowledge_graph: 关系图，知识图谱，是dict类型
        '''
        self.words_list = words_list
        self.knowledge_graph = knowledge_graph
        
        # 将术语列表的术语添加进分词器
        for word in words_list:
            jieba.add_word(word)
        
    
    def extract_tags(self,news,topK):
        '''
                        提取关键词，topK是前多少个关键词，自己设置
        '''
        return jieba.analyse.extract_tags(news,topK=topK, withWeight=False, allowPOS=())
    
    
    def trace(self,news,topK=40,path_len=-1,multi_tags=-1,root_tag=False):
        """
        根据分词结果追踪新闻涉及的相关代码：
        输入：
        news:新闻文本
        topK:根据分词结果的前topK个结果来追踪，默认为40；
        path_len:返回关联路径上届为path_len的关联股票，默认为-1，没有上届；
        multi_tags:返回涉及关键词数在multi_tags以上的股票，默认为-1，没有上届；
        root_tag:返回分类树上的底层标签；
         输出：dict类型，key是证券名或者证券代码，value是所有导向key的路径
        """
        
        # 提取关键词
        tags = self.extract_tags(news, topK)
        
        # 下面对关键词进行筛选
        tags_clean = []
        for tag in tags:
            # 只有知识图谱中有这个术语，才考虑进来
            if tag in self.knowledge_graph:
                # 如果只考虑根概念，那么将非根概念全部剔除
                if root_tag == True:
                    value_results = self.knowledge_graph[tag]
                    has_subclass = False
                    if kgraph.superclassof in value_results:
                        subclasses = value_results[kgraph.superclassof]
                            
                        for subclass in subclasses:
                            if subclass in tags:
                                has_subclass = True
    
                    if has_subclass == False:
                        tags_clean.append(tag)
                else:
                    tags_clean.append(tag)

                    
        # 计算所有关联路径
        all_results = self.trace_by_tags(tags_clean)     
        
        # 根据关联路径长度过滤出较短路径的关联股票
        filter_by_len = {}
        if path_len == -1:
            path_len = sys.maxsize
        
        for tag in all_results:
            paths = all_results[tag]
            new_paths = []
            for path in paths:
                if len(path) <= path_len:
                    new_paths.append(path)

            if len(new_paths) > 0:
                filter_by_len[tag]=new_paths
                        
        # 最后根据关键字个数进一步过滤
        filter_results = {}
        if multi_tags == -1:
            multi_tags = 0
        
        for tag in filter_by_len:
            paths = filter_by_len[tag]
            if len(paths) >= multi_tags:
                filter_results[tag] = paths
        
        
        return filter_results
    
    
    def trace_by_tags(self,tags):
        '''
                        输入：tags是术语列表
                        输出：result_dict是dict类型，key是证券名或者证券代码，value是所有导向key的路径
        '''
        
        result_dict = {}      
        for tag in tags:
            current_dict = self.trace_by_tag(tag)
            for key in current_dict:
                if key in result_dict:
                    result_dict[key].append(current_dict[key])
                else:
                    result_dict[key] = [current_dict[key]]
        
        return result_dict
    
    
    def trace_by_tag(self,tag):
        '''
                        输入：关键词tag
                        输出：相关股票，以及追踪路径，dict类型，key是证券名或者证券代码，value是单条导向key的路径
        '''
        
        result_dict = {}                         # 结果dict类型，key是股票名或者股票代码，value是单条关联路径
        tag_queue = deque([tag])                        # 术语队列，注意，必须是队列，因为得实现广度优先找到最浅的路径
        tag_visited = []                         # 用于记录经访问过的术语
        tag_trace_path = {}                      # 对于某一个术语，导向它的路径
        tag_trace_path[tag]=[(tag,'in','News')]  # 关联路径的第一条边，表明tag是从news提取出来的
       
        while len(tag_queue) > 0:
            '''
                                    该while循环的逻辑是这样：
                                    从给定的关键词tag出发，在knowledge graph上进行游走，
                                    直到把所有关联词都访问一遍，且只访问一遍，将关联词是证券代码或者名称
                                    的关联词作为结果返回，并且把关联路径记录下来。
            '''  
            
            # 如果术语栈还有未检查的术语，那么pop出来查看
            current_tag = tag_queue.popleft()
            # 将该术语放进tag_visited表明已经访问过
            tag_visited.append(current_tag)
            # 如果该术语存在知识图谱节点，再开展搜索
            if current_tag in self.knowledge_graph:
                # 将该术语所有的属性和值拿出来
                value_list = self.knowledge_graph[current_tag]
                
                # 根据travel_predicates对知识图谱进行游走
                travel_predicates = kgraph.get_travel_predicates()
                for predicate in travel_predicates:
                    
                    # 不会去考虑更范化的情况
                    if predicate == kgraph.subclassof:
                        continue
                    
                    # 如果value_list存在属性predicate
                    if predicate in value_list:
                        # 将该属性的值拿出来
                        candidate_tags = value_list[predicate]                                               
                        for candidate_tag in candidate_tags:              
                            # 判断是否已经追踪到股票代码或者股票名称
                            if candidate_tag == kgraph.SCode \
                                or candidate_tag == kgraph.SName:
                                if current_tag not in result_dict:
                                    result_dict[current_tag] = tag_trace_path[current_tag]
                            
                            # 对于属性值不为关键字的其他值，并且没有被访问过，那么进行一步的图游走
                            if candidate_tag not in tag_visited \
                                and kgraph.is_defined(candidate_tag) == False:
                                    # 将将要访问的术语放进tag_stack中
                                    tag_queue.append(candidate_tag)
                                    # 将前驱节点的关联路径复制给自己
                                    tag_trace_path[candidate_tag] = list(tag_trace_path[current_tag])
                                    # 添加从前驱节点到自己的边
                                    tag_trace_path[candidate_tag].append((current_tag,predicate,candidate_tag))
                    
        return result_dict

    
    
    
    
    
    
    
    
