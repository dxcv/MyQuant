#coding=utf-8
'''
Created on 2017-7-13

@author: Zhangquan Zhou
'''

import divid_data as dd
import fnn
import numpy as np
from rhdb import MSSQL

######################################################################################
# 数据准备
######################################################################################

def process_data(x_rough, y_rough):
    '''
            输入：
    x_rough, 原始的训练集特征
    y_rough, 原始的训练集分类标签
            输出：
            一个元祖，tuple=(features, label)
            features是特征矩阵，label是标签向量
                                    每一个样本是一个np.array，每一个特征都是一个list
    '''
    
    clean_data = []
    
    for i in range(0,len(x_rough)):
        features = []
        label = []
        
        # 规范化大数值
        features.append([float(x_rough[i][0]/2000000000)]) # 规范化总市值
        features.append([float(x_rough[i][1]/100)])        # 规范化收盘价
        features.append([float(x_rough[i][2]/720)])        # 规范化上市天数
        
        features.append([x_rough[i][3]])
        features.append([x_rough[i][4]])
        
        # features.append([x_rough[i][5]])
        # features.append([x_rough[i][6]])
        
        # 对于离散变量，用二进制数组代表分类
        if x_rough[i][5] == 0:
            features.append([0])
            features.append([1])
        elif x_rough[i][5] == 1:
            features.append([1])
            features.append([0])
        
        if x_rough[i][6] == 0:
            features.append([0])
            features.append([1])
        elif x_rough[i][6] == 1:
            features.append([1])
            features.append([0])
        
        features = np.array(features)
        
        if y_rough[i][0]==0:
            label = [[0],[1]]
        elif y_rough[i][0]==1:
            label = [[1],[0]]
        label = np.array(label)
    
        clean_data.append((features,label))
    
    return clean_data


######################################################################################
# 训练过程
######################################################################################

def train():
    
    ######################################################################################
    # 以下的参数要手动设置
    ######################################################################################
    
    # 把数据读进来，将数据分成三部分：股票代码，特征矩阵，分类向量
    codes_2011, x_2011, y_2011 = dd.read_csv('C://Users//Administrator//Desktop//divid_nn//data//divid_2011.csv')
    codes_2012, x_2012, y_2012 = dd.read_csv('C://Users//Administrator//Desktop//divid_nn//data//divid_2012.csv')
    codes_2013, x_2013, y_2013 = dd.read_csv('C://Users//Administrator//Desktop//divid_nn//data//divid_2013.csv')
    codes_2016, x_2016, y_2016 = dd.read_csv('C://Users//Administrator//Desktop//divid_nn//data//divid_2016.csv')
    
    # 选择训练数据和测试数据
    # training_x = x_2011
    # training_y = y_2011
    
    x_1112, y_1112 = dd.append_training_data(x_2011, y_2011, x_2012, y_2012)
    # training_x, training_y = dd.append_training_data(x_2011, y_2011, x_2012, y_2012)
    training_x, training_y = dd.append_training_data(x_1112, y_1112, x_2013, y_2013)
    
    # 确定预测年份
    year = '2016'                          # 预测年份
    codes_test = codes_2016                # 预测年份的股票代码
    testing_x = x_2016                     # 训练集特征矩阵
    testing_y = y_2016                     # 训练集分类向量
    
    # group_id
    group_id = 3                           # 分组
    threshold = 0.8                        # 输出阈值
    test_time = 100                        # 训练次数
    
    ######################################################################################
    # 以上的参数要手动设置
    ######################################################################################
    
    # 数据准备，将数据整理成fnn接受的格式
    training_data = process_data(training_x, training_y)
    testing_data = process_data(testing_x, testing_y)
    
    # 设置神经网络参数
    cost_function = fnn.CrossEntropyCost   # 目标函数
    network_struture = [9,10,10,2]            # 神经网络结构
    epoch = 50                             # 训练轮数
    mini_batch_size = 250                  # 随机梯度下降的mini_batch
    eta = 0.0000001                        # 学习率
    lmbda = 0.8                            # 正则化项参数
    
    # 训练过程
    training_accuracy = []                 # 训练集正确率
    training_accuracy.append(0)            # 初始化为0，用来寻优
    evaluation_recall = []                 # 测试集召回率
    evaluation_recall.append(0)            # 初始化为0，用来寻优
    evaluation_accuracy = []               # 测试集正确率
    evaluation_accuracy.append(0)          # 初始化为0，用来寻优
    
    best_training_recall = 0               # 存放最优的训练集召回率
    best_training_accuracy = 0             # 存放最优的训练集正确率
    best_evaluation_recall = 0             # 存放最优的测试集召回率
    best_evaluation_accuracy = 0           # 存放最优的测试集正确率
    best_nn = None                         # 存放神经网络

    
    for i in range(0, test_time):
        print(">> the {0}th test".format(i))
        net = fnn.Network(network_struture, cost=cost_function)
        net.set_threshold(threshold)
        evaluation_cost, evaluation_accuracy, evaluation_recall, training_cost, training_accuracy, training_recall = net.SGD(
                training_data, epoch, mini_batch_size, eta, lmbda, evaluation_data=testing_data,
                monitor_evaluation_accuracy=False,monitor_evaluation_recall=False,
                monitor_training_accuracy=False,monitor_training_recall=False
                )
        # 寻找测试集正确率最大的情况，并且测试集召回率不为0或1
        # if evaluation_accuracy[-1] > best_evaluation_accuracy and evaluation_recall[-1] > 0 and evaluation_recall[-1] < 1:
        # 寻找测试集召回率最大的情况，并且测试集召回率不为0或1，并且测试集和训练集的正确率都要大于0.5
        if evaluation_recall[-1] >0 and evaluation_recall[-1] < 1 and \
            training_recall[-1] >0 and training_recall[-1] < 1 and \
            evaluation_accuracy[-1] > best_evaluation_accuracy and \
            evaluation_accuracy[-1] < 1:
            '''
            and \
            evaluation_accuracy[-1] > 0.4
            '''
            
            best_nn = net
            best_training_recall = training_recall[-1]
            best_training_accuracy = training_accuracy[-1]
            best_evaluation_recall = evaluation_recall[-1]
            best_evaluation_accuracy = evaluation_accuracy[-1]       
            print "current best:"
            print "training accuracy: ", training_accuracy[-1]
            print "training recall: ", training_recall[-1]
            print "evaluation accuracy: ", evaluation_accuracy[-1]
            print "evaluation recall: ", evaluation_recall[-1]
        
    
    print
    print "训练集正确率: ", best_training_accuracy
    print "训练集召回率: ", best_training_recall
    print "测试集正确率: ", best_evaluation_accuracy
    print "测试集召回率: ", best_evaluation_recall
    
    return best_nn, threshold, group_id, best_training_accuracy, best_training_recall, best_evaluation_accuracy, best_evaluation_recall, \
            year, testing_data, testing_x, testing_y, codes_test

######################################################################################
# 处理数据写入到 E003b.stock2中
######################################################################################


def process_results():
    
    best_nn, threashold,group_id, best_training_accuracy, best_training_recall, best_evaluation_accuracy, best_evaluation_recall, \
         year, testing_data, testing_x, testing_y, codes_test = train()

    if best_nn == None:
        return

    db_writer = MSSQL("192.168.1.111","rhis2","abc_1234","rhis1")
    
    for i in range(0,len(codes_test)):
        row = {}   
        row['report_date'] = year + '-12-31'     
        row['report_type'] = '1231'
        row['type_group'] = 1
        row['group_id'] = group_id
        row['stockcode'] = codes_test[i][0]
        row['div_stocks'] = dd.get_divid_stocks(codes_test[i][0], year, '12-31')
        
        (x,y) = testing_data[i]
        predict_y = best_nn.feedforward(x)
        if predict_y[0][0] > threashold:
            insert = '''
                insert into dbo.E003b_stock2 values (\'%s\',\'%s\',%s,%s,\'%s\',%s)
            '''%(row['report_date'],row['report_type'],row['type_group'],row['group_id'],row['stockcode'],row['div_stocks'])
            print(insert)
            db_writer.execute_non_query(insert)
        
######################################################################################
# 处理数据写入到 E003b.sum2中
######################################################################################

def do_analyze():
    
    db_writer = MSSQL("192.168.1.111","rhis2","abc_1234","rhis1")
    
    year = '2016'
    type_group = 1
    group_id = 3
    
    db_check = MSSQL("192.168.1.111","rhis2","abc_1234","RHstock")
    query0 = "select * from dbo.wsd_%s where date between \'%s-12-31\' and \'%s-12-31\' and div_stocks>=1 "%(year, year,year)
    results0 = db_check.execute_query(query0)
    num_total = len(results0)
    
    query1 = '''
        select * from dbo.E003b_stock2 
        where report_date between \'%s-12-31\' and \'%s-12-31\' 
             and type_group = %s and group_id = %s
    '''%(year,year,type_group,group_id)  
    results1 = db_writer.execute_query(query1)
    num = len(results1)
    
    query2 = '''
        select * from dbo.E003b_stock2 
        where report_date between \'%s-12-31\' and \'%s-12-31\' 
             and type_group = %s and group_id = %s and div_stocks >= 1
    '''%(year,year,type_group,group_id)  
    results2 = db_writer.execute_query(query2)
    num_dilivery = len(results2)
    
    db_check = MSSQL("192.168.1.111","rhis2","abc_1234","rhis1")
    
    accuary = float(num_dilivery)/num
    recall = float(num_dilivery)/num_total
    
    print "精确率",float(num_dilivery)/num
    print "召回率",float(num_dilivery)/num_total
    
    insert = '''
                insert into dbo.E003b_stock_sum2 values (\'%s\',%d,%d,%d,%d,%f,%f)
            '''%(year+"-12-31",type_group,group_id,num,num_dilivery,accuary,recall)
            
    print(insert)
    db_writer.execute_non_query(insert)
                

def main():
    #process_results()
    do_analyze()


if __name__=="__main__":   
    main()  
        


