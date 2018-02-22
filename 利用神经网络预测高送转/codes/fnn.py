#coding=utf-8

'''
Created on 2017-7-13

@author: Zhangquan Zhou
'''

import numpy as np
import random

######################################################################################
# 神经网络实现
######################################################################################

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

# 二次cost函数类
class QuadraticCost(object):

    @staticmethod
    def fn(a, y):
        """
        a是输出值
        y是真实值
        """
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def delta(z, a, y):
        """
        z是输入，不是整个神经网络的输入，是最近这一层的输入
        a是输出值
        y是真实值
        这是sigmoid为激活函数时二次cost函数的导数
        """
        return (a-y) * sigmoid_prime(z)

# Cross-Entropy 函数
# 用于提高学习速度
class CrossEntropyCost(object):

    @staticmethod
    def fn(a, y):
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(z, a, y):
        return (a-y)


# 前馈神经网络实现
class Network(object):

    def __init__(self, sizes, cost=CrossEntropyCost):
        """
        size是一个list: [2, 3, 1], 代表每一层的元素个数；
        cost指定目标函数，默认为CrossEntropyCost
        """
        self.num_layers = len(sizes)        # 神经网络层数
        self.sizes = sizes                  # 神经网络结构，每一层的神经元个数
        self.default_weight_initializer()   # 初始化神经网络的权重和偏置
        self.cost=cost                      # cost函数设置（目标函数类对象）
        self.threshold=0.5

    def set_threshold(self,threashold):
        self.threshold = threashold

    def default_weight_initializer(self):
        """
        采用N(0,1/sqrt(n))的正态分布初始化参数
        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def large_weight_initializer(self):
        """
        采用N(0,1)的正态分布初始化参数
        """
        # 每一层的偏置是一个列表，大小是该层的神经元个数
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        
        # 每一层的权重是一个矩阵，第i行，第j列代表前一层第j个神经元
        # 到后一层第i个神经元
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, a):
        """
        前馈函数
        """
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            lmbda = 0.0,
            evaluation_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_evaluation_recall=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False,
            monitor_training_recall=False):
        """
        随机梯度下降算法
        training_data 训练集
        epochs 训练的轮数
        mini_batch_size 随机梯度下降算法的训练模块大小
        eta 学习率
        lmbda 正则化项的权重
        evaluation_data 检验集，就是测试集
        monitor_evaluation_cost 是否打印检验集的开销变化
        monitor_evaluation_accuracy 是否打印检验集的正确率变化
        monitor_evaluation_recall 是否打印检验集的召回率变化
        monitor_training_cost 是否打印训练集的开销变化
        monitor_training_accuracy 是否打印训练集的正确率变化
        monitor_training_accuracy 是否打印训练集的召回率变化
        """
        
        n = len(training_data)                                                   # 保存训练集的大小
        evaluation_cost, evaluation_accuracy, evaluation_recall = [], [], []     # 用于保存检验集的目标开销以及正确率
        training_cost, training_accuracy, training_recall = [], [], []           # 用于保存训练集的目标开销以及正确率
        
        # 循环epochs轮进行训练
        for j in range(epochs):
            # 首先打乱训练集的顺序
            random.shuffle(training_data)
            # 按照mini_batch_size取出mini_batch，放在mini_batches中
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            
            # 每一轮通过mini_batch中的训练样本更新数据；
            # 一共更新len(放在mini_batches中)次
            for mini_batch in mini_batches:
                self.update_mini_batch(
                    mini_batch, eta, lmbda, len(training_data))
            
            # print(">> Epoch {0} training complete".format(j))
            
            cost_training = self.total_cost(training_data, lmbda)
            training_cost.append(cost_training)
            
            accuracy_training = self.accuracy(training_data)
            training_accuracy.append(accuracy_training)
            
            recall_training = self.recall(training_data)
            training_recall.append(recall_training)
            
            cost_test = self.total_cost(evaluation_data, lmbda)
            evaluation_cost.append(cost_test)
            
            accuracy_test = self.accuracy(evaluation_data)
            evaluation_accuracy.append(accuracy_test)
            
            recall_test = self.recall(evaluation_data)
            evaluation_recall.append(recall_test)
            
            # 权重和偏置更新完后，计算更新完后的目标开销和正确率
            # 打印训练集目前的开销
            if monitor_training_cost:       
                print("Cost on training data: {0}".format(cost_training))
            # 打印训练集的正确率
            if monitor_training_accuracy:        
                print("Accuracy on training data: {0}".format(accuracy_training))
            # 打印训练集的召回率
            if monitor_training_recall:  
                print("Recall on training data: {0}".format(recall_training))
            # 打印检验集目前的开销
            if monitor_evaluation_cost:      
                print("Cost on evaluation data: {0}".format(cost_test))
            # 打印检验集目前的正确率
            if monitor_evaluation_accuracy:
                print("Accuracy on evaluation data: {0}".format(accuracy_test))
            # 打印检验集的召回率
            if monitor_evaluation_recall:
                print("Recall on evaluation data: {0}".format(recall_test))
        
        return evaluation_cost, evaluation_accuracy, evaluation_recall, training_cost, training_accuracy, training_recall

    # 更新权重
    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        """
        更新权重和偏置
        mini_batch 训练样本
        eta 学习率
        lmbda 正则化项的权重
        n 整个训练集的样本个数，不是mini_batch的样本个数
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]  # 用于计算偏置更新量
        nabla_w = [np.zeros(w.shape) for w in self.weights] # 用于计算权重更新量
        
        # 通过mini_batch中所有的样本计算更新后的值
        for x, y in mini_batch:
            # 首先计算偏置和权重的偏导
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [(1-eta*(lmbda/n))*w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    
    # 反向传播
    def backprop(self, x, y):
        
        nabla_b = [np.zeros(b.shape) for b in self.biases]  # 用于计算偏置的偏导
        nabla_w = [np.zeros(w.shape) for w in self.weights] # 用于计算权重的偏导

        activation = x     # 输入
        activations = [x]  # 用于保存每一层的输出，第一层是输入层
        zs = []            # 用于保存每一层的加权和，在作用激活函数之前的结果

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b      # 加权和
            zs.append(z)                     
            activation = sigmoid(z)          # 作用激活函数
            activations.append(activation)   # 把当下这一层的输出加入activations


        delta = (self.cost).delta(zs[-1], activations[-1], y)    # 输出层求偏导
        nabla_b[-1] = delta                                      # 输出层偏置的更新量就是delta
        # 输出层权重的更新量是上一层的输出（即输出层的输入）
        # 乘以delta得到的向量
        nabla_w[-1] = np.dot(delta, activations[-2].transpose()) 


        # 对每一层的权重和偏置计算更新量
        for l in range(2, self.num_layers):
            # 看倒数第l层的加权和
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            # 倒数第l层的偏置更新量
            nabla_b[-l] = delta
            # 倒数第l层的权重更新量
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)


    def total_cost(self, input_data, lmbda):
        """
        计算input_data中样本的开销
        """
        cost = 0.0
        for x, y in input_data:
            a = self.feedforward(x)
            cost += self.cost.fn(a, y)/len(input_data)
        
        # 加上正则化项 L2,lmbda是正则化项的权重
        cost += 0.5*(lmbda/len(input_data))*sum(
            np.linalg.norm(w)**2 for w in self.weights)
        return cost

    
    def accuracy(self, input_data):
        """
        计算input_data中样本的正确个数
        """
        positive_sum = 0
        predict_sum = 0
        
        for (x,y) in input_data:
            result = self.feedforward(x)
            if result[0][0] > self.threshold:
                predict_sum += 1

                if y[0][0] > y[1][0]:
                    positive_sum += 1
        
        if predict_sum == 0:
            return 0
        else:
            return float(positive_sum)/predict_sum

    
    def recall(self, input_data):
        """
        计算input_data中的召回率
        """
        positive_sum = 0
        predict_sum = 0
        
        for (x,y) in input_data:
            if y[0][0] > y[1][0]:
                positive_sum += 1
                
                result = self.feedforward(x)
                if result[0][0] > self.threshold:
                    predict_sum += 1
        
        if positive_sum == 0:
            return 0
        else:
            return float(predict_sum)/positive_sum
       










