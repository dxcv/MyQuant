
# coding: utf-8

# In[ ]:




import jqdata
import numpy as np
import random
import talib


######################################################################################
# 数据准备
######################################################################################


def prepare_data(start_date, end_date, cut_date):

    # 用沪深300做测试
    test_stock = '000300.XSHG'
    
    # 测试开始日期和终止日期
    #start_date = datetime.date(2007, 1, 4)
    #end_date = datetime.date(2017, 1, 3)
    
    # 所有的交易日期
    trading_days = jqdata.get_all_trade_days()
    start_date_index = trading_days.tolist().index(start_date)
    end_date_index = trading_days.tolist().index(end_date)
    cut_date_index = trading_days.tolist().index(cut_date)
    
    all_data = []
    training_data=[]
    testing_data=[]

    feature_list = []
    label_list = []
    # 获取所有的数据
    for index in range(start_date_index, end_date_index):
        
        #计算特征值
        start_day = trading_days[index - 35]
        end_day = trading_days[index]
        stock_data = get_price(test_stock, start_date=start_day, end_date=end_day,                                frequency='daily', fields=['close','high','low','volume'])
        close_prices = stock_data['close'].values
        high_prices = stock_data['high'].values
        low_prices = stock_data['low'].values
        volumes = stock_data['volume'].values
        #通过数据计算指标
        sma_data = talib.SMA(close_prices)[-1]    
        wma_data = talib.WMA(close_prices)[-1]
        mom_data = talib.MOM(close_prices)[-1]
        stck, stcd = talib.STOCH(high_prices, low_prices, close_prices)
        stck_data = stck[-1]
        stcd_data = stcd[-1]

        macd, macdsignal, macdhist = talib.MACD(close_prices)
        macd_data = macd[-1]
        rsi_data = talib.RSI(close_prices,timeperiod=10)[-1]
        willr_data = talib.WILLR(high_prices, low_prices, close_prices)[-1]
        cci_data = talib.CCI(high_prices, low_prices, close_prices)[-1]
        
        mfi_data = talib.MFI(high_prices, low_prices, close_prices, volumes)[-1]
        obv_data = talib.OBV(close_prices, volumes)[-1]
        roc_data = talib.ROC(close_prices)[-1]
        cmo_data = talib.CMO(close_prices)[-1]
        
        # 训练数据或者测试数据的输入特征
        features = []
        features.append([sma_data])  #0
        features.append([0])
        features.append([wma_data])  #2
        features.append([0])
        features.append([mom_data])  #4
        features.append([0])
        features.append([stck_data]) #6
        features.append([0])
        features.append([stcd_data]) #8
        features.append([0])
        features.append([macd_data]) #10
        features.append([0])
        features.append([rsi_data])  #12
        features.append([0])
        features.append([willr_data])#14
        features.append([0])
        features.append([cci_data])  #16
        features.append([0])
        features.append([mfi_data])  #18
        features.append([0])
        features.append([obv_data])  #20
        features.append([0])
        features.append([roc_data])  #22
        features.append([0])
        features.append([cmo_data])  #24
        features.append([0])
        features.append([close_prices[-1]])
        features = np.array(features)
        
        # 训练数据或者测试的标签数据，就是涨或者跌，涨用[1,0]，平或者跌用[0,1]
        label = [[0],[1]]
        if close_prices[-1] > close_prices[-2]:
            label = [[1],[0]]
        elif close_prices[-1] < close_prices[-2]:
            label = [[0],[1]]
        label = np.array(label)
        
        feature_list.append(features)
        label_list.append(label)
    
    # 连续数值离散化
    # 从后面开始向前面遍历，第一行数据需要舍弃,range只包含第一个元素，不包含第二个元素
    for index in range(len(feature_list)-1, 0, -1):
        # SMA
        if feature_list[index][0][0] < feature_list[index][-1][0]:
            feature_list[index][0][0] = 1
            feature_list[index][1][0] = 0
        else:
            feature_list[index][0][0] = 0
            feature_list[index][1][0] = 1
        # WMA
        if feature_list[index][2][0] < feature_list[index][-1][0]:
            feature_list[index][2][0] = 1
            feature_list[index][3][0] = 0
        else:
            feature_list[index][2][0] = 0
            feature_list[index][3][0] = 1
        # MOM
        if feature_list[index][4][0] > 0:
            feature_list[index][4][0] = 1
            feature_list[index][5][0] = 0
        else:
            feature_list[index][4][0] = 0
            feature_list[index][5][0] = 1
        # STCK
        if feature_list[index][6][0] > feature_list[index-1][6][0]:
            feature_list[index][6][0] = 1
            feature_list[index][7][0] = 0
        else:
            feature_list[index][6][0] = 0
            feature_list[index][7][0] = 1
        # STCD
        if feature_list[index][8][0] > feature_list[index-1][8][0]:
            feature_list[index][8][0] = 1
            feature_list[index][9][0] = 0
        else:
            feature_list[index][8][0] = 0
            feature_list[index][9][0] = 1
        # MACD
        if feature_list[index][10][0] > feature_list[index-1][10][0]:
            feature_list[index][10][0] = 1
            feature_list[index][11][0] = 0
        else:
            feature_list[index][10][0] = 0
            feature_list[index][11][0] = 1

        # RSI
        if feature_list[index][12][0] > 70:
            feature_list[index][12][0] = 0
            feature_list[index][13][0] = 1
        elif feature_list[index][12][0] < 30:
            feature_list[index][12][0] = 1
            feature_list[index][13][0] = 0
        else:
            if feature_list[index][12][0] > feature_list[index-1][12][0]:
                feature_list[index][12][0] = 1
                feature_list[index][13][0] = 0
            else:
                feature_list[index][12][0] = 0
                feature_list[index][13][0] = 1
        # WILLR
        if feature_list[index][14][0] > feature_list[index-1][14][0]:
            feature_list[index][14][0] = 1
            feature_list[index][15][0] = 0
        else:
            feature_list[index][14][0] = 0
            feature_list[index][15][0] = 1
        # CCI
        if feature_list[index][16][0] > 200:
            feature_list[index][16][0] = 0
            feature_list[index][17][0] = 1
        elif feature_list[index][16][0] < -200:
            feature_list[index][16][0] = 1
            feature_list[index][17][0] = 0
        else:
            if feature_list[index][16][0] > feature_list[index-1][16][0]:
                feature_list[index][16][0] = 1
                feature_list[index][17][0] = 0
            else:
                feature_list[index][16][0] = 0
                feature_list[index][17][0] = 1
                
        # MFI
        if feature_list[index][18][0] > 90:
            feature_list[index][18][0] = 0
            feature_list[index][19][0] = 1
        elif feature_list[index][18][0] < 10:
            feature_list[index][18][0] = 1
            feature_list[index][19][0] = 0
        else:
            if feature_list[index][18][0] > feature_list[index-1][18][0]:
                feature_list[index][18][0] = 1
                feature_list[index][19][0] = 0
            else:
                feature_list[index][18][0] = 0
                feature_list[index][19][0] = 1
        # OBV
        if feature_list[index][20][0] > feature_list[index-1][20][0]:
            feature_list[index][20][0] = 1
            feature_list[index][21][0] = 0
        else:
            feature_list[index][20][0] = 0
            feature_list[index][21][0] = 1
        # ROC
        if feature_list[index][22][0] > 0:
            feature_list[index][22][0] = 1
            feature_list[index][23][0] = 0
        else:
            feature_list[index][22][0] = 0
            feature_list[index][23][0] = 1
        # CMO
        if feature_list[index][24][0] > 50:
            feature_list[index][24][0] = 0
            feature_list[index][25][0] = 1
        elif feature_list[index][24][0] < -50:
            feature_list[index][24][0] = 1
            feature_list[index][25][0] = 0
        else:
            if feature_list[index][24][0] > feature_list[index-1][24][0]:
                feature_list[index][24][0] = 1
                feature_list[index][25][0] = 0
            else:
                feature_list[index][24][0] = 0
                feature_list[index][25][0] = 1       
        # 删除价格
        feature_list[index] = np.delete(feature_list[index],-1,axis=0)
    
    
    for i in range(1,len(feature_list)):
        sample_term = (feature_list[i],label_list[i])
        if 1+i+start_date_index < cut_date_index:
            training_data.append(sample_term)
        elif 1+i+start_date_index >= cut_date_index:
            testing_data.append(sample_term)
    
    
    print("Data prepared. training data size: {0}, testing data size: {1}".format(len(training_data),len(testing_data)))
    return training_data,testing_data

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
            monitor_training_cost=False,
            monitor_training_accuracy=False):
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
        monitor_training_cost 是否打印训练集的开销变化
        monitor_training_accuracy 是否打印训练集的正确率变化
        """
        
        n_data = len(evaluation_data)                     # 保存检验集的大小
        n = len(training_data)                            # 保存训练集的大小
        evaluation_cost, evaluation_accuracy = [], []     # 用于保存检验集的目标开销以及正确率
        training_cost, training_accuracy = [], []         # 用于保存训练集的目标开销以及正确率
        
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
            
            print("Epoch {0} training complete".format(j))
            
            # 权重和偏置更新完后，计算更新完后的目标开销和正确率
            # 打印训练集目前的开销
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print("Cost on training data: {0}".format(cost))
            # 打印训练集的正确率
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data)
                training_accuracy.append(float(accuracy)/n)
                print("Accuracy on training data: {0} / {1}".format(
                    accuracy, n))
            # 打印检验集目前的开销
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda)
                evaluation_cost.append(cost)
                print("Cost on evaluation data: {0}".format(cost))
            # 打印检验集目前的正确率
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(float(accuracy)/n_data)
                print("Accuracy on evaluation data: {0} / {1}".format(
                    self.accuracy(evaluation_data), n_data))
        
        return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy

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

    def accuracy(self, input_data):
        """
        计算input_data中样本的正确个数
        """
        results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                        for (x, y) in input_data]
        return sum(int(x == y) for (x, y) in results)


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


######################################################################################
# 测试代码
######################################################################################
    
start_date = datetime.date(2007, 1, 4)
end_date = datetime.date(2017, 1, 3)
cut_date = datetime.date(2016, 6, 3)
    
cost_function = CrossEntropyCost
network_struture = [26,50,2]
epoch = 70
mini_batch_size = 10
eta = 0.001
lmbda = 0.1
    
training_data, testing_data = prepare_data(start_date,end_date,cut_date)
net = Network(network_struture, cost=cost_function)
evaluation_cost, evaluation_accuracy, training_cost, training_accuracy =net.SGD(
        training_data, epoch, mini_batch_size, eta, lmbda, evaluation_data=testing_data,monitor_evaluation_accuracy=True,monitor_training_accuracy=True)
    

