fnn.py 是前馈神经网络实现，基于numpy包
divid_data.py 包含从数据库读数据，并且读写csv文件的相关函数
divid_nn.py 包含实验前对数据预处理，调用fnn做训练和预测，将实验结果写回数据库
rhdb.py 封装了对SQLServer连接和查询的类