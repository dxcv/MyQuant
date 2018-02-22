
#coding=utf-8 


import pymssql


class MSSQL:

    def __init__(self, host, user, pwd, db):
        self.host = host
        self.user = user
        self.pwd = pwd
        self.db = db
    
    def get_connect(self):
        """
        得到连接的信息
        """
        if not self.db:
            raise(NameError,"没有设置数据库信息")
        self.conn = pymssql.connect(host=self.host,user=self.user,password=self.pwd,database=self.db,charset="utf8")
        cur = self.conn.cursor()
        if not cur:
            raise(NameError,"连接数据库失败")
        else:
            return cur
    
    def execute_query(self,sql):
        """
        执行查询语句
        返回的是一个包含tuple的list，list的元素是记录行，tuple的元素是每行记录的字段

        调用示例：
                ms = MSSQL(host="localhost",user="sa",pwd="123456",db="PythonWeiboStatistics")
                resList = ms.ExecQuery("SELECT id,NickName FROM WeiBoUser")
                for (id,NickName) in resList:
                    print str(id),NickName
        """
        cur = self.get_connect()
        cur.execute(sql)
        resList = cur.fetchall()

        #查询完毕后必须关闭连接
        self.conn.close()
        return resList
    
    
    
    
    def execute_non_query(self,sql):
        """
        执行非查询语句

        调用示例：
            cur = self.get_connect()
            cur.execute(sql)
            self.conn.commit()
            self.conn.close()
        """
        cur = self.get_connect()
        cur.execute(sql)
        self.conn.commit()
        self.conn.close()
        
        
        
        
        
        
        
        
        