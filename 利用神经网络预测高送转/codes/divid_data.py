#coding=utf-8
'''
Created on 2017-7-12

@author: Zhangquan Zhou
'''

from rhdb import MSSQL
import csv


def get_query(year, day):
    '''
    year    年份
    day     9月份最后一个交易日
    '''
    
    query = '''
        select stockcode,
            total_shares,
            close_u,
            listdays_t,
            (select surplus_sum from dbo.wsd_%s a where DATEPART(MM,date)='9' and DATEPART(DD,date)='30' and a.stockcode=t.stockcode),
            (select np_parcomsh_chg from dbo.wsd_%s b where DATEPART(MM,date)='9' and DATEPART(DD,date)='30' and b.stockcode=t.stockcode),
            (select case profi_style when \'预增\' then 1 else 0 end from dbo.wsd_%s c where DATEPART(MM,date)='9' and DATEPART(DD,date)='30' and c.stockcode=t.stockcode),
            ipo_lessoneyear,
            (select case  when div_stocks >=1 then 1 else 0 end from dbo.wsd_%s d where DATEPART(MM,date)='12' and DATEPART(DD,date)='31' and d.stockcode=t.stockcode)
        from dbo.wsd_%s t
        where DATEPART(MM,date)='9' and DATEPART(DD,date)=\'%s\' and close_u>=10 and stockcode in(
            select stockcode from dbo.wsd_%s where DATEPART(MM,date)='6' and DATEPART(DD,date)='30' and div_stocks=0
            intersect
            select stockcode from dbo.wsd_%s where DATEPART(MM,date)='9' and DATEPART(DD,date)='30' and np_parcomsh_chg>=0.1 and surplus_sum>=1
            intersect
            select stockcode from dbo.wsd_%s where DATEPART(MM,date)='9' and DATEPART(DD,date)='30' and total_shares<=2000000000 and listdays_t<=720)
    '''%(year,year,year,year,year,day,year,year,year)
    
    return query

def get_divid_data(year, day):
    '''
    year    年份
    day     9月份最后一个交易日
    '''
    
    mssql = MSSQL("192.168.1.111","rhis2","abc_1234","RHStock") # 链接数据库
    query = get_query(year, day)                                # 获取相应的查询
    results = mssql.execute_query(query)                        # 返回结果
    
    return query,results


def get_divid_stocks(stockcode, year, date):
    '''
    stockcode 股票代码
    year      考察年份，格式为'2012'
    date      考察日期，格式为'12-31'
    '''
    mssql = MSSQL("192.168.1.111","rhis2","abc_1234","RHStock") # 链接数据库
    query = "select div_stocks from dbo.wsd_%s where stockcode=\'%s\' and date between \'%s-%s\' and \'%s-%s\'"%(year, stockcode, year, date, year, date)
    # print(query)
    results = mssql.execute_query(query)                       # 返回结果
    value = results[0][0]
    if value==None:
        value = 0
    return value

def write_csv():
    query,results = get_divid_data('2016','30')
    print "SQL query:"
    print query
    print "the query results:"
    for r in results:
        print r
    
    fields = ['stock_code','cap','close','listdays','surplus_sum','np_parcomsh_chg','profi_style','ipo_lessoneyear','is_div_stock']    
    with open('C://Users//Administrator//Desktop//divid_2016.csv','wb') as csvFile:
        writer = csv.DictWriter(csvFile, fieldnames=fields)
        
        writer.writeheader()
        for row in results:
            row_string={}
            for i in range(len(row)):
                row_string[fields[i]] = row[i]
            writer.writerow(row_string)


def read_csv(file_path):
    stock_codes = []
    X = []
    Y = []
    
    with open(file_path) as csvFile:
        reader = csv.DictReader(csvFile)

        for row in reader:
            x = []
            y = []
            code = []
            code.append(row['stock_code'])
            x.append(float(row['cap']))
            x.append(float(row['close']))
            x.append(float(row['listdays']))
            x.append(float(row['surplus_sum']))
            x.append(float(row['np_parcomsh_chg']))
            x.append(int(row['profi_style']))
            x.append(int(row['ipo_lessoneyear']))
            y.append(int(row['is_div_stock']))
            X.append(x)
            Y.append(y)
            stock_codes.append(code)
        
    
    return stock_codes,X,Y

def append_training_data(X_1, Y_1, X_2, Y_2):
    
    X_3 = []
    Y_3 = []
    for i in range(0,len(X_1)):
        X_3.append(X_1[i])
        Y_3.append(Y_1[i])
    
    for i in range(0,len(X_2)):
        X_3.append(X_2[i])
        Y_3.append(Y_2[i])
    
    return X_3, Y_3
    
    

def main():
    '''
    # write_csv() # 将数据库里的数据读出写入到csv文件中
    codes_2011, x_2011, y_2011 = read_csv('C://Users//Administrator//Desktop//divid_2011.csv')
    codes_2012, x_2012, y_2012 = read_csv('C://Users//Administrator//Desktop//divid_2012.csv')
    codes_2013, x_2013, y_2013 = read_csv('C://Users//Administrator//Desktop//divid_2013.csv')
    codes_2016, x_2016, y_2016 = read_csv('C://Users//Administrator//Desktop//divid_2016.csv')
    
    x_1112, y_1112 = append_training_data(x_2011, y_2011, x_2012, y_2012)
    
    print "length of x_2011, y_2011: ", len(x_2011), len(y_2011)
    print "length of x_2012, y_2012: ", len(x_2012), len(y_2012)
    print "length of x_201112, y_1112: ", len(x_1112), len(y_1112)
    '''
    
    result = get_divid_stocks('600028.SH', '2012', '12-31')
    print(result)


if __name__=="__main__":   
    main()



