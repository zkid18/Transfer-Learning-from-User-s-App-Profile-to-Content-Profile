import csv
import numpy as np

def datas():
    data = csv.reader(open('./5001rawdata.csv'))
    alltest = []
    alluser = []
    n = 0
    realdata = []
    for item in data:
        if n != 0:
            realdata.append(item)
        n = n + 1
    realdata = np.array(realdata)


    # =============================inputdata===================================
    user_dic = {}
    for i in range(0,len(realdata)):
        user_dic[realdata[i][0]] = []


    for i in range(0,len(realdata)):
        if user_dic[realdata[i][0]] == []:
            realdata[i][3] = realdata[i][3].replace(' ',',')
            xdata = eval(realdata[i][3])
            xdata = np.array(xdata)
            user_dic[realdata[i][0]] = xdata
    # print len(user_dic)
    # # =============================outputdata===================================
    output_dic = {}
    for i in range(0,len(realdata)):
        output_dic[realdata[i][0]] = np.zeros(150)

    for i in range(0,len(realdata)):
        output_dic[realdata[i][0]][int(realdata[i][1])-2] = output_dic[realdata[i][0]][int(realdata[i][1])-2] + float(realdata[i][2])
    # print len(output_dic)

    data = []
    for item in user_dic:
        user = []
        user.append(item)
        user.append(user_dic[item])
        user.append(output_dic[item])
        data.append(user)
    user_id = []
    input = []
    output = []
    for i in range(0,len(data)):
        user_id.append(data[i][0])
        input.append((data[i][1]))
        output.append((data[i][2]))
    user_id = np.array(user_id)
    # input = np.array(input)
    ouput = np.array(output)
    for i in range(0,len(output)):
        output[i] = output[i]/np.sum(output[i])
    # ====================================testdata====================================
    testdata = csv.reader(open('./5001testdata.csv'))
    k = 0
    testdatas = []
    for item in testdata:
        if k != 0:
            t = item[1].replace("  "," ").split(" ")
            alltest.append(t)
            alluser.append(item[0])
            testdatas.append(item)
        k = k + 1
    # =============================test inputdata===================================
    testuser_dic = {}
    for i in range(0,len(testdatas)):
        testuser_dic[testdatas[i][0]] = []


    for i in range(0,len(testdatas)):
        if testuser_dic[testdatas[i][0]] == []:
            testdatas[i][1] = testdatas[i][1].replace(' ',',')
            testdatas[i][1] = testdatas[i][1].replace(',,', ',')
            xdata = eval(testdatas[i][1])
            xdata = np.array(xdata)
            testuser_dic[testdatas[i][0]] = xdata
    # print len(testuser_dic)
    for item in testuser_dic:
        input.append(testuser_dic[item])
    input = np.array(input)
    return input,output,np.array(alltest,dtype=np.int),alluser
input,output,alltest,alluser = datas()
# mins = []
# for i in range(0,len(output)):
#     index = np.argwhere(output[i] == 0)
#     output[i][index] = 100
#     min = np.min(output[i])
#     mins.append(min)
# print np.min(mins)
# print float(1/150)
#







