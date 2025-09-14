import re
def parse_log(file):
    with open(file,'r') as f:
        lines = f.readlines()
    data = {}
    for line in lines:
        matchcontent = re.match(r'\[.*?\] epoch (\d+) : loss : (\d+.\d+), loss_ce: (\d+.\d+), loss_mean: (\d+.\d+)',line)
        # 正则表达式匹配内容
        if matchcontent!=None:
            # 不为空即为有效内容
            epoch = int(matchcontent.group(1))
            loss = float(matchcontent.group(2))
            loss_ce = float(matchcontent.group(3))
            loss_mean = float(matchcontent.group(4))
            data[epoch] = loss,loss_ce,loss_mean
            # print(epoch,loss_ce,loss_mean)
    for i in range(len(data)):
        # 检测一下数据是否是连续的（就是从0epoch连续到最后，即为data的长度）
        assert type(data[i])==tuple and len(data[i])==3
    #     每个数据都是tuple类型的，并且长度都为3
    return data
if __name__ == '__main__':
    # 调用函数解析txt文件
    data = parse_log(r'output2/log.txt')
    from matplotlib import pyplot as plt
    # 这是一个[]表达式，展开相当于下面这些代码
    # list1 = []
    # for x in range(len(data)):
    #     list1.append(data[x][2])
    # plt.plot(list1)
    # 其中data[x][2]相当于先对data取epoch索引，再取索引2,即loss_mean
    plt.plot([data[x][2] for x in range(len(data))])
    plt.show()