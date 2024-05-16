import numpy as np
import pandas as pd

# 定义通用输出层
class OutputLayer(object):
    # 股票的数量
    stock_num = 1
    # 预测时长
    day_num = 1

    # 股票权重
    vector: np.ndarray = np.ones((stock_num, day_num)) / (stock_num * day_num)
    # 优化前的股票权重
    pre_vector: np.ndarray = np.ones((stock_num, day_num))

    # 定义构造函数初始化计算层
    def __init__(self, vector: np.ndarray):
        self.vector = vector
        self.stock_num = vector.shape[0]
        self.day_num = vector.shape[1]
        self.pre_vector = np.ones((self.stock_num, self.day_num))

    # 定义静态方法生成初始化输出
    @staticmethod
    def init_value(stock_num, day_num):
        vector = np.ones((stock_num, day_num)) / (stock_num * day_num)
        return OutputLayer(vector)
    
    # 输出通用输出层
    def info(self):
        print("输出层: ", self.vector.shape)
        print(self.vector)

    # 判断是否可以结束循环
    def is_end(self, day: int):
        # TODO 判断是否收敛
        # print((abs(self.vector[:, day] - self.pre_vector[:, day]) < 0.001))
        return (abs(self.vector[:, day] - self.pre_vector[:, day]) < 0.001).all()

    # 更新输出层
    def update(self, day: int):
        self.pre_vector[:, day] = self.vector[:, day]


# 定义输入层
class InputLayer(object):
    # 股票数量
    stock_num = 1
    # 预测时长
    day_num = 1

    # 收益率
    r = np.ones((stock_num, day_num))
    # 预测收益率
    r_ = np.ones((stock_num, day_num))
    # 预测误差
    epsilon = r - r_
    # 协方差
    sigma = np.zeros((stock_num, stock_num))

    # 定义构造函数
    def __init__(self, r: np.ndarray, r_: np.ndarray):
        self.r = r
        self.r_ = r_
        self.stock_num = r.shape[0]
        self.day_num = r.shape[1]
        self.epsilon = r - r_
        self.sigma = np.zeros((self.stock_num, self.stock_num))

        # 计算平均收益率
        r_mean = np.mean(self.r, axis = 1)

        # TODO 复杂度待优化 计算协方差
        for i in range(self.stock_num):
            for j in range(self.stock_num):
                for k in range(self.day_num):
                    self.sigma[i][j] += (self.r[i][k] - r_mean[i]) * (self.r[j][k] - r_mean[j])

                self.sigma[i][j] /= self.day_num

# 定义计算层
class CalculateLayer(object):
    # 股票数量
    stock_num = 1
    # 预测时长
    day_num = 1

    # 输入层
    input_layer: InputLayer
    # 输出层
    output_layer: OutputLayer

    # 梯度下降步长
    step_size = 0.00001

    # 定义构造函数
    def __init__(self, input_: InputLayer):
        self.input_layer = input_

        self.stock_num = input_.stock_num
        self.day_num = input_.day_num
        
        # 初始化输出层
        self.output_layer = OutputLayer.init_value(self.stock_num, self.day_num)

    # 负梯度优化
    def optimize(self):
        for day in range(self.day_num):
            print("day: ", day)
            while not self.output_layer.is_end(day):
                self.step_optimize(day)

    # 单步负梯度优化
    def step_optimize(self, day: int):
        self.output_layer.update(day)
        # 计算负梯度
        fixed = np.zeros((self.stock_num,))
        for index in range(self.stock_num):
            fixed[index] = self.rho(index, day)

        # 输出梯度向量
        # print("梯度向量: ", fixed)

        # 更新输出层
        self.output_layer.vector[:, day] -= fixed * self.step_size
        self.output_layer.vector[:, day] /= np.sum(self.output_layer.vector[:, day])

    # 求偏导
    def rho(self, index: int, day: int):
        # 计算偏导
        return (2 * np.sum(self.output_layer.vector[:, day] * self.input_layer.sigma[index]) - self.input_layer.r[index][day] - self.input_layer.epsilon[index][day])

    # 输出信息
    def info(self):
        self.output_layer.info()


if __name__ == '__main__':
    # 读取数据
    origin = pd.read_excel("./assets/origin.xlsx", engine="openpyxl", sheet_name="di^")
    origin = np.array(origin)

    origin = origin[:, 1280:]
    print("origin.shape: ", origin.shape)

    # 读取预测数据
    predict = pd.read_excel("./assets/predict.xlsx", engine="openpyxl", sheet_name="y_pred")
    predict = np.array(predict)

    predict = predict[:, 1:]
    predict = predict.T
    print("predict.shape: ", predict.shape)
    input_layer = InputLayer(origin, predict)
    
    calc = CalculateLayer(input_layer)
    calc.optimize()
    calc.info()

    output_file = "./assets/output.csv"
    np.savetxt(output_file, calc.output_layer.vector, delimiter=",", fmt="%.5f")
