# -*- coding: utf-8 -*-

"""
@author     : zuti
@software   : PyCharm
@file       : rockpaperscissors_cfr_1.py
@time       : 2022/11/24 15:51
@desc       ：
尝试使用CFR算法来实现剪刀石头布游戏
第一次尝试，使用算法流程进行

"""
import numpy as np

"""游戏设置"""
# 动作设置
NUM_ACTIONS = 3  # 可选的动作数量
actions = [0, 1, 2]  # 0代表剪刀scissors ， 1代表石头rock ，2 代表布 paper
actions_print = ['剪刀', '石头', '布']
# 动作的收益 ，两个人进行博弈，结果
# utility_matrix = np.array([
#     [0, 1, -1],
#     [-1, 0, 1],
#     [1, -1, 0]
# ])
utility_matrix = np.array([
    [0, -2, 2],
    [2, 0, -1],
    [-2, 1, 0]
])

""" 游戏基本情况"""
# 玩家1 策略固定 [0.4,0.3,0.3]
# 玩家2，初始化策略为随机策略[1/3,1/3,1/3],的目的是通过CFR算法，学习得到一个能够获得最大收益的策略
# 整个游戏只有一个信息集，其中包含三个结点，在这个信息集合上可选的动作有3个

# 玩家，初始化
# 策略

player1_strategy = np.array([1 ,0 , 0])
player2_strategy = np.array([0 , 0,  1 ])
# 玩家2在信息集I上关于三个动作的累计的遗憾值
player2_regret_Information = np.zeros(NUM_ACTIONS)
player1_regret_Information = np.zeros(NUM_ACTIONS)
# 玩家2在信息集I上关于三个动作的累计的平均策略
player2_average_strategy = np.zeros(NUM_ACTIONS)
player1_average_strategy = np.zeros(NUM_ACTIONS)



def RegretToStrategy(update_sets,regret):
    """
    使用遗憾值匹配算法 ，根据累计的遗憾值，来确定新的策略

    :return:  新的策略 strategy
    """
    #在每个需要更新的信息集上都使用遗憾值匹配算法来更新策略，
    # 在这个博弈中，信息集只有1个

    #for iteration in update_sets: #遍历所有信息集
    iteration  = None
    strategy = RegretMatchingStrategy(iteration,regret)


    return strategy

def RegretMatchingStrategy(information_set,regret):
    """
    遗憾值匹配算法  ， 更新信息集I 上的策略
    :param information_set:
    :return:
    """
    # 归一化方法: 1 只看遗憾值大于0的部分，然后计算分布
    regret_normalisation = np.clip(regret, a_min=0, a_max=None)
    # print(f'归一化后的累计遗憾值     {regret_normalisation[0]};      {regret_normalisation[1]};         {regret_normalisation[2]} ')
    """根据归一化后的遗憾值产生新的策略"""
    regret_normalisation_sum = np.sum(regret_normalisation)  # 求和

    strategy = np.zeros(NUM_ACTIONS)
    if regret_normalisation_sum > 0:
        strategy = regret_normalisation / regret_normalisation_sum
    else:
        strategy = np.array([1 / 3, 1 / 3, 1 / 3])  # 否则就采取平均策略

    return  strategy



def UpdateAverage(update_sets, strategy , average_strategy ,count ):
    """
    根据本次计算出来的策略，更新信息集合上的平均策略
    进行历史累计，然后对迭代次数进行平均
    :param strategy:
    :param average_strategy:
    :return:
    """
    average_strategy_new = np.zeros( NUM_ACTIONS)
    #for iteration in update_sets:  #遍历所有需要更新的信息集

    for i in range(NUM_ACTIONS):
        average_strategy_new[i] = (count - 1) / count * average_strategy[i] + 1 / count * 1 * strategy[i]  # 不管玩家p2选择哪个动作，信息集I 的出现概率为 1


    return average_strategy_new


def StrategyToValues(update_sets,strategy):
    """
    计算反事实收益值 v
    :param strategy:
    :return:
    """

    # for iteration in update_sets:  #遍历所有需要更新的信息集

    #计算信息集I上所有动作的反事实收益 ，见第三节算例

    #计算每个动作的反事实收益
    counterfactual_value_action = np.zeros(NUM_ACTIONS)
    for  i in  range(NUM_ACTIONS) : #遍历可选动作集

        counterfactual_h1 = player1_strategy[0] * 1 * utility_matrix[0][i]
        counterfactual_h2 = player1_strategy[1] * 1 * utility_matrix[1][i]
        counterfactual_h3 = player1_strategy[2] * 1 * utility_matrix[2][i]

        counterfactual_value_action[i] = counterfactual_h1 + counterfactual_h2 +counterfactual_h3


    return counterfactual_value_action


def UpdateRegret(update_sets, regret , strategy , counterfactual_value_action):
    """
    更新累计反事实遗憾

    :param regret:
    :param strategy:
    :param counterfactual_value_action:
    :return:
    """

    # for iteration in update_sets:  #遍历所有需要更新的信息集

    # 每个动作的反事实值 乘以 策略（每一个动作的概率） 求和 得到 期望
    counterfactual_value_expect  = np.sum(counterfactual_value_action * strategy)

    for i  in range(NUM_ACTIONS):
        regret[i] = regret[i] +   counterfactual_value_action[i] - counterfactual_value_expect

    return  regret


def NormaliseAverage(update_sets,average_strategy):
    """
    归一化得到最后结果

    :param average_strategy:
    :return:
    """

    # for iteration in update_sets:  #遍历所有需要更新的信息集

    strategy_sum = sum(average_strategy)
    strategy = np.zeros(NUM_ACTIONS)
    for i in range( NUM_ACTIONS):

        strategy[i] = average_strategy[i] / strategy_sum

    return   strategy


#整个博弈，玩家2只有一个信息集 ，直接在上面计算即可，因此这个信息集合是个形式化的参数
player2_game_information_set = None
player1_game_information_set = None

#使用CFR求
for count in range(10000):
#使用CFR算法每一步都更新玩家2的策略

    print(f'玩家2 当前策略 ：{player2_strategy}')
    print(f'玩家1 当前策略 ：{player1_strategy}')
    #2 根据当前策略，更新平均策略
    player2_average_strategy = UpdateAverage(player2_game_information_set,player2_strategy , player2_average_strategy ,count+1 )
    print(f'累计平均策略 ：{player2_average_strategy}')
    # 3 根据当前策略计算反事实收益
    player2_counterfactual_value_action = StrategyToValues(player2_game_information_set,player2_strategy)
    print(f'当前策略对应的反事实收益 ：{player2_counterfactual_value_action}')
    #4 更新累计反事实遗憾
    player2_regret_Information = UpdateRegret(player2_game_information_set,player2_regret_Information, player2_strategy, player2_counterfactual_value_action)
    print(f'累计反事实遗憾 ：{player2_regret_Information}')
    # 1 用遗憾值匹配算法 ，根据累计的遗憾值，来确定新的策略
    player2_strategy = RegretToStrategy(player2_game_information_set,player2_regret_Information)
#使用CFR算法每一步更新玩家1的策略
#2 根据当前策略，更新平均策略
    player1_average_strategy = UpdateAverage(player1_game_information_set,player1_strategy , player1_average_strategy ,count+1 )
    print(f'累计平均策略 ：{player1_average_strategy}')
    # 3 根据当前策略计算反事实收益
    player1_counterfactual_value_action = StrategyToValues(player1_game_information_set,player1_strategy)
    print(f'当前策略对应的反事实收益 ：{player1_counterfactual_value_action}')
    #4 更新累计反事实遗憾
    player1_regret_Information = UpdateRegret(player1_game_information_set,player1_regret_Information, player1_strategy, player1_counterfactual_value_action)
    print(f'累计反事实遗憾 ：{player1_regret_Information}')
    # 1 用遗憾值匹配算法 ，根据累计的遗憾值，来确定新的策略
    player1_strategy = RegretToStrategy(player1_game_information_set,player1_regret_Information)





    print(f'-------------迭代次数{count+1}------------')


result2 = NormaliseAverage(player2_game_information_set,player2_average_strategy)
result1= NormaliseAverage(player1_game_information_set,player1_average_strategy)
print(f'最终结果：{result1}')
print(f'最终结果：{result2}')