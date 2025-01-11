# -*- coding: utf-8 -*-

"""
@author     : zuti
@software   : PyCharm
@file       : rock_cfr.py
@time       : 2022/11/21 9:26
@desc       ：

"""
import numpy as np

#动作设置
NUM_ACTIONS = 3  #可选的动作数量
actions = [0,1,2] # 0代表剪刀scissors ， 1代表石头rock ，2 代表布 paper
actions_print=['剪刀','石头','布']
#动作的收益 ，两个人进行博弈，结果
utility_matrix = np.array([
                [0,1,-1],
                [-1,0,1],
                [1,-1,0]
])

def utility_func(action_p1,action_p2,player):
    utility_player = None
    # 注意这里收益矩阵写反了，行代表player1,列代表play2
    if player == 1 :
        utility_player =  - utility_matrix[action_p1][action_p2]
    elif player == 2 :
        utility_player =  utility_matrix[action_p1][action_p2]

    return utility_player

"""基本信息初始化"""
# 玩家，初始化
#策略
player1_strategy = np.array([0.4,0.3,0.3])  # 固定玩家一策略
player2_strategy = np.array([1/3,1/3,1/3])
#动作收益
player1_utility = np.zeros(3)
player2_utility = np.zeros(3)
#遗憾值
player1_regret = np.zeros(3)
player2_regret = np.zeros(3)

#每一局策略(动作的概率分布)之和
player1_strategy_count = np.zeros(3)
player2_strategy_count = np.zeros(3)


for i in range(1000):
    """1局游戏的过程"""
    #对策略进行计数
    # player1_strategy_count += player1_strategy
    player2_strategy_count += player2_strategy
    print(f'----------------游戏开始-------------------')
    # 使用当前策略 选择动作
    action_p1 = np.random.choice(actions, p=player1_strategy)
    action_p2 = np.random.choice(actions, p=player2_strategy)
    print(f'玩家1 动作:{actions_print[action_p1]}--{action_p1} ,玩家2 动作:{actions_print[action_p2]}--{action_p2} .')
    # 得到收益
    # reward_p1 = utility_matrix[action_p2, action_p1]
    # reward_p2 = utility_matrix[action_p1, action_p2]
    reward_p1 = utility_func(action_p1, action_p2,1)
    reward_p2 = utility_func(action_p1, action_p2,2)

    # 输出游戏结果
    print(f'----游戏结束-----')
    print(f'玩家1 收益{reward_p1}  ,玩家2 收益{reward_p2}.')

    # 更新玩家的收益
    player1_utility[action_p1] += reward_p1
    player2_utility[action_p2] += reward_p2
    # 输出一局游戏后的动作收益矩阵
    print(f'收益更新---------动作:{actions_print[0]}        {actions_print[1]}         {actions_print[2]}')
    print(f'玩家1的累计收益   收益:{player1_utility[0]};      {player1_utility[1]};      {player1_utility[2]} ')
    print(f'玩家2的累计收益   收益:{player2_utility[0]};      {player2_utility[1]};      {player2_utility[2]} ')
    #
    """遗憾值更新"""
    # 根据结果收益计算所有动作的遗憾值
    for a in range(3):
        # 玩家2 更新计算遗憾值
        # 事后角度 选择别的动作的收益
        counterfactual_reward_p2 = utility_func(action_p1,a,2)  # 如果选择动作a(而不是事实上的动作action_p2) ,会获得的收益
        regret_p2 = counterfactual_reward_p2 - reward_p2  # 选择动作a和事实上的动作action_p1产生的收益的差别 ,也就是遗憾值(本可以获得更多)
        # 更新玩家的动作遗憾值,历史遗憾值累加
        player2_regret[a] += regret_p2

        # 玩家1 更新计算遗憾值
        # 事后角度 选择别的动作的收益
        counterfactual_reward_p1 = utility_func(a, action_p2, 1)  # 如果选择动作a(而不是事实上的动作action_p1) ,会获得的收益
        regret_p1 = counterfactual_reward_p1 - reward_p1  # 选择动作a和事实上的动作action_p1产生的收益的差别 ,也就是遗憾值(本可以获得更多)
        # 更新玩家的动作遗憾值,历史遗憾值累加
        player1_regret[a] += regret_p1





    print(f'遗憾值更新--------动作:{actions_print[0]}         {actions_print[1]}          {actions_print[0]}')
    print(f'玩家1的累计遗憾值     {player1_regret[0]};      {player1_regret[1]};         {player1_regret[2]} ')
    print(f'玩家2的累计遗憾值     {player2_regret[0]};      {player2_regret[1]};         {player2_regret[2]} ')

    """根据遗憾值更新策略"""
    """遗憾值归一化"""
    # 归一化方法: 1 只看遗憾值大于0的部分，然后计算分布
    palyer1_regret_normalisation = np.clip(player1_regret, a_min=0, a_max=None)
    palyer2_regret_normalisation = np.clip(player2_regret, a_min=0, a_max=None)
    print(f'遗憾值归一化')
    print(f'玩家1归一化后的累计遗憾值     {palyer1_regret_normalisation[0]};      {palyer1_regret_normalisation[1]};         {palyer1_regret_normalisation[2]} ')
    print(f'玩家2归一化后的累计遗憾值     {palyer2_regret_normalisation [0]};      {palyer2_regret_normalisation [1]};         {palyer2_regret_normalisation [2]} ')


    """根据归一化后的遗憾值产生新的策略"""
    palyer2_regret_normalisation_sum = np.sum(palyer2_regret_normalisation)  # 求和
    if palyer2_regret_normalisation_sum > 0:
        player2_strategy = palyer2_regret_normalisation / palyer2_regret_normalisation_sum
    else:
        player2_strategy = np.array([1 / 3, 1 / 3, 1 / 3]) #否则就采取平均策略

    # 玩家一也更新策略，若固定玩家一的策略这里不需要改变
    palyer1_regret_normalisation_sum = np.sum(palyer1_regret_normalisation)  # 求和
    if palyer1_regret_normalisation_sum > 0:
        player1_strategy = palyer1_regret_normalisation / palyer1_regret_normalisation_sum
    else:
        player1_strategy = np.array([1 / 3, 1 / 3, 1 / 3])  # 否则就采取平均策略



"""最终结果:得到平均策略"""
print(f'-----迭代结束,得到最终的平均策略---------')
#根据累计的策略计算平均策略
average_strategy = [0, 0, 0]
palyer2_strategy_sum = sum(player2_strategy_count)
for a in range(3):
    if palyer2_strategy_sum > 0:
        average_strategy[a] = player2_strategy_count[a] / palyer2_strategy_sum
    else:
        average_strategy[a] = 1.0 / 3
print(f'玩家2经过迭代学习得到的平均策略为')
print(f'玩家2的动作 \n 动作:{actions_print[0]} 概率:{average_strategy[0]};动作:{actions_print[1]} 概率:{average_strategy[1]};动作:{actions_print[2]} 概率:{average_strategy[2]} ')

# 同样输出玩家一策略
average_strategy = [0, 0, 0]
palyer1_strategy_sum = sum(player1_strategy_count)
for a in range(3):
    if palyer1_strategy_sum > 0:
        average_strategy[a] = player1_strategy_count[a] / palyer1_strategy_sum
    else:
        average_strategy[a] = 1.0 / 3
print(f'玩家1经过迭代学习得到的平均策略为')
print(f'玩家1的动作 \n 动作:{actions_print[0]} 概率:{average_strategy[0]};动作:{actions_print[1]} 概率:{average_strategy[1]};动作:{actions_print[2]} 概率:{average_strategy[2]} ')
