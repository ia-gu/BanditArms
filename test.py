import numpy as np
import matplotlib.pyplot as plt
import random

# アームを引く
class Bandit:
    # 初めにアームの数を決め，正規表現に従ってQ*(a)を選ぶ
    def __init__(self, arms):
        self.arms = arms
        self.Q_a = np.random.randn(arms)                # 引用(2)

    # 行動aを選択するたび，平均Q*(a)，分散1の正規表現から報酬を選ぶ
    def play(self, a):
        return np.random.normal(self.Q_a[a], 1.0)       # 引用(2)

# グリーディ
class Greedy:
    def __init__(self, act_num):
        self.Q_a = np.zeros(act_num)
        self.pre = None
        self.rewards = [[] for i in range(act_num)]     # 引用(1)

    # 行動を選択する
    def select(self):
        action = np.argmax(self.Q_a)
        # 行動を記録する
        self.pre = action
        return action

    # 行動の報酬を与える
    def get_reward(self, r):
        self.rewards[self.pre].append(r)
        self.update(self.pre)

    # 報酬を平均する
    def update(self, action):
        self.Q_a[action] = np.mean(self.rewards[action])

# εグリーディ
class eps_Greedy:
    def __init__(self, act_num, eps):
        self.act_num = act_num
        self.eps = eps
        self.Q_a = np.zeros(act_num)
        self.pre = None
        self.rewards = [[] for i in range(act_num)]     # 引用(1)
 

    def select(self):
        # 確率εで，ランダムに行動を選択する
        if random.random() < self.eps:
            action = random.randint(0, self.act_num - 1)
        # 他の場合はGreedyと同じ
        else:
            action = np.argmax(self.Q_a)
        self.pre = action
        return action

    # 行動の報酬を与える
    def get_reward(self, r):
        self.rewards[self.pre].append(r)
        self.update(self.pre)
        
    # 報酬を平均する
    def update(self, action):
        self.Q_a[action] = np.mean(self.rewards[action])

# プレイ
# それぞれのインスタンスで同じ名前のメソッドを用意したため，if分岐は不要
def do(play_num, player, tasks):
    avg = []
    for i in range(play_num):
        rewards = []
        
        for j in range(len(player)):
            action = player[j].select()
            reward = tasks[j].play(action)
            player[j].get_reward(reward)
            rewards.append(reward)

        reward_avg = np.mean(rewards)
        avg.append(reward_avg)
        print(f'【{i+1}回目】', end='')
        print(f'報酬平均：{reward_avg}')

    return avg

# 比較のため，seed固定
random_seed = 9999
random.seed(random_seed)
np.random.seed(random_seed)

# プレイ数，タスク数，行動の種類を入力
print('Bandit Task')
n_p = int(input('plays(int):'))
n_t = int(input('tasks(int):'))
n_a = int(input('arms(int):'))

# 行動と，インスタンスを用意
tasks = [Bandit(n_a) for i in range(n_t)]
greed = [Greedy(n_a) for i in range(n_t)]
eps_greed = [eps_Greedy(n_a, 0.1) for i in range(n_t)]
eps_greed2 = [eps_Greedy(n_a, 0.01) for i in range(n_t)]

# n_p回の行動を行う
print(f'\n##greedy method start##')
g_avg = do(n_p, greed, tasks)
print(f'\n##eps=0.1 greedy method start##')
e_g_avg = do(n_p, eps_greed, tasks)
print(f'\n##eps=0.01 greedy method start##')
e_g_avg2 = do(n_p, eps_greed2, tasks)

# 結果表示
x = [i for i in range(n_p)]
figure = plt.figure()
plt.xlabel('Plays')
plt.ylabel('Average reward')
plt.plot(x, g_avg, label='ε=0')
plt.plot(x, e_g_avg, label='ε=0.1')
plt.plot(x, e_g_avg2, label='ε=0.01')
plt.legend()
plt.show()
figure.savefig('result.png')