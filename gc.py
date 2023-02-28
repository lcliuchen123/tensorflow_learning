# coding:utf-8

# 遗传算法学习,https://blog.csdn.net/u010712012/article/details/82457655

import math
import random
import operator


class GA():
    def __init__(self, length, count):
        self.length = length  # 染色体长度
        self.count = count  # 种群中染色体数量
        self.population = self.gen_population(length, count)  # 随机生产初始种群

    def evolve(self, retrain_rate=0.2, random_select_rate=0.5, mutation_rate=0.01):
        """
        进化，对当前一代种群一次进行选择，交叉并产生新一代种群，然后对新一代种群进行变异
        """
        parents = self.selection(retrain_rate, random_select_rate)
        self.crossover(parents)
        self.mutation(mutation_rate)

    def gen_chromosome(self, length):
        """
        随机生成长度为length的染色体，每个基因的取值是0或者1
        """
        chromosome = 0
        for i in range(length):
            chromosome |= (1 << i) * random.randint(0, 1)

        return chromosome

    def gen_population(self, length, count):
        """获取初始种群，count个长度为length的染色体列表"""
        return [self.gen_chromosome(length) for i in range(count)]

    def fitness(self, chromosome):
        """计算适应度，数值越大，适应度越高"""
        x = self.decode(chromosome)
        return x + 10 * math.sin(5*x) + 7 * math.cos(4*x)

    def selection(self, retrain_rate, random_select_rate):
        """
        选择，按照适应度最大到小排序，选出存活的染色体
        再进行随机选择，选出适应度虽小，但是幸存的个体
        """
        graded = [(self.fitness(chromosome), chromosome) for chromosome in self.population]
        graded = [x[1] for x in sorted(graded, reverse=True)]

        # 选出适应性强的染色体
        retrain_length = int(len(graded) * retrain_rate)
        parents = graded[:retrain_length]

        # 选出适应性不强，但是幸存的染色体
        for chromosome in graded[retrain_length:]:
            if random.random() < random_select_rate:
                parents.append(chromosome)
        return parents

    def crossover(self, parents):
        """交叉"""
        children = []
        target_count = len(self.population) - len(parents)

        while len(children) < target_count:
            male = random.randint(0, len(parents) - 1)
            female = random.randint(0, len(parents) - 1)
            if male != female:
                # 随机选择交叉点
                cross_pos = random.randint(0, self.length)
                mask = 0
                for i in range(cross_pos):
                    mask |= (1 << i)
                male = parents[male]
                female = parents[female]

                # 孩子将获得父亲在交叉点的基因和母亲在交叉点后（包括交叉点）的基因
                child = ((male & mask) | (female & ~mask)) & ((1<<self.length) - 1)
                children.append(child)
                # 如果父母和孩子数量一致，可以更新种群
        self.population = parents + children

    def mutation(self, rate):
        """变异, 对种群的所有个体，随机改变某个个体中的某个基因"""
        for i in range(len(self.population)):
            if random.random() < rate:
                j = random.randint(0, self.length-1)
                self.population[j] ^= 1 << j

    def decode(self, chromosome):
        """解码"""
        return chromosome * 9.0 / (2**self.length-1)

    def result(self):
        """获取最优解"""
        graded = [(self.fitness(chromosome), chromosome) for chromosome in self.population]
        graded = [x[1] for x in sorted(graded, reverse=True)]
        return ga.decode(graded[0])


if __name__ == "__main__":
    ga = GA(10, 30)

    # 200次迭代
    for x in range(90):
        ga.evolve()
    print(ga.result())

