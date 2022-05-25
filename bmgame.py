from copy import deepcopy
from random import random
import numpy as np
from enum import Enum
import random
import math

class BMAction(Enum):
    NO_OP = 0
    MAKE_SCV = 1
    BUILD_DEPOT = 2
    BUILD_BARRACKS = 3
    MAKE_MARINE = 4

class BMGame:
    """
    An abstract simulation of a BuildMarines game
    tims/step scale: one step eq one second
    """
    def __init__(self):
        # minerals cost
        self.scv_cost = 50
        self.marine_cost = 50
        self.depot_cost = 100
        self.barracks_cost = 150

        # time cost
        self.scv_time = 17
        self.marine_time = 25
        self.depot_time = 30
        self.barracks_time = 60

        # limit
        self.total_minerals = 15000
        self.total_depots = 20
        self.total_barracks = 20
        self.total_steps = 600

        # actions
        self.actions = 5

        # reward
        self.reward = 0
        self.reward_op = 0

        # steps
        self.current_steps = 0
        self.current_steps_op = 0
        
    def minerals_every_second(self, population):
        minerals_rate = 5*math.log2(population[2])
        return round(minerals_rate, 0)

    def reset(self,): 
        # production queue, sure to be sleek
        self.scv_bq = []
        self.marine_bq = []
        self.depot_bq = []
        self.barracks_bq = []
        self.production_queue = [self.scv_bq,self.marine_bq,\
                                self.depot_bq,self.barracks_bq]

        # population
        self.population_cap = 15
        self.total_population = 12
        self.scv_population = 12
        self.marine_population = 0
        self.population = [self.population_cap,self.total_population,\
                            self.scv_population,self.marine_population]

        # building
        self.depot_num = 0
        self.barracks_num = 0
        self.building = [self.depot_num, self.barracks_num]

        # economic values
        self.total_values = 12*50
        self.population_values = 12*50
        self.buliding_values = 0

        # minerals rate
        # if self.scv_population <= 16:
        #     self.minerals_rate = 1.25
        # else:
        #     self.minerals_rate = 1.2
        self.minerals_rate = self.minerals_every_second(self.population)
        self.minerals = [50, self.minerals_rate, self.total_values]


        # op production queue, sure to be sleek
        self.scv_bq_op = []
        self.marine_bq_op = []
        self.depot_bq_op = []
        self.barracks_bq_op = []
        self.production_queue_op = [self.scv_bq_op,self.marine_bq_op,\
                                    self.depot_bq_op,self.barracks_bq_op]

        # op population
        self.population_cap_op = 15
        self.total_population_op = 12
        self.scv_population_op = 12
        self.marine_population_op = 0
        self.population_op = [self.population_cap_op,self.total_population_op,\
                            self.scv_population_op,self.marine_population_op]

        # op building
        self.depot_num_op = 0
        self.barracks_num_op = 0
        self.building_op = [self.depot_num_op, self.barracks_num_op]

        # op economic values
        self.total_values_op = 12*50
        self.population_values_op = 12*50
        self.buliding_values_op = 0

        # op minerals rate
        if self.scv_population_op <= 16:
            self.minerals_rate_op = 1.25
        else:
            self.minerals_rate_op = 1.2
        self.minerals_op = [50, self.minerals_rate_op, self.total_values_op]

        # reward
        self.rewrads = [self.reward, self.reward_op]

        # steps
        self.steps = [self.current_steps, self.current_steps_op]

        obs = [self.minerals, self.population, self.building, self.production_queue, \
                self.minerals_op, self.population_op, self.building_op, self.production_queue_op,\
                self.rewrads, self.steps]
        return obs
    
    def check_gameover(self,obs):
        minerals = obs[0]
        current_steps = obs[9][0]

        minerals_op = obs[4]
        current_steps_op = obs[9][1]

        total_value = minerals[2] + minerals_op[2]
        total_steps = current_steps + current_steps_op
        return(total_value>=self.total_minerals or total_steps>=self.total_steps)

    def check_make_scv(self, obs, player):
        minerals = obs[0]
        population = obs[1]
        production_queue = obs[3]

        minerals_op = obs[4]
        population_op = obs[5]
        production_queue_op = obs[7]

        if player == 1:
            return (minerals[0] >= self.scv_cost 
                    and population[1] < population[0]
                    and len(production_queue[0]) < 5) # limit of cc queue is 5
        elif player == -1:
            return (minerals_op[0] >= self.scv_cost 
                    and population_op[1] < population_op[0]
                    and len(production_queue_op[0]) < 5) # limit of cc queue is 5

    def check_build_depot(self, obs, player):
        minerals = obs[0]
        building = obs[2]
        production_queue = obs[3]

        minerals_op = obs[4]
        building_op = obs[6]
        production_queue_op = obs[7]

        total_depots = building[0] + len(production_queue[2]) + \
                       building_op[0] + len(production_queue_op[2])
        if player == 1:
            return (minerals[0] >= self.depot_cost
                    and total_depots < self.total_depots)
        elif player == -1:
            return (minerals_op[0] >= self.depot_cost
                    and total_depots < self.total_depots)
    
    def check_build_barracks(self, obs, player):
        minerals = obs[0]
        building = obs[2]
        production_queue = obs[3]

        minerals_op = obs[4]
        building_op = obs[6]
        production_queue_op = obs[7]

        total_barracks = building[1] + len(production_queue[3]) + \
                         building_op[1] + len(production_queue_op[3])
        if player == 1:
            return (minerals[0] >= self.barracks_cost
                    and total_barracks < self.total_barracks
                    and building[0] >= 1) # at least one depot is built
        elif player == -1:
            return (minerals_op[0] >= self.barracks_cost
                    and total_barracks < self.total_barracks
                    and building_op[0] >= 1) # at least one depot is built
    
    def check_make_marine(self, obs, player):
        minerals = obs[0]
        population = obs[1]
        building = obs[2]

        minerals_op = obs[4]
        population_op = obs[5]
        building_op = obs[6]

        if player == 1:
            return (minerals[0] >= self.marine_cost
                    and population[1] < population[0]
                    and building[1] > 0) # at least one barracks is built
        elif player == -1:
            return (minerals_op[0] >= self.marine_cost
                    and population_op[1] < population_op[0]
                    and building_op[1] > 0) # at least one barracks is built
    
    def check(self, obs, player):
        checkers = [
                    (lambda x,y: True),
                    self.check_make_scv,
                    self.check_build_depot,
                    self.check_build_barracks,
                    self.check_make_marine,
                ]

        available_action = [i for i in range(self.actions) if checkers[i](obs, player)]
        return available_action

    def update_minerals_rate(self,population):
        if population[2] <= 16:
            minerals_rate = 1.25
        else:
            minerals_rate = 1.2
        return minerals_rate

    def no_op(self, obs, player):

        minerals = obs[0]
        population = obs[1]
        building = obs[2]
        production_queue = obs[3]

        minerals_op = obs[4]
        population_op = obs[5]
        building_op = obs[6]
        production_queue_op = obs[7]

        rewards = obs[8]
        steps = obs[9]

        if player == 1:
            steps[0] += 1
            # minerals[0] += round(population[2]*minerals[1],0)
            # minerals[2] += round(population[2]*minerals[1],0)
            minerals[0] += minerals[1]
            minerals[2] += minerals[1]
            new_queue = []
            for cls,p in enumerate(production_queue):
                p = [x-1 for x in p]
                if len(p) > 0:
                    # unit cls has been completed, because sleek, pop first
                    if p[0] == 0:
                        p.pop(0)
                        # a new scv add to population
                        if cls == 0:
                            population[2] += 1
                            # minerals_rate = self.update_minerals_rate(population)
                            minerals_rate = self.minerals_every_second(population)
                            minerals[1] = minerals_rate
                        # a new marine add to population
                        elif cls == 1:
                            population[3] += 1
                            rewards[0] += 1
                        # a new depot add to building  
                        elif cls == 2:
                            building[0] += 1
                            population[0] += 12 # a depot add 12 population cap
                        # a new barracks add to building 
                        elif cls == 3:
                            building[1] += 1
                new_queue.append(p)
            production_queue = new_queue
        elif player == -1:
            steps[1] += 1
            # minerals_op[0] += round(population_op[2]*minerals_op[1],0)
            # minerals_op[2] += round(population_op[2]*minerals_op[1],0)
            minerals_op[0] += minerals_op[1]
            minerals_op[2] += minerals_op[1]
            new_queue = []
            for cls,p in enumerate(production_queue_op):
                p = [x-1 for x in p]
                if len(p) > 0:
                    # unit cls has been completed, because sleek, pop first
                    if p[0] == 0:
                        p.pop(0)
                        # a new scv add to population
                        if cls == 0:
                            population_op[2] += 1
                            # minerals_rate_op = self.update_minerals_rate(population_op)
                            minerals_rate_op = self.minerals_every_second(population_op)
                            minerals_op[1] = minerals_rate_op
                        # a new marine add to population
                        elif cls == 1:
                            population_op[3] += 1
                            rewards[1] += 1
                        # a new depot add to building  
                        elif cls == 2:
                            building_op[0] += 1
                            population_op[0] += 12 # a depot add 12 population cap
                        # a new barracks add to building 
                        elif cls == 3:
                            building_op[1] += 1
                new_queue.append(p)
            production_queue_op = new_queue
        obs = [minerals, population, building, production_queue, minerals_op, \
                population_op, building_op, production_queue_op, rewards, steps]
        return obs

    def make_scv(self, obs, player):
        # first do no_op
        obs = self.no_op(obs, player)
        # then add a scv to the production queue and total population plus one
        minerals = obs[0]
        population = obs[1]
        building = obs[2]
        production_queue = obs[3]

        minerals_op = obs[4]
        population_op = obs[5]
        building_op = obs[6]
        production_queue_op = obs[7]

        rewards = obs[8]
        steps = obs[9]

        if player == 1:
            minerals[0] -= self.scv_cost
            production_queue[0].append(self.scv_time)
            population[1] += 1
        elif player == -1:
            minerals_op[0] -= self.scv_cost
            production_queue_op[0].append(self.scv_time)
            population_op[1] += 1
        obs = [minerals, population, building, production_queue, minerals_op, \
                population_op, building_op, production_queue_op, rewards, steps]
        return obs

    def make_marine(self, obs, player):
        # first do no_op
        obs = self.no_op(obs, player)
        # then add a marine to the production queue and total population plus one
        minerals = obs[0]
        population = obs[1]
        building = obs[2]
        production_queue = obs[3]

        minerals_op = obs[4]
        population_op = obs[5]
        building_op = obs[6]
        production_queue_op = obs[7]

        rewards = obs[8]
        steps = obs[9]

        if player == 1:
            minerals[0] -= self.marine_cost
            production_queue[1].append(self.marine_time)
            population[1] += 1
        elif player == -1:
            minerals_op[0] -= self.marine_cost
            production_queue_op[1].append(self.marine_time)
            population_op[1] += 1
        obs = [minerals, population, building, production_queue, minerals_op, \
                population_op, building_op, production_queue_op, rewards, steps]
        return obs

    def build_depot(self, obs, player):
        # first do no_op
        obs = self.no_op(obs, player)
        # then add a depot to the production queue
        minerals = obs[0]
        population = obs[1]
        building = obs[2]
        production_queue = obs[3]

        minerals_op = obs[4]
        population_op = obs[5]
        building_op = obs[6]
        production_queue_op = obs[7]

        rewards = obs[8]
        steps = obs[9]

        if player == 1:
            minerals[0] -= self.depot_cost
            production_queue[2].append(self.depot_time)
        elif player == -1:
            minerals_op[0] -= self.depot_cost
            production_queue_op[2].append(self.depot_time)
        obs = [minerals, population, building, production_queue, minerals_op, \
                population_op, building_op, production_queue_op, rewards, steps]
        return obs

    def build_barracks(self, obs, player):
        # first do no_op
        obs = self.no_op(obs, player)
        # then add a barracks to the production queue
        minerals = obs[0]
        population = obs[1]
        building = obs[2]
        production_queue = obs[3]

        minerals_op = obs[4]
        population_op = obs[5]
        building_op = obs[6]
        production_queue_op = obs[7]

        rewards = obs[8]
        steps = obs[9]

        if player == 1:
            minerals[0] -= self.barracks_cost
            production_queue[3].append(self.barracks_time)
        elif player == -1:
            minerals_op[0] -= self.barracks_cost
            production_queue_op[3].append(self.barracks_time)

        obs = [minerals, population, building, production_queue, minerals_op, \
                population_op, building_op, production_queue_op, rewards, steps]
        return obs

    def step(self, obs, action, player):
        """
        NO_OP = 0
        MAKE_SCV = 1
        BUILD_DEPOT = 2
        BUILD_BARRACKS = 3
        MAKE_MARINE = 4
        """
        copy_obs = deepcopy(obs)
        gameover = self.check_gameover(copy_obs)
        if gameover:
            print("The game is over")
            return copy_obs, gameover
        else:
            available_action = self.check(copy_obs, player)
            # print(f"available action {available_action}")

            if action in available_action:
                # update minerals
                # update population
                # update building
                # update production queue
                # update minerals rate
                if action == 0:
                    obs = self.no_op(copy_obs, player)
                elif action == 1:
                    obs = self.make_scv(copy_obs, player)
                elif action == 2:
                    obs = self.build_depot(copy_obs, player)
                elif action == 3:
                    obs = self.build_barracks(copy_obs, player)
                elif action == 4:
                    obs = self.make_marine(copy_obs, player)
                return obs, gameover
            else:
                print(f"There is an unavailable action -{action}- for obs now")

    def get_value(self, obs, player):
        reward =  obs[8][0] - obs[8][1]
        if reward == 0:
            return 0
        else:
            if player == 1:
                value = 1 if reward > 0 else -1
            if player == -1:
                value = 1 if reward < 0 else -1
            return value

if __name__ == '__main__':
    game = BMGame()
    obs = game.reset()
    player = 1
    for i in range(6000):
        available_action = game.check(obs,player)
        aciton = random.choice(available_action)
        obs, gameover = game.step(obs, aciton, player)
        if gameover:
            break
        player *= -1
    print(obs)