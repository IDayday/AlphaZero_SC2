import numpy as np
from enum import Enum

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
        self.total_mainerals = 10000
        self.total_depots = 20
        self.total_barracks = 20

        # actions
        self.actions = 5

        # reward
        self.reward = 0
        self.reward_op = 0
    
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

        # mainerals rate
        if self.scv_population <= 16:
            self.mainerals_rate = 1.25
        else:
            self.mainerals_rate = 1.2
        self.mainerals = [50, self.mainerals_rate]


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

        # op mainerals rate
        if self.scv_population_op <= 16:
            self.mainerals_rate_op = 1.25
        else:
            self.mainerals_rate_op = 1.2
        self.mainerals_op = [50, self.mainerals_rate_op]

        # reward
        self.rewrads = [self.reward, self.reward_op]

        obs = [self.mainerals, self.population, self.building, self.production_queue, \
                self.mainerals_op, self.population_op, self.building_op, self.production_queue_op,\
                self.rewrads]
        return obs

    def check_make_scv(self, obs, player):
        mainerals = obs[0]
        population = obs[1]
        production_queue = obs[3]

        mainerals_op = obs[4]
        population_op = obs[5]
        production_queue_op = obs[7]

        if player == 1:
            return (mainerals[0] >= self.scv_cost 
                    and population[1] < population[0]
                    and len(production_queue[0]) < 5) # limit of cc queue is 5
        elif player == -1:
            return (mainerals_op[0] >= self.scv_cost 
                    and population_op[1] < population_op[0]
                    and len(production_queue_op[0]) < 5) # limit of cc queue is 5

    def check_build_depot(self, obs, player):
        mainerals = obs[0]
        building = obs[2]
        production_queue = obs[3]

        mainerals_op = obs[4]
        building_op = obs[6]
        production_queue_op = obs[7]

        total_depots = building[0] + len(production_queue[2]) + \
                       building_op[0] + len(production_queue_op[2])
        if player == 1:
            return (mainerals[0] >= self.depot_cost
                    and total_depots < self.total_depots)
        elif player == -1:
            return (mainerals_op[0] >= self.depot_cost
                    and total_depots < self.total_depots)
    
    def check_build_barracks(self, obs, player):
        mainerals = obs[0]
        building = obs[2]
        production_queue = obs[3]

        mainerals_op = obs[4]
        building_op = obs[6]
        production_queue_op = obs[7]

        total_barracks = building[1] + len(production_queue[3]) + \
                         building_op[1] + len(production_queue_op[3])
        if player == 1:
            return (mainerals[0] >= self.barracks_cost
                    and total_barracks < self.total_barracks
                    and building[0] >= 1) # at least one depot is built
        elif player == -1:
            return (mainerals_op[0] >= self.barracks_cost
                    and total_barracks < self.total_barracks
                    and building_op[0] >= 1) # at least one depot is built
    
    def check_make_marine(self, obs, player):
        mainerals = obs[0]
        population = obs[1]
        building = obs[2]

        mainerals_op = obs[4]
        population_op = obs[5]
        building_op = obs[6]

        if player == 1:
            return (mainerals[0] >= self.marine_cost
                    and population[1] < population[0]
                    and building[1] > 0) # at least one barracks is built
        elif player == -1:
            return (mainerals_op[0] >= self.marine_cost
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

    def update_mainerals_rate(self,population):
        if population[2] <= 16:
            mainerals_rate = 1.25
        else:
            mainerals_rate = 1.2
        return mainerals_rate

    def no_op(self, obs, player):
        mainerals = obs[0]
        population = obs[1]
        building = obs[2]
        production_queue = obs[3]

        mainerals_op = obs[4]
        population_op = obs[5]
        building_op = obs[6]
        production_queue_op = obs[7]

        rewards = obs[8]

        if player == 1:
            mainerals[0] += round(population[2]*mainerals[1],0)
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
                            mainerals_rate = self.update_mainerals_rate(population)
                            mainerals[1] = mainerals_rate
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
            mainerals_op[0] += round(population_op[2]*mainerals_op[1],0)
            new_queue = []
            for cls,p in enumerate(production_queue_op):
                p = [x-1 for x in p]
                if len(p) > 0:
                    # unit cls has been completed, because sleek, pop first
                    if p[0] == 0:
                        p.pop(0)
                        # a new scv add to population
                        if cls == 0:
                            population_op[1] += 1
                            population_op[2] += 1
                            mainerals_rate_op = self.update_mainerals_rate(population_op)
                            mainerals_op[1] = mainerals_rate_op
                        # a new marine add to population
                        elif cls == 1:
                            population_op[1] += 1
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
        obs = [mainerals, population, building, production_queue, mainerals_op, \
                population_op, building_op, production_queue_op, rewards]
        return obs

    def make_scv(self, obs, player):
        # first do no_op
        obs = self.no_op(obs, player)
        # then add a scv to the production queue and total population plus one
        mainerals = obs[0]
        population = obs[1]
        building = obs[2]
        production_queue = obs[3]

        mainerals_op = obs[4]
        population_op = obs[5]
        building_op = obs[6]
        production_queue_op = obs[7]

        rewards = obs[8]

        if player == 1:
            mainerals[0] -= self.scv_cost
            production_queue[0].append(self.scv_time)
            population[1] += 1
        elif player == -1:
            mainerals_op[0] -= self.scv_cost
            production_queue_op[0].append(self.scv_time)
            population_op[1] += 1
        obs = [mainerals, population, building, production_queue, mainerals_op, \
                population_op, building_op, production_queue_op, rewards]
        return obs

    def make_marine(self, obs, player):
        # first do no_op
        obs = self.no_op(obs, player)
        # then add a marine to the production queue and total population plus one
        mainerals = obs[0]
        population = obs[1]
        building = obs[2]
        production_queue = obs[3]

        mainerals_op = obs[4]
        population_op = obs[5]
        building_op = obs[6]
        production_queue_op = obs[7]

        rewards = obs[8]

        if player == 1:
            mainerals[0] -= self.marine_cost
            production_queue[1].append(self.marine_time)
            population[1] += 1
        elif player == -1:
            mainerals_op[0] -= self.marine_cost
            production_queue_op[1].append(self.marine_time)
            population_op[1] += 1
        obs = [mainerals, population, building, production_queue, mainerals_op, \
                population_op, building_op, production_queue_op, rewards]
        return obs

    def build_depot(self, obs, player):
        # first do no_op
        obs = self.no_op(obs, player)
        # then add a depot to the production queue
        mainerals = obs[0]
        population = obs[1]
        building = obs[2]
        production_queue = obs[3]

        mainerals_op = obs[4]
        population_op = obs[5]
        building_op = obs[6]
        production_queue_op = obs[7]

        rewards = obs[8]

        if player == 1:
            mainerals[0] -= self.depot_cost
            production_queue[2].append(self.depot_time)
        elif player == -1:
            mainerals_op[0] -= self.depot_cost
            production_queue_op[2].append(self.depot_time)
        obs = [mainerals, population, building, production_queue, mainerals_op, \
                population_op, building_op, production_queue_op, rewards]
        return obs

    def build_barracks(self, obs, player):
        # first do no_op
        obs = self.no_op(obs, player)
        # then add a barracks to the production queue
        mainerals = obs[0]
        population = obs[1]
        building = obs[2]
        production_queue = obs[3]

        mainerals_op = obs[4]
        population_op = obs[5]
        building_op = obs[6]
        production_queue_op = obs[7]

        rewards = obs[8]

        if player == 1:
            mainerals[0] -= self.barracks_cost
            production_queue[3].append(self.barracks_time)
        elif player == -1:
            mainerals_op[0] -= self.barracks_cost
            production_queue_op[3].append(self.barracks_time)

        obs = [mainerals, population, building, production_queue, mainerals_op, \
                population_op, building_op, production_queue_op, rewards]
        return obs

    def step(self, obs, action, player):
        """
        NO_OP = 0
        MAKE_SCV = 1
        BUILD_DEPOT = 2
        BUILD_BARRACKS = 3
        MAKE_MARINE = 4
        """

        available_action = self.check(obs, player)
        print(f"available action {available_action}")

        if action in available_action:
            # update mainerals
            # update population
            # update building
            # update production queue
            # update mainerals rate
            if action == 0:
                obs = self.no_op(obs, player)
            elif action == 1:
                obs = self.make_scv(obs, player)
            elif action == 2:
                obs = self.build_depot(obs, player)
            elif action == 3:
                obs = self.build_barracks(obs, player)
            elif action == 4:
                obs = self.make_marine(obs, player)
            return obs
        else:
            print(f"There is an unavailable action -{action}- for obs now")


if __name__ == '__main__':
    game = BMGame()
    obs = game.reset()
    action = [1,0,0,0,0,0,0,2,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,0,0,0,0,3,\
                1,1,2,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,2,0,0,0,0,0,1,0,1,0,1,0,1,3,1,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,\
                0,0,0,0,0,0,0,0,0,0,0,0,0,4]
    player = 1
    for a in action:
        obs = game.step(obs, a, player)
    print(obs)