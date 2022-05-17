import random
from enum import Enum
import numpy as np
from collections import namedtuple

from pysc2.agents import base_agent
from pysc2.lib.actions import FUNCTIONS as F
from pysc2.lib.units import Terran
from s2clientprotocol.error_pb2 import ActionResult


class BMInfo:
    class LOCATION:
        '''locations in the map of 84 * 84'''
        # second mineral
        mineral = (14, 14)
        # command center
        cc = (35, 29)
        # attack location
        attack = (60, 29)
        # supply depots
        depot = [(20 + 7 * i, 5 + 7 * j) for j in range(2) for i in range(9)]
        # barracks
        barracks = [(15 + 11 * i, 60) for i in range(7)] \
                   + [(26 + 11 * i, 49) for i in range(6)] \
                   + [(48 + 11 * i, 38) for i in range(4)]

    class COST:
        '''cost of units'''
        scv = 50  # SCV
        marine = 50  # Marine
        depot = 100  # Supply Depot
        barracks = 150  # Barracks

    class TIME:
        '''time requried to build units (in secs)'''
        scv = 17
        marine = 25
        depot = 30
        barracks = 60

    max_depot = len(LOCATION.depot)
    max_barracks = len(LOCATION.barracks)


class BMAction(Enum):
    NO_OP = 0
    MAKE_SCV = 1
    BUILD_DEPOT = 2
    BUILD_BARRACKS = 3
    MAKE_MARINE = 4
    KILL_MARINE = 5


BMState = namedtuple('BMState', ['n_scv', 'n_marine', 'n_depot', 'n_barracks',
                                 'mineral', 'food_used', 'food_cap', 'food_workers', 'army_count'])


# setting for two player in one zero-game
class BaseBMAgent(base_agent.BaseAgent):
    def __init__(self):
        super().__init__()
        # whether in progress
        # -1 means not initiated
        self.in_progress = -1
        self.in_progress_mirror = -1
        # stage of a progress
        self.progress_stage = 0
        self.progress_stage_mirror = 0
        # number of SCVs, including those queued
        # number of current SCVs is `player.food_workers`
        self.n_scv = 12
        self.n_scv_mirror = 12
        # number of marines, including those queued
        # number of current marines is `player.army_count`
        self.n_marine = 0
        self.n_marine_mirror = 0
        # number of depots, including those being built
        # number of current SCVs is (player.food_cap - 15) // 8
        self.n_depot = 0
        self.n_depot_mirror = 0
        # number of barracks, including those being built
        self.n_barracks = 0
        self.n_barracks_mirror = 0
        # index of next building SCV
        self.work_idx = -1
        self.work_idx_mirror = -1
        # whether any barracks is finished
        self.barracks_finished = False
        self.barracks_finished_mirror = False
        # reward of current step
        self.step_reward = 0
        self.step_reward_mirror = 0

        self.scv_group = 0
        self.n_state = len(BMState._fields)*2
        self.n_action = len(BMAction)

    def reset(self):
        super().reset()
        self.reward = self.step_reward = 0
        self.in_progress = -1
        self.progress_stage = 0
        self.n_scv = 12
        self.n_marine = self.n_depot = self.n_barracks = 0
        self.work_idx = -1
        self.barracks_finished = False

        self.reward_mirror = self.step_reward_mirror = 0
        self.in_progress_mirror = -1
        self.progress_stage_mirror = 0
        self.n_scv_mirror = 12
        self.n_marine_mirror = self.n_depot_mirror = self.n_barracks_mirror = 0
        self.work_idx_mirror = -1
        self.barracks_finished_mirror = False

    def initiate(self, obs, mirror):
        '''select all SCVs to group 0'''
        if obs.observation.player.idle_worker_count > 0:
            # Below is the case when a bug happens.
            # Some minerals are missing and SCVs not mining.
            raise ValueError('A bug happened! Mineral missing!')
        if not mirror:
            if self.progress_stage == 0:
                self.progress_stage += 1
                return F.select_rect('select', (0, 0), (83, 83))
            else:
                self._finish(mirror)
                return F.select_control_group('set', self.scv_group)
        elif mirror:
            if self.progress_stage_mirror == 0:
                self.progress_stage_mirror += 1
                return F.select_rect('select', (0, 0), (83, 83))
            else:
                self._finish(mirror)
                return F.select_control_group('set', self.scv_group)
    
    # 每个真实动作完成后，agent.in_progress = 0
    def _finish(self, mirror, r=0.):
        if not mirror:
            self.in_progress = BMAction.NO_OP
            self.progress_stage = 0
            self.step_reward = r
        elif mirror:
            self.in_progress_mirror = BMAction.NO_OP
            self.progress_stage_mirror = 0
            self.step_reward_mirror = r

    ############################################################
    ## Below implement the actions defined in `BMAction`      ##
    ############################################################

    def select_scv(self, obs, mirror):
        '''select one SCV
        1. select group
        2. select next SCV
        '''
        if not mirror:
            if self.progress_stage == 0:
                self.progress_stage += 1
                return F.select_control_group('recall', self.scv_group)
            else:
                self.progress_stage += 1
                self.work_idx = (self.work_idx + 1) % 12
                return F.select_unit('select', self.work_idx)
        elif mirror:
            if self.progress_stage_mirror == 0:
                self.progress_stage_mirror += 1
                return F.select_control_group('recall', self.scv_group)
            else:
                self.progress_stage_mirror += 1
                self.work_idx_mirror = (self.work_idx_mirror + 1) % 12
                return F.select_unit('select', self.work_idx_mirror)

    def make_scv(self, obs, mirror):
        '''make one SCV
        1. select commander center
        2. build SCV if possible
        '''
        if not mirror:
            if self.progress_stage == 0:
                self.progress_stage += 1
                return F.select_point('select', BMInfo.LOCATION.cc)
            else:
                if F.Train_SCV_quick.id not in obs.observation.available_actions:
                    self._finish(mirror, -0.1)
                    return F.no_op()

                self._finish(mirror, 0.5)
                self.n_scv += 1
                player = obs.observation.player
                print(f'Make one SCV! Total {self.n_scv}, current {player.food_workers}')
                return F.Train_SCV_quick('now')
        elif mirror:
            if self.progress_stage_mirror == 0:
                self.progress_stage_mirror += 1
                return F.select_point('select', BMInfo.LOCATION.cc)
            else:
                if F.Train_SCV_quick.id not in obs.observation.available_actions:
                    self._finish(mirror, -0.1)
                    return F.no_op()

                self._finish(mirror, 0.5)
                self.n_scv_mirror += 1
                player = obs.observation.player
                print(f'Mirror Make one SCV! Total {self.n_scv_mirror}, current {player.food_workers}')
                return F.Train_SCV_quick('now')

    def build_depot(self, obs, mirror):
        '''build one supply depot
        1. select a SCv
        2. build depot
        3. send back to mining
        '''
        if not mirror:
            if self.progress_stage < 2:
                return self.select_scv(obs, mirror)
            elif self.progress_stage == 2:
                if F.Build_SupplyDepot_screen.id not in obs.observation.available_actions \
                        or self.n_depot >= BMInfo.max_depot:
                    self._finish(mirror, -0.1)
                    return F.no_op()

                build_loc = BMInfo.LOCATION.depot[self.n_depot]
                self.progress_stage += 1
                self.n_depot += 1
                player = obs.observation.player
                print(f'Build one depot! Total {self.n_depot}, current {(player.food_cap - 15) // 8}')
                return F.Build_SupplyDepot_screen('now', build_loc)
            else:
                self._finish(-0.1)
                return F.Harvest_Gather_screen('queued', BMInfo.LOCATION.mineral)
        elif mirror:
            if self.progress_stage_mirror < 2:
                return self.select_scv(obs, mirror)
            elif self.progress_stage_mirror == 2:
                if F.Build_SupplyDepot_screen.id not in obs.observation.available_actions \
                        or self.n_depot_mirror >= BMInfo.max_depot:
                    self._finish(mirror, -0.1)
                    return F.no_op()

                build_loc = BMInfo.LOCATION.depot[self.n_depot_mirror]
                self.progress_stage_mirror += 1
                self.n_depot_mirror += 1
                player = obs.observation.player
                print(f'Mirror Build one depot! Total {self.n_depot_mirror}, current {(player.food_cap - 15) // 8}')
                return F.Build_SupplyDepot_screen('now', build_loc)
            else:
                self._finish(mirror, -0.1)
                return F.Harvest_Gather_screen('queued', BMInfo.LOCATION.mineral)


    def build_barracks(self, obs, mirror):
        '''build one supply depot
        1. select a SCV
        2. build barracks
        3. send back to mining
        '''
        if not mirror:
            if self.progress_stage < 2:
                return self.select_scv(obs, mirror)
            elif self.progress_stage == 2:
                if F.Build_Barracks_screen.id not in obs.observation.available_actions \
                        or self.n_barracks >= BMInfo.max_barracks:
                    self._finish(mirror, -0.1)
                    return F.no_op()

                build_loc = BMInfo.LOCATION.barracks[self.n_barracks]
                self.progress_stage += 1
                self.n_barracks += 1
                print(f'Build one barracks! Total {self.n_barracks}')
                return F.Build_Barracks_screen('now', build_loc)
            else:
                self._finish(mirror, 0.1)
                return F.Harvest_Gather_screen('queued', BMInfo.LOCATION.mineral)
        elif mirror:
            if self.progress_stage_mirror < 2:
                return self.select_scv(obs, mirror)
            elif self.progress_stage_mirror == 2:
                if F.Build_Barracks_screen.id not in obs.observation.available_actions \
                        or self.n_barracks_mirror >= BMInfo.max_barracks:
                    self._finish(mirror, -0.1)
                    return F.no_op()

                build_loc = BMInfo.LOCATION.barracks[self.n_barracks_mirror]
                self.progress_stage_mirror += 1
                self.n_barracks_mirror += 1
                print(f'Mirror Build one barracks! Total {self.n_barracks_mirror}')
                return F.Build_Barracks_screen('now', build_loc)
            else:
                self._finish(mirror, 0.1)
                return F.Harvest_Gather_screen('queued', BMInfo.LOCATION.mineral)

    def make_marine(self, obs, mirror):
        '''make marines
        1. select all barracks
        2. make marines
        '''
        if not mirror:
            if self.progress_stage == 0:
                self.progress_stage += 1
                return F.select_point('select_all_type', BMInfo.LOCATION.barracks[0])
            else:
                if F.Train_Marine_quick.id not in obs.observation.available_actions:
                    self._finish(mirror, -0.1)
                    return F.no_op()
                self._finish(mirror, 1)
                self.n_marine += 1
                player = obs.observation.player
                print(f'Make one marine! Total {self.n_marine}, current {player.army_count}, reward {self.reward}')
                return F.Train_Marine_quick('now')
        elif mirror:
            if self.progress_stage_mirror == 0:
                self.progress_stage_mirror += 1
                return F.select_point('select_all_type', BMInfo.LOCATION.barracks[0])
            else:
                if F.Train_Marine_quick.id not in obs.observation.available_actions:
                    self._finish(mirror, -0.1)
                    return F.no_op()
                self._finish(mirror, 1)
                self.n_marine_mirror += 1
                player = obs.observation.player
                print(f'Mirror Make one marine! Total {self.n_marine_mirror}, current {player.army_count}, reward {self.reward_mirror}')
                return F.Train_Marine_quick('now')

    def kill_marine(self, obs, mirror):
        '''kill marines
        1. select army
        2. attack a specific point
        '''
        if not mirror:
            if self.progress_stage == 0:
                self.progress_stage += 1
                if F.select_army.id not in obs.observation.available_actions:
                    self._finish(mirror, -0.1)
                    return F.no_op()
                return F.select_army('select')
            else:
                reward = (obs.observation.player.army_count - 3) / 10
                self._finish(mirror, reward)
                self.n_marine = len(obs.observation.multi_select)
                return F.Attack_screen('queued', BMInfo.LOCATION.attack)
        elif mirror:
            if self.progress_stage_mirror == 0:
                self.progress_stage_mirror += 1
                if F.select_army.id not in obs.observation.available_actions:
                    self._finish(mirror, -0.1)
                    return F.no_op()
                return F.select_army('select')
            else:
                reward = (obs.observation.player.army_count - 3) / 10
                self._finish(mirror, reward)
                self.n_marine_mirror = len(obs.observation.multi_select)
                return F.Attack_screen('queued', BMInfo.LOCATION.attack)

    def choose_act(self, obs, mirror):
        if not mirror:
            if self.in_progress == BMAction.MAKE_SCV:
                return self.make_scv(obs, mirror), 1
            elif self.in_progress == BMAction.BUILD_DEPOT:
                return self.build_depot(obs, mirror), 2
            elif self.in_progress == BMAction.BUILD_BARRACKS:
                return self.build_barracks(obs, mirror), 3
            elif self.in_progress == BMAction.MAKE_MARINE:
                return self.make_marine(obs, mirror), 4
            elif self.in_progress == BMAction.KILL_MARINE:
                return self.kill_marine(obs, mirror), 5
            elif self.in_progress == -1:
                return self.initiate(obs, mirror), -1
            return F.no_op(), 0
        elif mirror:
            if self.in_progress_mirror == BMAction.MAKE_SCV:
                return self.make_scv(obs, mirror), 1
            elif self.in_progress_mirror == BMAction.BUILD_DEPOT:
                return self.build_depot(obs, mirror), 2
            elif self.in_progress_mirror == BMAction.BUILD_BARRACKS:
                return self.build_barracks(obs, mirror), 3
            elif self.in_progress_mirror == BMAction.MAKE_MARINE:
                return self.make_marine(obs, mirror), 4
            elif self.in_progress_mirror == BMAction.KILL_MARINE:
                return self.kill_marine(obs, mirror), 5
            elif self.in_progress_mirror == -1:
                return self.initiate(obs, mirror), -1
            return F.no_op(), 0

    def execute_act(self, obs, mirror):
        if not mirror:
            if self.in_progress == BMAction.MAKE_SCV:
                return self.make_scv(obs, mirror), 1
            elif self.in_progress == BMAction.BUILD_DEPOT:
                return self.build_depot(obs, mirror), 2
            elif self.in_progress == BMAction.BUILD_BARRACKS:
                return self.build_barracks(obs, mirror), 3
            elif self.in_progress == BMAction.MAKE_MARINE:
                return self.make_marine(obs, mirror), 4
            elif self.in_progress == BMAction.KILL_MARINE:
                return self.kill_marine(obs, mirror), 5
            elif self.in_progress == -1:
                return self.initiate(obs, mirror), -1
            return F.no_op(), 0
        elif mirror:
            if self.in_progress_mirror == BMAction.MAKE_SCV:
                return self.make_scv(obs, mirror), 1
            elif self.in_progress_mirror == BMAction.BUILD_DEPOT:
                return self.build_depot(obs, mirror), 2
            elif self.in_progress_mirror == BMAction.BUILD_BARRACKS:
                return self.build_barracks(obs, mirror), 3
            elif self.in_progress_mirror == BMAction.MAKE_MARINE:
                return self.make_marine(obs, mirror), 4
            elif self.in_progress_mirror == BMAction.KILL_MARINE:
                return self.kill_marine(obs, mirror), 5
            elif self.in_progress_mirror == -1:
                return self.initiate(obs, mirror), -1
            return F.no_op(), 0


    ############################################################
    ## End of actions                                         ##
    ## Below functions check whether an action is available   ##
    ############################################################

    def check_make_scv(self, player, mirror):
        if not mirror:
            return (player.minerals >= BMInfo.COST.scv  # mineral
                    and player.food_used < player.food_cap  # supply
                    and self.n_scv - player.food_workers < 5)  # limit of cc queue is 5
        elif mirror:
            return (player.minerals >= BMInfo.COST.scv  # mineral
                    and player.food_used < player.food_cap  # supply
                    and self.n_scv_mirror - player.food_workers < 5)  # limit of cc queue is 5

    def check_build_depot(self, player, mirror):
        if not mirror:
            return (player.minerals >= BMInfo.COST.depot  # mineral
                    and self.n_depot < BMInfo.max_depot)  # limit
        elif mirror:
            return (player.minerals >= BMInfo.COST.depot  # mineral
                    and self.n_depot_mirror < BMInfo.max_depot)  # limit

    def check_build_barracks(self, player, mirror):
        if not mirror:
            return (player.minerals >= BMInfo.COST.barracks  # mineral
                    and self.n_barracks < BMInfo.max_barracks  # limit
                    and player.food_cap >= 23)  # at least one depot is built
        elif mirror:
            return (player.minerals >= BMInfo.COST.barracks  # mineral
                    and self.n_barracks_mirror < BMInfo.max_barracks  # limit
                    and player.food_cap >= 23)  # at least one depot is built

    def check_make_marine(self, player, mirror):
        if not mirror:
            return (player.minerals >= BMInfo.COST.marine  # mineral
                    and player.food_used < player.food_cap  # supply
                    and self.barracks_finished)  # at least one barracks is built
        elif mirror:
            return (player.minerals >= BMInfo.COST.marine  # mineral
                    and player.food_used < player.food_cap  # supply
                    and self.barracks_finished_mirror)  # at least one barracks is built

    def check_kill_marine(self, player, mirror):
        return player.army_count > 0  # at least one marine

    ############################################################
    ## End of checker functions                               ##
    ############################################################

    def step(self, obs, mirror):
        # super().step(obs)

        # print warning
        act_result = obs.observation.action_result
        if len(act_result) > 0:
            print(f'** Warning: {ActionResult.Name(act_result[0])}! **')

        # check barracks
        feature_screen = obs.observation.feature_screen
        x, y = BMInfo.LOCATION.barracks[0]
        if not mirror:
            self.reward += obs.reward
            if feature_screen.unit_type[y, x] == Terran.Barracks \
                    and feature_screen.build_progress[y, x]== 0:
                self.barracks_finished = True
        elif mirror:
            self.reward_mirror += obs.reward
            if feature_screen.unit_type[y, x] == Terran.Barracks \
                    and feature_screen.build_progress[y, x]== 0:
                self.barracks_finished_mirror = True
        return F.no_op()

    def get_state(self, obs):
        '''get current state'''
        player = obs.observation['player']
        return np.array([self.n_scv / 10, self.n_marine / 10, self.n_depot / 5, self.n_barracks / 5,
                    player.minerals / 1000, player.food_used / 10, player.food_cap / 10,
                    player.food_workers / 10, player.army_count / 10])
        


class ScriptBMAgent(BaseBMAgent):
    def __init__(self, max_scv=30, max_depot=5, max_barracks=10):
        super().__init__()

        self.max_scv = max_scv
        self.max_depot = max_depot
        self.max_barracks = max_barracks

    def step(self, obs):
        super().step(obs)
        player = obs.observation.player

        # 1. check whether a supply depot is needed
        # 2. check whether a SCV can be made
        # 3. check whether a barracks can be built
        # 4. check whether need to kill marines
        # 5. check whether marines can be made
        # 6. no op
        if self.in_progress == BMAction.NO_OP:
            if self.check_build_depot(player) \
                    and player.food_used >= player.food_cap - 2 \
                    and self.n_depot < self.max_depot:
                self.in_progress = BMAction.BUILD_DEPOT
            elif self.check_make_scv(player) \
                    and self.n_scv < self.max_scv:
                self.in_progress = BMAction.MAKE_SCV
            elif self.check_build_barracks(player) \
                    and self.n_barracks < self.max_barracks:
                self.in_progress = BMAction.BUILD_BARRACKS
            elif player.army_count > 5:
                self.in_progress = BMAction.KILL_MARINE
            elif self.check_make_marine(player):
                self.in_progress = BMAction.MAKE_MARINE

        # continue previous action
        return self.choose_act(obs)


class RandomBMAgent(BaseBMAgent):
    def __init__(self):
        super().__init__()

    def step(self, obs):
        super().step(obs)

        player = obs.observation.player
        checkers = [
            (lambda x: True),
            self.check_make_scv,
            self.check_build_depot,
            self.check_build_barracks,
            self.check_make_marine,
            self.check_kill_marine
        ]
        choices = [i for i in range(self.n_action) if checkers[i](player)]
        if self.in_progress == BMAction.NO_OP:
            self.in_progress = BMAction(random.choice(choices))
        return self.choose_act(obs)
