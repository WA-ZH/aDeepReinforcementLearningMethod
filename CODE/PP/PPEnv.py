
from dataclasses import dataclass
import torch
import numpy as np
from PProblemDef import get_random_problems


@dataclass
class Reset_State:

    static: torch.Tensor = None
    # shape: (batch, problem, 5)
    node_demand: torch.Tensor = None
    # shape: (batch, problem)


@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor = None
    POMO_IDX: torch.Tensor = None
    # shape: (batch, pomo)
    selected_count: int = None
    load: torch.Tensor = None
    # shape: (batch, pomo)
    current_node: torch.Tensor = None
    # shape: (batch, pomo)
    ninf_mask: torch.Tensor = None
    # shape: (batch, pomo, problem+1)
    finished: torch.Tensor = None
    # shape: (batch, pomo)


class CVRPEnv:
    def __init__(self, **env_params):

        # Const @INIT
        ####################################
        self.env_params = env_params
        self.problem_size = env_params['problem_size']
        self.pomo_size = env_params['pomo_size']
        self.n_size = env_params['n_size']
        self.minusN = 1/self.n_size

        self.FLAG__use_saved_problems = False
        self.saved_depot_xy = None
        self.saved_node_xy = None
        self.saved_node_demand = None
        self.saved_index = None
        self.saved_static = None

        # Const @Load_Problem
        ####################################
        self.batch_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None
        # IDX.shape: (batch, pomo)
        self.depot_node_xy = None
        # shape: (batch, problem+1, 2)
        self.depot_node_demand = None
        # shape: (batch, problem+1)
        self.static = None


        # Dynamic-1
        ####################################
        self.selected_count = None
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = None
        # shape: (batch, pomo, 0~)

        # Dynamic-2
        ####################################

        self.load = None
        # shape: (batch, pomo)
        self.visited_ninf_flag = None
        # shape: (batch, pomo, problem+1)
        self.ninf_mask = None
        # shape: (batch, pomo, problem+1)
        self.finished = None
        # shape: (batch, pomo)

        # states to return
        ####################################
        self.reset_state = Reset_State()
        self.step_state = Step_State()

    def use_saved_problems(self, staticfilename, demandfilename, device):
        self.FLAG__use_saved_problems = True

        saved_static1 = np.loadtxt('../data/pp{}_test_data.txt'.format(self.problem_size), dtype=np.float, delimiter=',')
        saved_node_demand1 = np.loadtxt('../data/pp{}_demand.txt'.format(self.problem_size), dtype=np.float, delimiter=',')
        self.saved_static = torch.tensor(saved_static1, dtype=torch.float32).reshape(20,self.problem_size,5)
        self.saved_node_demand = torch.tensor(saved_node_demand1, dtype=torch.float32)
        self.saved_index = 0

    def load_problems(self, batch_size, aug_factor=1):
        self.batch_size = batch_size

        if not self.FLAG__use_saved_problems:
            static1, node_demand = get_random_problems(batch_size, self.problem_size, self.n_size, self.pomo_size)
        else:
            static1 = self.saved_static[self.saved_index:self.saved_index+batch_size]
            node_demand = self.saved_node_demand[self.saved_index:self.saved_index+batch_size]
            self.saved_index += batch_size


        self.static = static1



        self.depot_node_demand = node_demand
        # shape: (batch, problem+1)


        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)

        self.reset_state.static = static1
        self.reset_state.node_demand = node_demand

        self.step_state.BATCH_IDX = self.BATCH_IDX
        self.step_state.POMO_IDX = self.POMO_IDX

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long)
        # shape: (batch, pomo, 0~)

        Pre = self.static[:, :, 1].clone()
        PreZero = Pre.eq(0)
        firstIndex = PreZero.nonzero().squeeze()

        self.load = torch.ones(size=(self.batch_size, self.pomo_size))
        # shape: (batch, pomo)
        #self.visited_ninf_flag = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size))
        self.visited_ninf_flag = torch.empty((self.batch_size, self.pomo_size, self.problem_size)).fill_(float('-inf'))
        self.visited_ninf_flag[firstIndex[:, 0], :, firstIndex[:, 1]] = 0
        # shape: (batch, pomo, problem+1)
        #self.ninf_mask = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size))
        self.ninf_mask = torch.empty((self.batch_size, self.pomo_size, self.problem_size)).fill_(float('-inf'))
        self.ninf_mask[firstIndex[:, 0], :, firstIndex[:, 1]] = 0
        # shape: (batch, pomo, problem+1)
        self.finished = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.bool)
        # shape: (batch, pomo)

        reward = None
        done = False
        return self.reset_state, reward, done

    def pre_step(self):
        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished

        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, selected):
        # selected.shape: (batch, pomo)

        # Dynamic-1
        ####################################
        self.selected_count += 1
        self.current_node = selected
        # shape: (batch, pomo)
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)
        # shape: (batch, pomo, 0~)

        # Dynamic-2
        ####################################



        self.load -= self.minusN

        Cid = self.static[:,:,0].clone()
        Pre = self.static[:,:,1].clone()
        cid = Cid[:, None, :].expand(self.batch_size, self.pomo_size, -1)
        pid= Pre[:, None, :].expand(self.batch_size, self.pomo_size, -1)
        gathering_index = selected[:, :, None]
        select_cid = cid.gather(dim=2, index=gathering_index)
        select_pid = pid.gather(dim=2, index=gathering_index)

        chosen_index_cid = torch.nonzero(cid == select_cid, as_tuple=False)
        self.visited_ninf_flag[chosen_index_cid[:,0], chosen_index_cid[:,1], chosen_index_cid[:,2]] = float('-inf')

        chosen_index_pid = torch.nonzero(pid == select_cid, as_tuple=False)
        self.visited_ninf_flag[chosen_index_pid[:,0], chosen_index_pid[:,1], chosen_index_pid[:,2]] = float(0)

        #self.visited_ninf_flag[self.BATCH_IDX, self.POMO_IDX, selected] = float('-inf')
        # shape: (batch, pomo, problem+1)


        self.ninf_mask = self.visited_ninf_flag.clone()


        newly_finished = (self.visited_ninf_flag == float('-inf')).all(dim=2)
        # shape: (batch, pomo)
        self.finished = self.finished + newly_finished
        # shape: (batch, pomo)



        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished

        # returning values
        done = self.finished.all()
        if done:
            reward = -self._get_travel_distance()  # note the minus sign!
        else:
            reward = None

        return self.step_state, reward, done

    def _get_travel_distance(self):
        gathering_index = self.selected_node_list[:, :, :, None].expand(-1, -1, -1, 5)
        # shape: (batch, pomo, selected_list_length, 2)
        all_xy = self.static[:, None, :, :].expand(-1, self.pomo_size, -1, -1)
        # shape: (batch, pomo, problem+1, 2)

        ordered_seq = all_xy.gather(dim=2, index=gathering_index)





        machine = ordered_seq[:,:,:,3].long()
        machineChage1 = machine[:, :, :-1]
        machineChage2 = machine[:, :, 1:]
        machineChage = machineChage1.ne(machineChage2)
        machineCost1 = machineChage.sum(2)
        machineCost = machineCost1 * 160


        tool = ordered_seq[:,:,:,4].long()
        toolChage1 = tool[:, :, :-1]
        toolChage2 = tool[:, :, 1:]
        toolChage = toolChage1.ne(toolChage2)
        toolCost1 = torch.logical_or(machineChage, toolChage)  #machineChage or toolChage
        toolCost2 = toolCost1.sum(2)
        toolCost = toolCost2 * 20


        setup = ordered_seq[:,:,:,2].long()
        setupChage1 = setup[:, :, :-1]
        setupChage2 = setup[:, :, 1:]
        setupChage = setupChage1.ne(setupChage2)
        setupCost1 = torch.logical_or(machineChage, setupChage) #machineChage or setupChage
        setupCost2 = setupCost1.sum(2) + 1
        setupCost = setupCost2 * 100


        machineList = self.machineCostList[None,None,:].expand(self.batch_size,self.pomo_size,-1)
        machinecostAll1 = machineList.gather(dim=2, index=machine)
        machinecostAll = machinecostAll1.sum(2)



        toolList = self.toolCostList[None, None, :].expand(self.batch_size, self.pomo_size, -1)
        toolcostAll1 = toolList.gather(dim=2, index=tool)
        toolcostAll = toolcostAll1.sum(2)



        cost = machinecostAll + toolcostAll + setupCost + machineCost + toolCost

        return cost

