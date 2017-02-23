from grid_world_car_Cane import Car
from grid_world_customer_Cane import Customer
import random
import numpy as np

class GridWorld(object):
    def __init__(self, grid_size = 5, num_cars = 3, cust_popup_episode = None, cust_wait_time = 5, demand_limit = 4,
        reward_per_cust = 1, penalty_per_move = -0.1, penalty_per_collision = -1,penalty_per_hitting_wall = -0.2, reward_per_stay = -0.1, terminal_reward = 10, customer_rate = 0.04, obstacle_grid = None,
        reward_backprop_rate = 0.9, reward_others_discount_rate = 0.0, num_actions = 5, unfulfilled_penalty = -1):
        
        self.grid_size = grid_size
        self.t = 0
        self.history = []
        self.customer_rate = customer_rate
        self.demand_init_prob = np.random.random((self.grid_size, self.grid_size)) * self.customer_rate
        self.cust_popup_episode = cust_popup_episode
        self.cust_popup_history = []
        self.reward_per_cust = reward_per_cust
        self.penalty_per_move = penalty_per_move
        self.penalty_per_collision = penalty_per_collision
        self.penalty_per_hitting_wall = penalty_per_hitting_wall
        self.unfulfilled_penalty = unfulfilled_penalty
        self.reward_per_stay = reward_per_stay
        self.reward_backprop_rate = reward_backprop_rate
        self.reward_others_discount_rate = reward_others_discount_rate
        self.num_actions = num_actions
        self.system_reward = 0
        self.maxWaitTime = cust_wait_time # + self.grid_size 
        self.minWaitTime = cust_wait_time 
        self.demand_limit = demand_limit
        self.num_cars = num_cars
        self.terminal_reward = terminal_reward
        self.observation_shape = self.grid_size * self.grid_size * 4
        
        self.obstacle_grid = self.loadObstacleGrid(obstacle_grid)
        self.cars_list = self.initiate_cars(self.num_cars)
        self.demand_list = []
        self.initiate_cust()  
        
    def reset(self,cust_popup_episode = None, cars_list = None):
        self.t = 0
        self.history = []
        self.demand_init_prob = np.random.random((self.grid_size, self.grid_size)) * self.customer_rate
        self.cust_popup_episode = cust_popup_episode
        self.cust_popup_history = []
        self.system_reward = 0
        if cars_list != None:
            #Using 
            self.cars_list = self.initiate_static_cars(cars_list)
        else:
            self.cars_list = self.initiate_cars(self.num_cars)
        self.demand_list = []
        self.initiate_cust()
    # ----------------------------------------------------------------Intentions/Moves and Events  
    def stepAll(self, model,epsilon):            
        intended_actions = self.intended_step(model,epsilon)
        all_agents_actual_step = self.actual_step(intended_actions)
        return all_agents_actual_step
    
    def convert_action_to_loc(self, car, action):
        curr_loc = car.location
        new_loc = curr_loc
        if action == 0:
            pass
        elif action == 1: # left
            new_loc = (curr_loc[0], curr_loc[1] - 1)
        elif action == 2: # up
            new_loc = (curr_loc[0]-1, curr_loc[1])
        elif action == 3: # right
            new_loc = (curr_loc[0], curr_loc[1] + 1)
        elif action == 4: # down
            new_loc = (curr_loc[0]+1, curr_loc[1])
        else:
            print "ERROR on INTENDED ACTION. Returning current location"
        return new_loc
    
    def intended_step(self,model,epsilon):
        intended_actions = {}
        for car in self.cars_list:
            state = self.get_state_view(car.carID, self.cars_grid, self.cust_grid)
            if model == 'Naive':
                action = car.choose_action(state,model,epsilon,self.num_actions,self.observation_shape,demand = self.getCustomerlocs())
            else:
                action = car.choose_action(state,model,epsilon,self.num_actions,self.observation_shape)
            intended_actions[car] = (state, action)
        return intended_actions
            
    def actual_step(self,intended_actions):
        all_agents_step = []
        for car in self.cars_list:
            agent_step = {}
            agent_step['car_obj'] = car
            agent_step['carID'] = car.carID
            agent_step['original_location'] = car.location
            
            state = intended_actions[car][0]
            action = intended_actions[car][1]
            agent_step = self.makeMove(car,action,agent_step)
            
            agent_step['action'] = action
            agent_step['state'] = state
            all_agents_step.append(agent_step)
        
        self.update_demand() 
        for agent_step in all_agents_step:
            car = agent_step['car_obj'] 
            new_state = self.get_state_view(car.carID, self.cars_grid, self.cust_grid)
            agent_step['new_state'] = new_state
            
        return all_agents_step
    
    def makeMove(self,Car,action,agent_step):
        car_locations = self.getCarlocs()
        customer_locations = self.getCustomerlocs()
        (x,y) = self.convert_action_to_loc(Car, action)
        agent_step['intended_location'] = (x,y)
        
        #Out of the grid
        if x not in range(self.grid_size) or y not in range(self.grid_size):
            agent_step['event'] = 'hit_wall'
            agent_step['new_location'] = agent_step['original_location']
            agent_step['reward'] = self.penalty_per_hitting_wall
            
        #Collision with other agent or wall
        elif ((x,y) in car_locations and (x,y) != agent_step['original_location']) or self.obstacle_grid[x, y] == 1:
            agent_step['event'] = 'collision'
            agent_step['new_location'] = agent_step['original_location']
            agent_step['reward'] = self.penalty_per_collision
            
        #Pick up customer
        elif (x,y) in customer_locations:
            agent_step['event'] = 'pickup'
            agent_step['new_location'] = (x,y)
            agent_step['reward'] = self.reward_per_cust
            self.move_car(Car, (x,y))
            self.cust_grid[(x,y)] = 0
            #Delete cutomer from demand_list
            customers_to_delete = []
            for index in range(len(self.demand_list)):
                Customer = self.demand_list[index]
                if (x,y) == Customer.location:
                    customers_to_delete.append(Customer)
                    self.system_reward += self.reward_per_cust
            self.removeCustomers(customers_to_delete)
                
        #Standard Movement
        else:
            if action == 0:
                agent_step['event'] = 'stay'
                agent_step['new_location'] = (x,y)
                agent_step['reward'] = self.reward_per_stay
            else: 
                agent_step['event'] = 'move'
                agent_step['new_location'] = (x,y)
                agent_step['reward'] = self.penalty_per_move
                self.move_car(Car, (x,y))    
        return agent_step
        
    # ---------------------------------------------------------------- Cars
    def randPair(self, s,e):
        return np.random.randint(s,e), np.random.randint(s,e)
    
    def get_car_random_loc(self):
        rand_loc = self.randPair(0,self.grid_size)
        if (self.cars_grid[rand_loc] != 0 or self.obstacle_grid[rand_loc] != 0):
            rand_loc = self.get_car_random_loc()
        return rand_loc
    
    def initiate_cars(self, num_cars):
        self.cars_grid = np.zeros((self.grid_size, self.grid_size))
        self.cars_list = []
        for i in range(1, num_cars + 1):
            rand_loc = self.get_car_random_loc()
            this_car = Car(i, rand_loc, self.grid_size, self.reward_backprop_rate, self.reward_others_discount_rate)
            self.cars_list.append(this_car)
            self.add_car_to_specific_loc(this_car, rand_loc)
        return self.cars_list 
    
    def initiate_static_cars(self, car_list):
        self.cars_grid = np.zeros((self.grid_size, self.grid_size))
        self.cars_list = []
        #i = 1
        for car in car_list:
            loc = car.location
            this_car = Car(car.carID, loc, self.grid_size, self.reward_backprop_rate, self.reward_others_discount_rate)
            self.cars_list.append(this_car)
            self.add_car_to_specific_loc(this_car, loc)
            #i += 1
        return self.cars_list 
         
    def add_car_to_specific_loc(self, car, loc):
        self.cars_grid[loc] = car.carID
        car.location = loc
    
    def move_car(self, car, new_loc):
        self.cars_grid[car.location] = 0
        self.add_car_to_specific_loc(car, new_loc)    
        
    def getCarlocs(self):
        locs = []
        for car in self.cars_list:
            locs.append(car.location)
        return locs
    
    # ---------------------------------------------------------------- Customers        
    def initiate_cust(self):
        self.cust_grid = np.zeros((self.grid_size, self.grid_size))
        if self.cust_popup_episode is None:
            self.add_new_cust()
        else:
            self.add_new_cust_from_popup_episode(0)
        
    def add_new_cust(self):
        cust_popup = np.zeros((self.grid_size, self.grid_size))
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if (self.cust_grid[i, j] == 0 and random.random() < self.demand_init_prob[i, j] and len(self.demand_list) < self.demand_limit and self.obstacle_grid[i, j] == 0): # and self.cars_grid[i, j] == 0):
                    random_waittime = np.random.randint(self.minWaitTime, self.maxWaitTime+1)
                    self.add_cust_to_specific_loc((i, j), random_waittime)
                    cust_popup[i, j] = random_waittime
        self.cust_popup_history.append(cust_popup)

    
    def add_new_cust_from_popup_episode(self, t):
        if t <= len(self.cust_popup_episode) - 1:
            popup_arr = self.cust_popup_episode[t]
            positions = np.where(popup_arr > 0)
            for i in range(len(positions[0])):
                loc = (positions[0][i], positions[1][i])
                waittime = popup_arr[loc]
                if (len(self.demand_list) < self.demand_limit and self.cust_grid[loc] == 0and self.obstacle_grid[loc] == 0):
                    self.add_cust_to_specific_loc(loc, waittime)
            self.cust_popup_history.append(popup_arr)
        else:
            pass # no more customers in the given demand episode
                
            
    def add_cust_to_specific_loc(self, loc, waittime):
        this_customer = Customer(loc, waittime)
        self.demand_list.append(this_customer)
        self.cust_grid[loc] = waittime

    def update_demand(self):
        #Decrement and remove demand if timer is up
        self.cust_grid = np.maximum(0, self.cust_grid - 1) 
        customers_to_delete = []
        for index in range(len(self.demand_list)):
            customer = self.demand_list[index]
            customer.timer = customer.timer - 1
            if(customer.timer <= 0):
                customers_to_delete.append(customer)
        self.removeCustomers(customers_to_delete)
        unfulilled_penalty = len(customers_to_delete) * self.unfulfilled_penalty
        
        if self.cust_popup_episode is None:
            self.add_new_cust()   
        else:
            self.add_new_cust_from_popup_episode(self.t+1)  
            
    def removeCustomers(self, customer_list):
        for cust in customer_list:
            self.demand_list.remove(cust)
    
    def getCustomerlocs(self):
        locs = []
        for customer in self.demand_list:
            locs.append(customer.location)
        return locs
    # ---------------------------------------------------------------- Helpers
    def isTerminal(self):
        if self.system_reward >= self.terminal_reward:
            return True
        else:
            return False
        
    def print_grid(self):
        print " "  + "------" * (self.grid_size - 1)
        state_to_print = np.zeros((self.grid_size, self.grid_size), dtype='<U2')
        state_to_print[state_to_print == ''] = '  '
        # customers
        state_to_print[self.cust_grid > 0] = 'De'
        # obstacles
        state_to_print[self.obstacle_grid > 0] = '@@'
        # cars
        for i in range(len(self.cars_list)):
            if state_to_print[self.cars_list[i].location] == 'De':
                state_to_print[self.cars_list[i].location] = "CD"
            else:
                state_to_print[self.cars_list[i].location] = self.cars_list[i]
        print '| ' + '\n| '.join([' . '.join(i) + ' |' for i in state_to_print])       
        print " "  + "------" * (self.grid_size - 1)

    def get_state_view(self, carID, cars_grid, cust_grid):
        state = np.zeros((4, self.grid_size, self.grid_size))
        cars_grid_self = np.zeros((self.grid_size, self.grid_size))
        cars_grid_self[cars_grid == carID] = 1
        cars_grid_others = np.zeros((self.grid_size, self.grid_size))
        cars_grid_others[(cars_grid != carID) & (cars_grid != 0)] = 1
        state[2, :, :] = cars_grid_self
        state[0, :, :] = cars_grid_others
        state[1, :, :] = cust_grid.copy()
        state[3, :, :] = self.obstacle_grid
        return state    
    
    def loadObstacleGrid(self,obstacle_grid):
        if obstacle_grid is not None:
            grid = np.zeros((self.grid_size, self.grid_size))
            positions = np.where(obstacle_grid > 0)
            for i in range(len(positions[0])):
                loc = (positions[0][i], positions[1][i])
                grid[loc] = 1
            return grid
        
        else:
            return np.zeros((self.grid_size, self.grid_size))

      