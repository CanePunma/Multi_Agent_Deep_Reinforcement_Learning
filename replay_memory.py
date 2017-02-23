import random

class ReplayMemory(object):
    def __init__(self, buffer, batchSize, gamma = None, TD = None):
        self.buffer = buffer
        self.batchSize = batchSize
        self.h = 0
        self.replay = []
        self.intermediateReplay = {}
        self.gamma = gamma
        self.TD = TD
        
    def addToMemory(self,memory,terminal):
        if memory['event'] == 'pickup':
            inter_terminal = True
        else:
            inter_terminal = False
        record = (memory['state'],memory['action'],memory['reward'],memory['new_state'],inter_terminal)
        carID = memory['carID']
        
        if self.TD == None:
            self.add(record)
        else:
            self.addToIntermediateMemory(record,carID,terminal)
            
    def add(self,record):
        if (len(self.replay) < self.buffer):
            self.replay.append(record)
        else:
            if (self.h < (self.buffer-1)):
                self.h += 1
            else:
                self.h = 0
            self.replay[self.h] = (record)
            
    def getMinibatch(self):
        return random.sample(self.replay, self.batchSize)
    
    def getSize(self):
        return len(self.replay)
    
    def isFull(self):
        if len(self.replay) >= self.buffer:
            return True
        else:
            return False
        
    def addToIntermediateMemory(self,record, carID,terminal):
        if carID not in self.intermediateReplay.keys():
            self.intermediateReplay[carID] = []
            self.intermediateReplay[carID].append(record)
        else:
            self.intermediateReplay[carID].append(record)
            
        if len(self.intermediateReplay[carID]) == self.TD or terminal:
            self.eligibilityTrace(self.intermediateReplay[carID])
            self.intermediateReplay[carID] = []
            
               
    def eligibilityTrace(self, sequence):
        res = []
        for i in range(len(sequence)):
            timestep = sequence[i]
            reward = timestep[2]
            eligibility_trace = reward
            power = 1
            for j in range(i+1,len(sequence)):
                other_reward = sequence[j][2]
                eligibility_trace = eligibility_trace + ((self.gamma**power) * other_reward)
                power = power + 1
                
                #If you want to do limited traces vs accumulating traces
                #if eligibility_trace > 1:
                #    eligibility_trace = 1

            record = (timestep[0], timestep[1], eligibility_trace, timestep[3],timestep[4])
            self.add(record)
            


                