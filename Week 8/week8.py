import numpy as np


class HMM:
    """
    HMM model class
    Args:
        A: State transition matrix
        states: list of states
        emissions: list of observations
        B: Emmision probabilites
    """

    def __init__(self, A, states, emissions, pi, B):
        self.A = A
        self.B = B
        self.states = states
        self.emissions = emissions
        self.pi = pi
        self.N = len(states)
        self.M = len(emissions)
        self.make_states_dict()

    def make_states_dict(self):
        """
        Make dictionary mapping between states and indexes
        """
        self.states_dict = dict(zip(self.states, list(range(self.N))))
        self.emissions_dict = dict(
            zip(self.emissions, list(range(self.M))))

    def viterbi_algorithm(self, seq):
        """
        Function implementing the Viterbi algorithm
        Args:
            seq: Observation sequence (list of observations. must be in the emmissions dict)
        Returns:
            nu: Porbability of the hidden state at time t given an obeservation sequence
            hidden_states_sequence: Most likely state sequence 
        """
        # TODO
        
        L = len(seq)      
        nu = np.zeros((L, self.N))
        temp = np.zeros((L, self.N), dtype=int)
        
        
        #INITIALIZATION 
        for i in range(0,self.N):
            temp[0, i] = 0
            nu[0, i] = self.B[i, self.emissions_dict[seq[0]]]  * self.pi[i]    
                     
            
        #RECURSION
        for i in range(1, L):
            for j in range(0, self.N):
                nuMax = -1
                tempMax = -1
                for k in range(0, self.N):
                    localNu = nu[i - 1, k] * self.A[k, j] * \
                        self.B[j, self.emissions_dict[seq[i]]]
                    if localNu > nuMax:
                        nuMax = localNu
                        tempMax = k
                nu[i, j] = nuMax
                temp[i, j] = tempMax
                
                
        #TERMINATION
        nuMax = -1
        tempMax = -1
        for i in range(0, self.N):
            localNu = nu[L - 1, i]
            if localNu > nuMax:
                nuMax = localNu
                tempMax = i
                
                
        #BACKTRACKING
        states = [tempMax]
        
        for i in range(L - 1, 0, -1):
            states.append(temp[i, states[-1]]) # reverse order of states
            
        states.reverse()        
        self.states_dict = {v: k for k, v in self.states_dict.items()}
        
        return [self.states_dict[i] for i in states]       
        pass
