"""
You can create any other helper funtions.
Do not modify the given functions
"""
import copy
import queue
import sys

def A_star_Traversal(cost, heuristic, start_point, goals):
    """
    Perform A* Traversal and find the optimal path 
    Args:
        cost: cost matrix (list of floats/int)
        heuristic: heuristics for A* (list of floats/int)
        start_point: Staring node (int)
        goals: Goal states (list of ints)
    Returns:
        path: path to goal state obtained from A*(list of ints)
    """
    path = []
    # TODO      
    n = len(cost) # Number of nodes in the graph
    if n<1:
    	sys.exit("Invalid number of nodes")
    	
    visited = [0 for i in range(n)] # to keep check on nodes visited
    frontierq = queue.PriorityQueue()             

    frontierq.put((heuristic[start_point], ([start_point], start_point, 0))) # Insert(est_total_cost, (start_point, A*_path_till_node, total_cost_till_node)) 

    # Until priority queue is not empty
    while(frontierq.qsize() != 0):        
        total_estimated_cost, nodes_tuple = frontierq.get() # Pop the node info from the priority queue
        Astar_path_till_node = nodes_tuple[0]
        node = nodes_tuple[1]
        node_cost = nodes_tuple[2]

        if visited[node] == 0:
            visited[node] = 1

            if node in goals:
                return Astar_path_till_node # If node is a goal point, return A* Path till Goal Node

            for neigh_node in range(1, n): 
                if cost[node][neigh_node] > 0 and visited[neigh_node] == 0:               
                    total_cost_till_node = node_cost + cost[node][neigh_node] # Compute total cost till neighbour node
                    est_total_cost = total_cost_till_node + heuristic[neigh_node]                    
                    Astar_path_till_neigh_node = copy.deepcopy(Astar_path_till_node) 
                    Astar_path_till_neigh_node.append(neigh_node) # Add neighbour node to new A* path list
                    frontierq.put((est_total_cost, (Astar_path_till_neigh_node, neigh_node, total_cost_till_node)))

    # return empty path to goal nodes if not found
    return path


def DFS_Traversal(cost, start_point, goals):
    """
    Perform DFS Traversal and find the optimal path 
        cost: cost matrix (list of floats/int)
        start_point: Staring node (int)
        goals: Goal states (list of ints)
    Returns:
        path: path to goal state obtained from DFS(list of ints)
    """
    path = []
    # TODO
    n = len(cost) # No. of nodes in graph
    if n<1:
    	sys.exit("Invalid number of nodes")
    	
    visited = [0 for i in range(n)]  #visited tells whether a particular node has been visited or not
    frontier_stk = queue.LifoQueue() #DFS uses LIFO principle

    # Insert start_point into stack
    frontier_stk.put((start_point, [start_point]))

    while(frontier_stk.qsize() != 0): #Until stk not empty
        node,path = frontier_stk.get() #Removes and returns an item from the queue

        if visited[node] == 0:# If node was not visited
            visited[node] = 1

            if node in goals:
                return path

            # For every neighbouring connected node
            for neigh_node in range(n-1, 0, -1):
                if cost[node][neigh_node] > 0: 
                    if visited[neigh_node] == 0:
                        path_until_neigh_node = copy.deepcopy(path) #deepcopy stores copy of original obj 
                        path_until_neigh_node.append(neigh_node)
                        frontier_stk.put((neigh_node, path_until_neigh_node))         
            
    return path #if path to goal nodes is not found
         
    
