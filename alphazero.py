import numpy as np
from copy import deepcopy
from tqdm import tqdm
from state import state as st
from random import shuffle
from IPython.core.debugger import set_trace
import database as db
from os.path import exists, join
from os import mkdir
import pickle as p

def dirichlet_noise(size, eta=None):
    if eta is None:
        eta = 8/size
    noise_vec = []

    for _ in range(size):
        noise_vec.append(eta)

    noise = np.random.dirichlet(noise_vec, 1)
    return noise[0]

def convert_to_string(arr):
    string_ver = ""
    for num in arr:
        string_ver += str(num) + "_"
    string_ver = string_ver[:-1]
    return string_ver

def create_node(parent, action, proba):
    #print("in create node")
    node = {}
    node["children"] = []
    node["visits"] = 0
    node["total_value"] = 0
    node["mean_value"] = 0
    node["probability"] = proba
    node["parent"] = parent
    node["action"] = action
    node["is_root"] = False
    node["depth"] = parent["depth"] + 1
    return node

def select(node, is_stochastic=False, most_visits=False, T=1, selected_moves=None):
    #print("in select")
    scores = []
    probabilities = []
    c = np.sqrt(2)
    for i, child in enumerate(node["children"]):
        other_visits = node["visits"] - child["visits"]
        U = c * child["probability"] * np.sqrt(other_visits)/(1 + child["visits"])
#         U = child["probability"]/(1 + child["visits"])
#        print("U: {}, child visits: {}, child proba: {}, child mean value: {}".format(U, child["visits"],
#                                                                                     child["probability"],
#                                                                                     child["mean_value"]))
        score = child["mean_value"] + U
        scores.append(score)
        proba = (child["visits"]**(1/T))/(other_visits**(1/T))
        probabilities.append(proba)
        
    probabilities = np.array(probabilities)
    if np.sum(probabilities) == 0:
        probabilities = np.zeros(len(probabilities))
        probabilities += 1/len(probabilities)

    if selected_moves is not None:
        while np.argmax(scores) in selected_moves:
            scores[np.argmax(scores)] = 0

        if len(selected_moves) > 0:
            for move in selected_moves:
                probabilities[move] = 0
            probas_sum = np.sum(probabilities)
            if probas_sum == 0:
                diff = len(probabilities) - len(selected_moves)
                for i in range(len(probabilities)):
                    if i not in selected_moves:
                        probabilities[i] = 1/diff 
                        
                probas_sum = np.sum(probabilities)
                
            for i, _ in enumerate(probabilities):
                probabilities[i] /= probas_sum
    
    if is_stochastic and most_visits:
        try:
            node = np.random.choice(node["children"], p = probabilities)
        except ValueError as e:
            probas_sum = np.sum(probabilities)
            for i, proba in enumerate(probabilities):
                probabilities[i] = proba/probas_sum
            node = np.random.choice(node["children"], p = probabilities)
    elif most_visits:
        set_trace()
        node = node["children"][np.argmax(probabilities)]
    else:
        node = node["children"][np.argmax(scores)]
        
    return node, probabilities
    
def _transition(state, node):
    #print("in transition")
    state.board[node["depth"]] = node["action"]
            
def expand(state, node, net, board_converter):
    #print("in expand")
    converted_board = board_converter(state.board)
    converted_board = np.array(converted_board)
    converted_board = np.expand_dims(converted_board, axis=0)
    (probabilities, value) = net.predict(converted_board)
    probabilities = probabilities[0]
    value = value[0][0]
    eta = .25
    if node["is_root"]:
        probabilities = probabilities * (1 - eta) + eta*dirichlet_noise(size=len(probabilities))
    for i, proba in enumerate(probabilities):
        action = state.move_list_numeric[i]
        n = create_node(node, action, proba)
        node["children"].append(n)
        
    return value
        
def backup(node, value, orig_depth):
    #print("in backup")
    
    node["visits"] += 1
    node["total_value"] += value
    node["mean_value"] = node["total_value"]/node["visits"]
    orig_depth_node = node
    while "parent" in node:
        if node["depth"] == orig_depth:
            orig_depth_node = node
        node = node["parent"]
        node["visits"] += 1
        node["total_value"] += value
        node["mean_value"] = node["total_value"]/node["visits"]
        
    return orig_depth_node

# consider switching to in memory sqlite (s, a) records

def MCTS(root_state=None, root_node=None, terminal_function=None, net = None, is_stochastic=False, num_simulations=1,
        size=None, board_converter=None, move_list=None, T=1, custom_select_rule=None, selected_moves=None, batch=None,
        game=None):
    assert terminal_function is not None
    assert net is not None
    assert board_converter is not None
    assert size is not None
    assert move_list is not None
    #if num_simulations < 2:
    #    raise Exception("num_simulations must be at least 2 for the algorithm to work properly")
    
    if not exists("data"):
        mkdir("data")
        
    if batch is None:
        batch = []
        
    if game is None:
        game = []
    
    if root_state is None:
        root_state = st({"size": size, "move_list": move_list})
    
    if root_node is None:
        root_node = {"children": [], "depth": -1, "mean_value": 0, "total_value": 0, "visits": 0}
        
    node = root_node
    
    node["is_root"] = True
    
    if node["depth"] == -1:
        state = root_state.clone()
        value = expand(state, node, net, board_converter)
        node = backup(node, value, root_node["depth"])

    for i in range(num_simulations):  
        state = root_state.clone()

        while node["children"] != []:
            node, probas = select(node, selected_moves=selected_moves)
            _transition(state, node)
            
        if node["depth"] < state.size-1:
            value = expand(state, node, net, board_converter)
            node = backup(node, value, root_node["depth"])
        else:
            result = terminal_function(state, node)
            node = backup(node, result, root_node["depth"])
            
    node, probas = select(node, most_visits=True, is_stochastic=is_stochastic, T=T, selected_moves=selected_moves)
    _transition(state, node)
    game.append((state.board, probas))
    
    result = terminal_function(state, node)
    if result is not None:
        temp = []
        
        for (board, probas) in game:
            temp.append((board, probas, result))
            
        game = temp
        batch.append(game)
        node = backup(node, result, -1)
        state = st({"size": size, "move_list": move_list})
        
    return state, node, game, batch

def self_play(net, terminal_function, board_converter, batch_size=1, num_training_loops=1, num_simulations=1, is_stochastic=True, early_stop_func=None, augment_data_func=None, size=None, move_list=None, duplicate_moves=True):
    for _ in tqdm(range(num_training_loops)):
        state, node = None, None
        T = 1
        batch = []
        for _ in tqdm(range(batch_size)):
            game = []
            i = 1
            if not duplicate_moves:
                selected_moves = []
            else:
                selected_moves = None
            while len(batch) != batch_size:
                print("On move {} of {}".format(i, size))
                i += 1
                state, node, game, batch  = MCTS(state, node, terminal_function, net=net, board_converter=board_converter, num_simulations=num_simulations, is_stochastic=is_stochastic, size=size, move_list=move_list, T=T, selected_moves=selected_moves, game=game, batch=batch)
                if not duplicate_moves:
                    selected_moves.append(node["action"])
            T -= 1/batch_size
        train_net(net, batch, board_converter)
        
    print("Saving net")
    net.save_model("best_net.h5")
        
    return net
       
def train_net(net, batch, board_converter):
    boards = []
    probas = []
    results = []
    
    shuffle(batch)
    for game in batch:
        shuffle(game)
        for (board, proba, result) in game:
            boards.append(board_converter(board))
            probas.append(proba)
            results.append(result)
            
    boards = np.array(boards)
    probas = np.array(probas)
    results = np.array(results)
        
    net.fit(x=boards, y=[probas, results])

def get_optimal_play(num_simulations=100):
    state, node = None, None
    is_terminal = False
    while not is_terminal:
        state, node, is_terminal = MCTS(state, node, terminal_function, net=net, num_simulations=num_simulations, 
                                        is_stochastic=False, early_stop_func=None)
        
    return state.board
        