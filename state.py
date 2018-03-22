import numpy as np
from sklearn.preprocessing import LabelBinarizer

class state():
    def __init__(self, kwargs={}):
        self.move_list = kwargs.get("move_list", None)
        assert self.move_list is not None
        self.move_list_numeric = kwargs.get("move_list_numeric", 
                                              [i for i in range(len(self.move_list))])
        self.size = kwargs.get("size", None)
        assert self.size is not None
        self.board = kwargs.get("board", np.zeros(self.size, dtype=int))
        self.lb = kwargs.get("lb", LabelBinarizer())
        self.lb.fit(self.move_list)
        
    def get_ohe_board(self):
        ohe_board = state.lb.transform(self.board)
        ohe_board = np.expand_dims(ohe_board, axis=0)
        return ohe_board
            
    def clone(self):
        kwargs = {}
        kwargs["move_list"] = self.move_list
        kwargs["move_list_numeric"] = self.move_list_numeric
        kwargs["size"] = self.size
        kwargs["board"] = self.board
        kwargs["lb"] = self.lb
        return state(kwargs)