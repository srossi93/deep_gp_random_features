from sklearn.metrics import normalized_mutual_info_score
import numpy as np
import loss

class NMI(loss.Loss):
    def __init__(self, dout):
        loss.Loss.__init__(self,dout)

    def eval(self, ytrue, ypred):
        score = normalized_mutual_info_score(np.argmax(ytrue, 1), np.argmax(ypred, 1))
        return score

    def get_name(self):
        return "NMI"
