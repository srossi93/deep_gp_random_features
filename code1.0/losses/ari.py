from sklearn.metrics import adjusted_rand_score
import numpy as np
import loss

class ARI(loss.Loss):
    def __init__(self, dout):
        loss.Loss.__init__(self,dout)

    def eval(self, ytrue, ypred):
        score = adjusted_rand_score(np.argmax(ytrue, 1), np.argmax(ypred, 1))
        return score

    def get_name(self):
        return "ARI"
