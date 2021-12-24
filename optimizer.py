class SteepestDescentOptimizer:
    def __init__(self, stepsize):
        self.stepsize = stepsize  # Constant stepsize.

    def update(self, x, grad):
        return x - self.stepsize * grad
