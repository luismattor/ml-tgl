from sklearn.linear_model import Perceptron

from context import Context
from trainer import Trainer

if __name__ == "__main__":
    #percep = Perceptron()
    percep = Perceptron(n_iter=100, verbose=0, random_state=None, fit_intercept=True, eta0=0.002)
    trainer = Trainer(Context(), percep)
    trainer.train_and_test()
