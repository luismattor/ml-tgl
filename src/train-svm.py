from sklearn.svm import SVC, NuSVC, LinearSVC

from context import Context
from trainer import Trainer

if __name__ == "__main__":
    trainer = Trainer(Context(), SVC())
    trainer.train_and_test()
