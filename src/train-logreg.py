from sklearn.linear_model import LogisticRegression

from context import Context
from trainer import Trainer

if __name__ == "__main__":
    trainer = Trainer(Context(), LogisticRegression())
    trainer.train_and_test()
