import numpy as np
from utils import sign
from sklearn.base import BaseEstimator
from typing import List
from sklearn.metrics import accuracy_score

class ActiveLearner():
    def __init__(self, estimator: BaseEstimator, X_init: np.array = None, Y_init: np.array = None, bootstrap_init: bool = False, **fit_kwargs) -> None:
        self.estimator = estimator
        self.X = None
        self.Y = None
        if X_init is not None:
            self.X = X_init.copy()
        if Y_init is not None:
            self.Y = Y_init.copy()
        self.fit(bootstrap=bootstrap_init, **fit_kwargs)
        self.is_byzantine = False
        self.flip_proba = None
        self.rand_proba = None

    def get_classes_default_pred(self):
        if self.X_train is None or self.Y_train is None:
            self.n_classes = 0
            self.default_pred = 1.
        else:
            self.n_classes = len(np.unique(self.Y_train))
            self.default_pred = self.Y_train[0]

    def add_training_sample(self, x, y):
        if self.X is None or self.Y is None:
            self.X = x.reshape(1, -1).copy()
            self.Y = y.reshape(-1,).copy()
        else:
            self.X = np.concatenate([self.X, x])
            self.Y = np.concatenate([self.Y, y])
    
    def make_byzantine(self, mode: str = 'flip', p: float = 0.5):
        self.is_byzantine = True
        if mode.lower() == 'flip':
            self.flip_proba = p
        else:
            self.rand_proba = p

    def prepare_train_set(self, bootstrap: bool = False):
        if self.X is None or self.Y is None:
            self.X_train = None
            self.Y_train = None
        elif bootstrap:
            inds = np.random.choice(np.arange(len(self.X)), size=len(self.X), replace=True)
            self.X_train = self.X[inds]
            self.Y_train = self.Y[inds]
        else:
            self.X_train = self.X
            self.Y_train = self.Y
    
    def fit(self, bootstrap: bool = False, **fit_kwargs):
        self.prepare_train_set(bootstrap=bootstrap)
        self.get_classes_default_pred()
        if self.n_classes > 1:
            self.estimator.fit(self.X_train, self.Y_train, **fit_kwargs)
    
    def teach(self, x, y, bootstrap: bool = False, **fit_kwargs):
        self.add_training_sample(x, y)
        self.fit(bootstrap=bootstrap, **fit_kwargs)
    
    def predict(self, X):
        if self.n_classes > 1:
            preds = self.estimator.predict(X)
        else:
            preds = np.ones(len(X)) * self.default_pred

        if self.is_byzantine and self.flip_proba is not None:
            flip_signs = sign(np.random.rand(len(preds)) - self.flip_proba)
            preds *= flip_signs
        elif self.is_byzantine and self.rand_proba is not None:
            rand_inds = np.random.rand(len(preds)) <= self.rand_proba
            rand_preds = sign(np.random.rand(rand_inds.astype(int).sum()) - 0.5)
            preds[rand_inds] = rand_preds
        return preds
    
    def predict_proba(self, X):
        if self.n_classes > 1:
            probas = self.estimator.predict_proba(X)[:, 1]
        else:
            if self.default_pred == 1.:
                probas = np.column_stack([
                    np.zeros(len(X)),
                    np.ones(len(X))
                ])
            else:
                probas = np.column_stack([
                    np.ones(len(X)),
                    np.zeros(len(X))
                ])

        if self.is_byzantine and self.flip_proba is not None:
            flip_inds = np.random.rand(len(probas)) <= self.flip_proba
            probas[flip_inds] = 1 - probas[flip_inds]
        elif self.is_byzantine and self.rand_proba is not None:
            rand_inds = np.random.rand(len(probas)) <= self.rand_proba
            rand_preds = np.random.rand(rand_inds.astype(int).sum())
            probas[rand_inds] = rand_preds
        return probas

class Committee():
    def __init__(self, learners: List[ActiveLearner], strategy: str = 'vote') -> None:
        self.learners = learners
        self.good_inds = []
        for i, learner in enumerate(self.learners):
            if not learner.is_byzantine:
                self.good_inds.append(i)
        self.strategy = strategy
    
    def teach(self, x, y, bag: bool = False, **fit_kwargs):
        for learner in self.learners:
            learner.teach(x, y, bootstrap=bag, **fit_kwargs)
    
    def rebag(self, **fit_kwargs):
        for learner in self.learners:
            learner.fit(bootstrap=True, **fit_kwargs)
            
    
    def query(self, X: np.array, random_tie_break: bool = False):
        if self.strategy == 'vote':
            learner_preds = np.column_stack([
                learner.predict(X) for learner in self.learners
            ])
            disagreement = np.abs(np.sum(learner_preds, axis=1))
        elif self.strategy == 'consensus':
            learner_probas = np.column_stack([
                learner.predict_proba(X) for learner in self.learners
            ])
            avg_probas = np.mean(learner_probas, axis=1)
            disagreement = np.maximum(avg_probas, 1. - avg_probas)
        else:
            disagreement = np.random.rand(len(X))
            
        if random_tie_break:
            disagreement_with_inds = np.column_stack([
                np.arange(len(disagreement)),
                disagreement 
            ])
            np.random.shuffle(disagreement_with_inds)
            return int(disagreement_with_inds[np.argmin(disagreement_with_inds[:, 1]), 0]), np.min(disagreement)
        else:
            return np.argmin(disagreement), np.min(disagreement)

    def score(self, X, Y, good_only: bool = False):
        if good_only:
            preds = sign(np.column_stack([
                self.learners[good_ind].predict(X) for good_ind in self.good_inds
            ]).sum(axis=1))
        else:
            preds = sign(np.column_stack([
                learner.predict(X) for learner in self.learners
            ]).sum(axis=1))
        return accuracy_score(Y, preds)
