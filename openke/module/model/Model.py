from ..BaseModule import BaseModule


class Model(BaseModule):
    def __init__(self, ent_tot, rel_tot):
        super(Model, self).__init__()
        self.ent_tot = ent_tot
        self.rel_tot = rel_tot

    def forward(self, data):
        raise NotImplementedError

    def predict(self, data):
        raise NotImplementedError
