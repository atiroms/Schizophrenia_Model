import numpy
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import variable
from PredNet-master.net import EltFilter
from PredNet-master.net import ConvLSTM

class SiS(chainer.Chain)
    def __init__(self, n_actions, batchSize = 1):
        super(SiS, self).__init__()

		self.batchSize = batchSize
        self.n_actions = n_actions

        self.add_link('ConvLSTM1', ConvLSTM(84, 84, (8, 32), 4))
        self.add_link('ConvLSTM2', ConvLSTM(21, 21, (64, 64), 32))
        self.add_link('ConvLSTM3', ConvLSTM(7, 7, (128, 64), 64))
        self.add_link('ConvLSTM4', ConvLSTM(7, 7, (128, 128), 64))

        self.add_link('ConvP1', L.Convolution2D(4, 4, 3, pad=1))
        self.add_link('ConvP2', L.Convolution2D(32, 32, 3, pad=1))
        self.add_link('ConvP3', L.Convolution2D(64, 64, 3, pad=1))
        self.add_link('ConvP4', L.Convolution2D(64, 64, 3, pad=1))

        self.add_link('FCI5', L.Linear(512, 6272))
        self.add_link('FCI6', L.Linear(self.n_actions, 512))

        self.add_link('FCO5', L.Linear(6272, 512))
        self.add_link('FCO6', L.Linear(512, self.n_actions))

        self.reset_state()

    def to_cpu(self):
        super(SiS, self).to_cpu()
        for nth in range(1, 5):
            if getattr(self, 'E' + str(nth)) is not None:
                getattr(self, 'E' + str(nth)).to_cpu()

    def to_gpu(self, device=None):
        super(SiS, self).to_gpu(device)
        for nth in range(1, 5):
            if getattr(self, 'E' + str(nth)) is not None:
                getattr(self, 'E' + str(nth)).to_gpu(device)

    def reset_state(self):
        for nth in range(1, 5):
            setattr(self, 'E' + str(nth)) = None
            getattr(self, 'ConvLSTM' + str(nth)).reset_state

    def __call__(self, x, activation, action = None):
		if self.E1 is None:
			self.E1 = variable.Variable(self.xp.zeros((self.batchSize, 8, 84, 84), dtype=x.data.dtype), volatile='auto'))
		if self.E2 is None:
			self.E2 = variable.Variable(self.xp.zeros((self.batchSize, 64, 21, 21), dtype=x.data.dtype), volatile='auto'))
		if self.E3 is None:
			self.E3 = variable.Variable(self.xp.zeros((self.batchSize, 128, 7, 7), dtype=x.data.dtype), volatile='auto'))
		if self.E4 is None:
			self.E4 = variable.Variable(self.xp.zeros((self.batchSize, 128, 7, 7), dtype=x.data.dtype), volatile='auto'))

        if action is None:
            R4 = self.ConvLSTM4((E4,))
        else:
            I6 = self.FCI6(action)
		    I5 = variable.Variable(self.xp.reshape(self.FCI5(I6),(128,7,7)))
		    R4 = self.ConvLSTM4((E4, I5))
		P4 = F.relu(self.ConvP4(R4))
		R3 = self.ConvLSTM3((E3, R4))
		P3 = F.relu(self.ConvP3(R3))
		upR3 = F.unpooling_2d(R3, ksize = 3, stride = 3, cover_all=False)
		R2 = self.ConvLSTM2((E2, upR3))
		P2 = F.relu(self.ConvP2(R2))
		upR2 = F.unpooling_2d(R3, ksize = 2, stride = 2, cover_all=False)
		R1 = self.ConvLSTM1((E1, upR2))
		P1 = F.clipped_relu(self.ConvP1(R1),1.0)
		self.E1 = F.concat((F.relu(x - P1), F.relu(P1 - x)))
		A2 = F.relu(self.ConvA2(E1))
		self.E2 = F.concat((F.relu(A2 - P2), F.relu(P2 - A2)))
		A3 = F.relu(self.ConvA3(E2))
		self.E3 = F.concat((F.relu(A3 - P3), F.relu(P3 - A3)))
		A4 = F.relu(self.ConvA4(E3))
		self.E4 = F.concat((F.relu(A4 - P4), F.relu(P4 - A4)))
		O5 = variable.Variable(self.xp.reshape(self.FCO5(E4), 6272))
		O6 = self.FCO6(O5)
		return O6
