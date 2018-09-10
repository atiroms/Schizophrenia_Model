import numpy as np

import chainer
from chainer import variable
from chainer import links as L
from chainer import functions as F

from chainerrl import recurrent
from chainerrl import policy
from chainerrl import v_function

import a3c09 as a3c


class EltFilter(chainer.Link):
    def __init__(self, width, height, channels, batchSize = 1, wscale=1, bias=0, nobias=False,
                initialW=None, initial_bias=None):
        W_shape = (batchSize, channels, height, width)
        super(EltFilter, self).__init__(W=W_shape)

        if initialW is not None:
            self.W.data[...] = initialW
        else:
            std = wscale * np.sqrt(1. / (width * height * channels))
            self.W.data[...] = np.random.normal(0, std, W_shape)

        if nobias:
            self.b = None
        else:
            self.add_param('b', W_shape)
            if initial_bias is None:
                initial_bias = bias
            self.b.data[...] = initial_bias

    def __call__(self, x):
        y = x * self.W
        if self.b is not None:
            y = y + self.b
        return y

class ConvLSTM(chainer.Chain):
    def __init__(self, width, height, in_channels, out_channels, batchSize = 1):
        self.state_size = (batchSize, out_channels, height, width)
        self.in_channels = in_channels
        super(ConvLSTM, self).__init__(
            h_i=L.Convolution2D(out_channels, out_channels, 3, pad=1),
            c_i=EltFilter(width, height, out_channels, nobias=True),

            h_f=L.Convolution2D(out_channels, out_channels, 3, pad=1),
            c_f=EltFilter(width, height, out_channels, nobias=True),

            h_c=L.Convolution2D(out_channels, out_channels, 3, pad=1),

            h_o=L.Convolution2D(out_channels, out_channels, 3, pad=1),
            c_o=EltFilter(width, height, out_channels, nobias=True),
        )

        for nth in range(len(self.in_channels)):
            self.add_link('x_i' + str(nth), L.Convolution2D(self.in_channels[nth], out_channels, 3, pad=1, nobias=True))
            self.add_link('x_f' + str(nth), L.Convolution2D(self.in_channels[nth], out_channels, 3, pad=1, nobias=True))
            self.add_link('x_c' + str(nth), L.Convolution2D(self.in_channels[nth], out_channels, 3, pad=1, nobias=True))
            self.add_link('x_o' + str(nth), L.Convolution2D(self.in_channels[nth], out_channels, 3, pad=1, nobias=True))

        self.reset_state()

    def to_cpu(self):
        super(ConvLSTM, self).to_cpu()
        if self.c is not None:
            self.c.to_cpu()
        if self.h is not None:
            self.h.to_cpu()

    def to_gpu(self, device=None):
        super(ConvLSTM, self).to_gpu(device)
        if self.c is not None:
            self.c.to_gpu(device)
        if self.h is not None:
            self.h.to_gpu(device)

    def reset_state(self):
        self.c = self.h = None

    def unchain_backward_all(self):
        self.h.unchain_backward()

    def __call__(self, x):
        if self.h is None:
            self.h = variable.Variable(
                self.xp.zeros(self.state_size, dtype=x[0].data.dtype),
                volatile='auto')
        if self.c is None:
            self.c = variable.Variable(
                self.xp.zeros(self.state_size, dtype=x[0].data.dtype),
                volatile='auto')


        ii = self.x_i0(x[0])
        for nth in range(1, len(self.in_channels)):
            ii += getattr(self, 'x_i' + str(nth))(x[nth])
        ii += self.h_i(self.h)
        ii += self.c_i(self.c)
        ii = F.sigmoid(ii)

        ff = self.x_f0(x[0])
        for nth in range(1, len(self.in_channels)):
           ff += getattr(self, 'x_f' + str(nth))(x[nth])
        ff += self.h_f(self.h)
        ff += self.c_f(self.c)
        ff = F.sigmoid(ff)

        cc = self.x_c0(x[0])
        for nth in range(1, len(self.in_channels)):
           cc += getattr(self, 'x_c' + str(nth))(x[nth])
        cc += self.h_c(self.h)
        cc = F.tanh(cc)
        cc *= ii
        cc += (ff * self.c)

        oo = self.x_o0(x[0])
        for nth in range(1, len(self.in_channels)):
           oo += getattr(self, 'x_o' + str(nth))(x[nth])
        oo += self.h_o(self.h)
        oo += self.c_o(self.c)
        oo = F.sigmoid(oo)
        y = oo * F.tanh(cc)

        self.c = cc
        self.h = y
        return y

class SiS(chainer.Chain, recurrent.RecurrentChainMixin):
    def __init__(self, n_actions, input_action, skip_P, n_input_channels = 4, activation = F.relu, bias = 0.1, batchSize = 1):
        super(SiS, self).__init__()

        self.batchSize = batchSize
        self.n_actions = n_actions
        self.activation = activation
        self.input_action=input_action
        self.skip_P=skip_P

        self.add_link('ConvLSTM1', ConvLSTM(84, 84, (8, 32), 4))
        self.add_link('ConvLSTM2', ConvLSTM(21, 21, (64, 64), 32))
        self.add_link('ConvLSTM3', ConvLSTM(7, 7, (128, 64), 64))
        if self.input_action:
            self.add_link('ConvLSTM4', ConvLSTM(7, 7, (128, 128), 64))
        else:
            self.add_link('ConvLSTM4', ConvLSTM(7, 7, (128,), 64))

        self.add_link('ConvA2', L.Convolution2D(8, 32, 8, stride=4, pad=2))
        self.add_link('ConvA3', L.Convolution2D(64, 64, 5, stride=3, pad=2))
        self.add_link('ConvA4', L.Convolution2D(128, 64, 3, pad=1))

        self.add_link('ConvP1', L.Convolution2D(4, 4, 3, pad=1))
        if not self.skip_P:
            self.add_link('ConvP2', L.Convolution2D(32, 32, 3, pad=1))
            self.add_link('ConvP3', L.Convolution2D(64, 64, 3, pad=1))
            self.add_link('ConvP4', L.Convolution2D(64, 64, 3, pad=1))

        if self.input_action:
            self.add_link('FCI5', L.Linear(512, 6272))
            self.add_link('FCI6', L.Linear(self.n_actions, 512))

        self.add_link('FCO5', L.Linear(6272, 512, bias = bias))

        self.pi = policy.FCSoftmaxPolicy(512, n_actions)
        self.v = v_function.FCVFunction(512)

        self.reset_state()

    def to_cpu(self):
        super(SiS, self).to_cpu()
        for nth in range(1, 5):
            if getattr(self, 'E' + str(nth)) is not None:
                getattr(self, 'E' + str(nth)).to_cpu()
            if getattr(self, 'ConvLSTM' + str(nth)) is not None:
                getattr(self, 'ConvLSTM' + str(nth)).to_cpu()

    def to_gpu(self, device=None):
        super(SiS, self).to_gpu(device)
        for nth in range(1, 5):
            if getattr(self, 'E' + str(nth)) is not None:
                getattr(self, 'E' + str(nth)).to_gpu(device)
            if getattr(self, 'ConvLSTM' + str(nth)) is not None:
                getattr(self, 'ConvLSTM' + str(nth)).to_gpu(device)

    def reset_state(self):
        for nth in range(1, 5):
            setattr(self, 'E' + str(nth), None)
            getattr(self, 'ConvLSTM' + str(nth)).reset_state

    def unchain_backward_all(self):
        for nth in range(1, 5):
            getattr(self, 'ConvLSTM' + str(nth)).unchain_backward_all()

    def __call__(self, x, action_array = None):

        if self.E1 is None:
            self.E1 = variable.Variable(self.xp.zeros((self.batchSize, 8, 84, 84), dtype=self.xp.float32), volatile = 'auto')
        if self.E2 is None:
            self.E2 = variable.Variable(self.xp.zeros((self.batchSize, 64, 21, 21), dtype=self.xp.float32), volatile = 'auto')
        if self.E3 is None:
            self.E3 = variable.Variable(self.xp.zeros((self.batchSize, 128, 7, 7), dtype=self.xp.float32), volatile = 'auto')
        if self.E4 is None:
            self.E4 = variable.Variable(self.xp.zeros((self.batchSize, 128, 7, 7), dtype=self.xp.float32), volatile = 'auto')

        if self.input_action:
            I6 = self.FCI6(action_array)
            I5 = F.reshape(self.FCI5(I6), (self.batchSize, 128, 7, 7))
            R4 = self.ConvLSTM4((self.E4, I5))
        else:
            R4 = self.ConvLSTM4((self.E4,))

        R3 = self.ConvLSTM3((self.E3, R4))
        upR3 = F.unpooling_2d(R3, ksize = 3, stride = 3, cover_all=False)
        R2 = self.ConvLSTM2((self.E2, upR3))
        upR2 = F.unpooling_2d(R2, ksize = 4, stride = 4, cover_all=False)
        R1 = self.ConvLSTM1((self.E1, upR2))
        P1 = F.clipped_relu(self.ConvP1(R1),1.0)
        if self.skip_P:
            self.E1 = F.concat((self.activation(x - P1), self.activation(P1 - x)))
            A2 = self.activation(self.ConvA2(self.E1))
            self.E2 = F.concat((self.activation(A2 - R2), self.activation(R2 - A2)))
            A3 = self.activation(self.ConvA3(self.E2))
            self.E3 = F.concat((self.activation(A3 - R3), self.activation(R3 - A3)))
            A4 = self.activation(self.ConvA4(self.E3))
            self.E4 = F.concat((self.activation(A4 - R4), self.activation(R4 - A4)))
        else:
            self.E1 = F.concat((self.activation(x - P1), self.activation(P1 - x)))
            A2 = self.activation(self.ConvA2(self.E1))
            P2 = self.activation(self.ConvP2(R2))
            self.E2 = F.concat((self.activation(A2 - P2), self.activation(P2 - A2)))
            A3 = self.activation(self.ConvA3(self.E2))
            P3 = self.activation(self.ConvP3(R3))
            self.E3 = F.concat((self.activation(A3 - P3), self.activation(P3 - A3)))
            A4 = self.activation(self.ConvA4(self.E3))
            P4 = self.activation(self.ConvP4(R4))
            self.E4 = F.concat((self.activation(A4 - P4), self.activation(P4 - A4)))
        O5 = self.activation(self.FCO5(F.reshape(self.E4, (1, 6272))))

        return self.pi(O5), self.v(O5)
