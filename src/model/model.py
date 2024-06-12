from pyvqnet.nn.module import Module
from pyvqnet.qnn.quantumlayer import QuantumLayer

from model.quantum_line import qcl_circuits, qvc_circuits, single_line_weight, qvc_block

class Model(Module):
    def __init__(self, raw=4, col=4):
        super().__init__()
        self.qvc = QuantumLayer(qcl_circuits, single_line_weight * (raw * col), "cpu", raw * col)
        # self.qvc = QuantumLayer(qvc_circuits, qvc_block * (raw * col) * 3, "cpu", raw * col)

    def forward(self, x):
        x = self.qvc(x)
        return x