from pyvqnet.qnn.quantumlayer import QuantumLayer
from pyvqnet.optim import *
from pyvqnet.nn.loss import *
from pyvqnet.nn.module import Module
import numpy as np
from pyqpanda import *
import pyqpanda as pq
from pyvqnet.tensor import QTensor as QTensor

def question1(n_qubits):
    lr = 0.03
    epoch = 1000
    qbit_num  = n_qubits
    param_num = qbit_num * 4

    def prepara_HE_ansetz(input,weights,qlist,clist,machine):
            x1 = input.squeeze()
            param1 = weights.squeeze()
            circult = pq.QCircuit()
            qn = len(qlist)
            circult.insert(pq.H(qlist))
            for i in range(qn):
                circult.insert(pq.RZ(qlist[i], param1[4 * i]))
                circult.insert(pq.RX(qlist[i], param1[4 * i + 1]))
                circult.insert(pq.RZ(qlist[i], param1[4 * i + 2]))
            for j in range(qn-1):
                ry_control = pq.RY(qlist[j + 1], param1[4 * j + 3]).control(qlist[j])
                circult.insert(ry_control)
            ry_last = pq.RY(qlist[0], param1[4 * qn - 1]).control(qlist[qn - 1])
            circult.insert(ry_last)

            prog = pq.QProg()
            prog.insert(circult)
            prob = machine.prob_run_dict(prog, qlist, -1)
            prob = list(prob.values())
            return prob

    class Model(Module):
        def __init__(self):
            super(Model, self).__init__()
            self.pqc = QuantumLayer(prepara_HE_ansetz,param_num,"cpu",qbit_num)
        def forward(self, x):
            x = self.pqc(x)
            return x

    # 随机产生待训练数据的函数
    def circle(samples:int,  rads =  np.sqrt(2/np.pi)) :
        data_x, data_y = [], []
        x = np.random.rand(2)
        y = [0] * (1 << qbit_num)
        for i in range(1, qbit_num + 1):
            y[1 << i - 1] = 1.0
        data_x.append(x)
        data_y.append(y)
        return np.array(data_x,dtype=np.float32), np.array(data_y,np.int64)

    model = Model()
    optimizer = Adam(model.parameters(),lr=lr, )
    Closs = CategoricalCrossEntropy()

    data, label = circle(1)
    for i in range(epoch):
        model.train()
        loss = 0
        optimizer.zero_grad()
        output = model(data)
        losss = Closs(label, output)
        losss.backward()
        optimizer._step()
        loss += losss.item()
    res = {}
    for i in range(1, qbit_num + 1):
        res[format(1 << i - 1, f'0{qbit_num}b')] = output.getdata()[0][1 << i - 1]
    return res

def question2(n_qubits):
    lr = 0.03
    epoch = 200
    qbit_num  = n_qubits
    param_num = qbit_num * 4

    def prepara_HE_ansetz(input,weights,qlist,clist,machine):
            x1 = input.squeeze()
            param1 = weights.squeeze()
            circult = pq.QCircuit()
            qn = len(qlist)
            circult.insert(pq.H(qlist))
            for i in range(qn):
                circult.insert(pq.RZ(qlist[i], param1[4 * i]))
                circult.insert(pq.RX(qlist[i], param1[4 * i + 1]))
                circult.insert(pq.RZ(qlist[i], param1[4 * i + 2]))
            for j in range(qn-1):
                ry_control = pq.RY(qlist[j + 1], param1[4 * j + 3]).control(qlist[j])
                circult.insert(ry_control)
            ry_last = pq.RY(qlist[0], param1[4 * qn - 1]).control(qlist[qn - 1])
            circult.insert(ry_last)

            prog = pq.QProg()
            prog.insert(circult)
            prob = machine.prob_run_dict(prog, qlist, -1)
            prob = list(prob.values())
            return prob

    class Model(Module):
        def __init__(self):
            super(Model, self).__init__()
            self.pqc = QuantumLayer(prepara_HE_ansetz,param_num,"cpu",qbit_num)
        def forward(self, x):
            x = self.pqc(x)
            return x

    # 随机产生待训练数据的函数
    def circle(samples:int,  rads =  np.sqrt(2/np.pi)) :
        data_x, data_y = [], []
        x = np.random.rand(2)
        y = [0] * (1 << qbit_num)
        for i in range(1, qbit_num + 1):
            y[1 << i - 1] = 1
        y = y.normalize()
        print(y)
        data_x.append(x)
        data_y.append(y)
        return np.array(data_x,dtype=np.float32), np.array(data_y,np.int64)

    model = Model()
    optimizer = Adam(model.parameters(),lr=lr, )
    Closs = CategoricalCrossEntropy()

    data, label = circle(1)
    for i in range(epoch):
        model.train()
        loss = 0
        optimizer.zero_grad()
        output = model(data)
        losss = Closs(label, output)
        losss.backward()
        optimizer._step()
        loss += losss.item()
    res = {}
    for i in range(1, qbit_num + 1):
        res[format(1 << i - 1, f'0{qbit_num}b')] = output.getdata()[0][1 << i - 1]
    return res

if __name__ == "__main__":
    res1 = question1(4)
    print(res1)