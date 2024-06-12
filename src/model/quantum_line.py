import pyqpanda as pq
from pyqpanda import *
from pyvqnet.qnn.template import AmplitudeEmbeddingCircuit

single_line_weight = 6  # >= 3

def qcl_circuits(input, weights, qlist, clist, machine):
    assert single_line_weight >= 3, "single_line_weight must be >= 3"
    def build_circuit(weights, qubits):
        cir = pq.QCircuit()
        for i in range(len(qubits)):
            cir.insert(pq.RZ(qubits[i], weights[i][0]))
            cir.insert(pq.RY(qubits[i], weights[i][1]))
            cir.insert(pq.RZ(qubits[i], weights[i][2]))
        for d in range(3, single_line_weight):
            for i in range(len(qubits) - 1):
                cir.insert(pq.CNOT(qubits[i], qubits[i + 1]))
            cir.insert(pq.CNOT(qubits[len(qubits) - 1], qubits[0]))

            for i in range(len(qubits)):
                cir.insert(pq.RY(qubits[i], weights[i][d]))
        return cir
    
    weights = weights.reshape([len(qlist), single_line_weight])
    cir = pq.QCircuit()
    cir.insert(AmplitudeEmbeddingCircuit(input, qlist))
    subcir = build_circuit(weights, qlist)
    cir.insert(subcir)
    prog = pq.QProg()
    prog.insert(cir)
    res = machine.prob_run_dict(prog, qlist[0], -1)
    res = list(res.values())
    return res

qvc_block = 2   # >= 1
def qvc_circuits(input, weights, qlist, clist, machine):
    assert qvc_block >= 1, "qvc_block must be >= 1"
    def build_circuit(weights, qubits):
        cir = pq.QCircuit()
        for w in range(weights.shape[0]):
            for i in range(len(qubits)):
                cir.insert(pq.RZ(qubits[i], weights[w][i][0]))
                cir.insert(pq.RY(qubits[i], weights[w][i][1]))
                cir.insert(pq.RZ(qubits[i], weights[w][i][2]))
            for i in range(len(qubits) - 1):
                cir.insert(pq.CNOT(qubits[i], qubits[i + 1]))
            cir.insert(pq.CNOT(qubits[len(qubits) - 1], qubits[0]))
        return cir
    weights = weights.reshape([qvc_block, len(qlist), 3])
    cir = pq.QCircuit()
    cir.insert(AmplitudeEmbeddingCircuit(input, qlist))
    prog = build_circuit(weights, qlist)
    cir.insert(prog)
    prob = machine.prob_run_dict(cir, qlist[0], -1)
    prob = list(prob.values())
    return prob