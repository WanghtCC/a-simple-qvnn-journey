from pyqpanda import *
import pyvqnet
import pyqpanda as pq
import numpy as np
from pyvqnet.qnn.template import AmplitudeEmbeddingCircuit

def run_for_result(machine, prog, c, num):
    print(prog)
    results = machine.run_with_configuration(prog, c, num)
    for result in results.items():
        print('{\'' + result[0] + '\':', str(result[1]) + '}')

def run_with_prob(machine, prog, q):
    print(prog)
    result = machine.prob_run_dict(prog, q)
    print(result)
    return result

def easy_test(machine, num_q = 3, num_c = 3):
    # 创建一个量子程序
    prog = pq.QProg()
    # 创建两个量子比特
    q = machine.qAlloc_many(num_q)
    # 创建两个经典比特
    c = machine.cAlloc_many(num_c)

    # 量子程序开始
    prog << pq.H(q) \
        << CNOT(q[0], q[1]) \
        << pq.RZ(q[1], np.pi/3) \
        << pq.CNOT(q[0], q[1]) \
        << CNOT(q[1], q[2]) \
        << pq.RZ(q[2], np.pi/3) \
        << pq.CNOT(q[1], q[2]) \
        << CNOT(q[2], q[0]) \
        << RZ(q[0], np.pi/3) \
        << CNOT(q[2], q[0]) \
        << RX(q, np.pi/3) \
        # << pq.meas_all(q, c)

    # run_for_result(machine, prog, c, 1000)
    res = run_with_prob(machine, prog, q)
    res_np = np.array(list(res.values()))
    print(res_np)
    print(res_np.conj())


    # print(res.values())

def oracle_test(machine, num_q, num_c):
    # 创建一个量子程序
    prog = pq.QProg()
    # 创建两个量子比特
    q = machine.qAlloc_many(num_q)
    c = machine.cAlloc_many(num_c)

    # 量子程序开始
    prog << pq.H(q[0]) \
        << pq.CNOT(q[0], q[1])
    mat = get_matrix(prog, True)
    prog1 = QProg()
    prog1 << QOracle(q, mat)

    run_with_prob(machine, prog1, q)

def ghz_test(machine, num_q, num_c):
    # 创建一个量子程序
    prog = pq.QProg()
    # 创建三个量子比特
    q = machine.qAlloc_many(num_q)
    c = machine.cAlloc_many(num_c)

    # 量子程序开始 
    prog << pq.H(q[0]) \
        << pq.X(q[1]) \
        << pq.X(q[2]) \
        << pq.CNOT(q[0], q[1]) \
        << pq.CNOT(q[0], q[2]) \
        << pq.CNOT(q[0], q[3])
    
    run_with_prob(machine, prog, q)
    # run_for_result(machine, prog, c, 1000)

def swap_test(machine, num_q, num_c):
    # 创建一个量子程序
    prog = pq.QProg()
    # 创建三个量子比特
    q = machine.qAlloc_many(num_q)
    c = machine.cAlloc_many(num_c)

    # 量子程序开始 
    prog << pq.H(q[0]) \
        << pq.X(q[1]) \
        << pq.iSWAP(q[0], q[1]) \
        << pq.CNOT(q[1], q[2]) \
        << pq.H(q[3]) \
        << meas_all(q, c)
    
    run_for_result(machine, prog, c, 1000)

def qif_test(machine, num_q = 3, num_c = 3):
    # 创建一个量子程序
    prog = pq.QProg()
    branch_true = pq.QProg()
    branch_false = pq.QProg()
    # 创建三个量子比特
    q = machine.qAlloc_many(num_q)
    c = machine.cAlloc_many(num_c)
    c[0].set_val(0)
    c[1].set_val(3)

    # 量子程序开始
    branch_true << pq.H(q[0]) << pq.H(q[1]) << pq.H(q[2])
    branch_false << pq.H(q[0]) << pq.CNOT(q[0], q[1]) << H(q[2]) << pq.CNOT(q[1], q[2])

    prog << pq.QIfProg(c[0] > c[1], branch_true, branch_false)
    run_with_prob(machine, prog, q)

def qwhile_test(machine, num_q = 3, num_c = 3):
    # 创建一个量子程序
    prog = pq.QProg()
    prog_while = pq.QProg()
    # 创建三个量子比特
    q = machine.qAlloc_many(num_q)
    c = machine.cAlloc_many(num_c)
    c[0].set_val(0)
    c[1].set_val(1)

    # 量子程序开始
    prog_while << H(q[0]) << H(q[1]) << H(q[2])\
        << assign(c[0], c[0] + 1) << Measure(q[1], c[1])
    

    prog << QWhileProg(c[1], prog_while)
    run_with_prob(machine, prog, q)
    # run_for_result(machine, prog, c, 1000)

def qW_test(machine, num_q = 3, num_c = 3):
    # 创建一个量子程序
    prog = pq.QProg()
    # 创建三个量子比特
    q = machine.qAlloc_many(num_q)
    c = machine.cAlloc_many(num_c)

    # 量子程序开始
    prog << H(q[0]) \
        << CNOT(q[0], q[1]) \
        << CNOT(q[0], q[2]) \
        << CNOT(q[1], q[2]) \
        << RX(q[0], np.pi/2)\
    << meas_all(q, c)

    run_with_prob(machine, prog, q)
    run_for_result(machine, prog, c, 1000)

def two_qubit_dj_algorithm(machine, num_q = 2, num_c = 2):
    fx0 = 0
    fx1 = 0
    oracle_function = [fx0, fx1]
    prog = QCircuit()
    qubit1 = QVec(num_q)
    qubit2 = machine.qAlloc_many(1)
    if oracle_function[0] == 0 and oracle_function[1] == 1:
        prog << CNOT(qubit1[0], qubit2)
    elif oracle_function[0] == 1 and oracle_function[1] == 0:
        prog << CNOT(qubit1[0], qubit2) \
            << X(qubit2)
    elif oracle_function[0] == 1 and oracle_function[1] == 1:
        prog << X(qubit2)
    
def pouli_test(machine, num_q = 2, num_c = 2):
    p1 = PauliOperator()
    p2 = PauliOperator({'z0 z1': 2, 'x2 y3': 3})
    p3 = PauliOperator('z4 z5', 2)
    p4 = PauliOperator(2)
    p5 = p2

    mul = p2 * p3
    index_map = {}
    remap_pauli = mul.remapQubitIndex(index_map)
    print(remap_pauli)
    print(remap_pauli.getMaxIndex())
    print(remap_pauli.get_max_index())

# def qaoa_test(machine, num_q = 4, num_c = 4):
#     # 创建一个量子程序
#     prog = pq.QProg()
#     # 创建四个量子比特
#     q = machine.qAlloc_many(num_q)
#     c = machine.cAlloc_many(num_c)

#     param_b = [0.0, 0.0, 0.0, 0.0]
#     param_r = [0.0, 0.0, 0.0, 0.0]

#     # 量子程序开始
#     prog << H(q) \
#         << CNOT(q[0], q[1]) << RZ(q[1], param_b[1]) << CNOT(q[0], q[1]) \
#         << CNOT(q[1], q[2]) << RZ(q[2], param_b[2]) << CNOT(q[1], q[2]) \
#         << CNOT(q[2], q[3]) << RZ(q[3], param_b[3]) << CNOT(q[2], q[3]) \
#         << CNOT(q[3], q[0]) << RZ(q[0], param_b[0]) << CNOT(q[3], q[0]) \
#         << RX(q[0], param_r[0]) \
#         << RX(q[1], param_r[1]) \
#         << RX(q[2], param_r[2]) \
#         << RX(q[3], param_r[3]) \
#         << meas_all(q, c)
    
#     optimizer = OptimizerFactory.makeOptimizer(OptimizerType.NELDER_MEAD)
#     optimizer.registerFunc(lossFunc, param_b, param_r)
#     optimizer.setXatol(1e-6)
#     optimizer.setFatol(1e-6)
#     optimizer.setMaxFCalls(200)
#     optimizer.setMaxIter(200)

#     optimizer.exec()

#     result = optimizer.getResult()
#     print(result.message)
#     print(' Current function value:', result.fun_val)
#     print(' Iterations:', result.iters)
#     print(' Funcion evaluations:', result.fcalls)
#     print(' Optimized para W:', result.fcalls)
    
#     run_for_result(machine, prog, c, 1000)

def fermi_test(machine, num_q = 4, num_c = 4):
    p1 = FermionOperator()
    p2 = FermionOperator({'3+ 1 2+ 0':3, '1+ 0': 2, })
    p3 = FermionOperator('1+ 0', 2)
    p4 = FermionOperator(2)
    p5 = p2

    plus = p2 + p3
    minus = p2 - p3
    mul = p2 * p3

    print('a + b = {}'.format(plus))
    print('a - b = {}'.format(minus))
    print('a * b = {}'.format(mul))
    print(p2.normal_ordered())

def verFermi_test(machine, num_q = 4, num_c = 4):
    a = var(2, True)
    b = var(3, True)
    fermion_op = VarFermionOperator('1+ 0', a)
    pauli_op = VarPauliOperator('z1 z0', b)

    print(fermion_op)
    print(pauli_op)

def JW_test(machine, num_q = 4, num_c = 4):
    def get_fermion_jw(fermion_item):
        pauli = PauliOperator('', 1)
        for i in fermion_item:
            op_qubit = i[0]
            op_str = ''
            for j in range(op_qubit):
                op_str += 'z' + str(j) + ' '
            op_str1 = op_str + 'x' + str(op_qubit)
            op_str2 = op_str + 'y' + str(op_qubit)
            pauli_map = {}
            pauli_map[op_str1] = 0.5
            if i[1]:
                pauli_map[op_str2] = -0.5j
            else:
                pauli_map[op_str2] = 0.5j
            print(pauli_map)
            pauli *= PauliOperator(pauli_map)
        return pauli
    def JW_transform(fermion_op):
        data = fermion_op.data()
        print(data)
        pauli = PauliOperator()
        for term in data:
            pauli += get_fermion_jw(term[0][0]) * term[1]
        return pauli
    p2 = FermionOperator({'1+ 0': 2, })
    print(p2)
    pauli = JW_transform(p2)
    print(pauli)

def h_test(machine, num_q = 4, num_c = 4):
    pq.init(pq.QMachineType.CPU)
    q = pq.qAlloc_many(4)

    # 构建pauli 算子
    X = np.mat([[0, 1], [1, 0]])
    Y = np.mat([[0, -1j], [1j, 0]])
    Z = np.mat([[1, 0], [0, -1]])
    t = np.pi

    circuit_x = pq.create_empty_circuit()
    circuit_y = pq.create_empty_circuit()
    circuit_z = pq.create_empty_circuit()

    # 构造Hamiltonian Operator
    circuit_x << pq.H(q[0]) \
            << pq.X(q[0]) \
            << pq.RZ(q[0], -t) \
            << pq.X(q[0]) \
            << pq.RZ(q[0], t) \
            << pq.H(q[0])

    circuit_y << pq.RX(q[0], t / 2) \
            << pq.X(q[0]) \
            << pq.RZ(q[0], -t) \
            << pq.X(q[0]) \
            << pq.RZ(q[0], t) \
            << pq.RX(q[0], -t / 2)

    circuit_z << pq.X(q[0]) \
            << pq.RZ(q[0], -t) \
            << pq.X(q[0]) \
            << pq.RZ(q[0], t)

    operator_x = pq.QOperator(circuit_x)
    operator_y = pq.QOperator(circuit_y)
    operator_z = pq.QOperator(circuit_z)

    unitary_x = operator_x.get_matrix()
    unitary_y = operator_y.get_matrix()
    unitary_z = operator_z.get_matrix()
    print(operator_x)

    conf = complex(0, -1)
    U_x = pq.expMat(conf, X, t)
    U_y = pq.expMat(conf, Y, t)
    U_z = pq.expMat(conf, Z, t)
    print(U_x)

    f_ave_x = pq.average_gate_fidelity(U_x, unitary_x)
    f_ave_y = pq.average_gate_fidelity(U_y, unitary_y)
    f_ave_z = pq.average_gate_fidelity(U_z, unitary_z)

    print("Pauli-X Average Gate Fidelity: F = {:f}".format(f_ave_x))
    print("Pauli-Y Average Gate Fidelity: F = {:f}".format(f_ave_y))
    print("Pauli-Z Average Gate Fidelity: F = {:f}".format(f_ave_z))


def qcl_circuits(machine, num_q = 4, num_c = 4):
    def build_circuit(qubits):
        cir = pq.QCircuit()
        for i in range(len(qubits)):
            cir.insert(pq.RZ(qubits[i], np.pi / 2))
            cir.insert(pq.RY(qubits[i], np.pi / 2))
            cir.insert(pq.RZ(qubits[i], np.pi / 2))
        for d in range(3,6*4):
            for i in range(len(qubits) - 1):
                cir.insert(pq.CNOT(qubits[i], qubits[i + 1]))
            cir.insert(pq.CNOT(qubits[len(qubits) - 1], qubits[0]))

            for i in range(len(qubits)):
                cir.insert(pq.RY(qubits[i], np.pi / 2))
        return cir
    pq.init(pq.QMachineType.CPU)
    q = pq.qAlloc_many(num_q)
    c = pq.cAlloc_many(num_c)
    cir = pq.QCircuit()
    subcir = build_circuit(q)
    cir.insert(subcir)
    prog = pq.QProg()
    prog.insert(cir)
    pauli_dict = {'Z0': 1}
    print(cir)

if __name__ == "__main__":
    # 创建一个量子虚拟机
    machine = pq.init_quantum_machine(pq.QMachineType.CPU)

    # easy_test(machine)
    # oracle_test(machine, 2, 2)
    # ghz_test(machine, 4, 4)
    # swap_test(machine, 4, 4)
    # qif_test(machine)
    # qwhile_test(machine)
    # qW_test(machine, 3, 3)
    # pouli_test(machine)
    # qaoa_test(machine)
    # fermi_test(machine)
    # verFermi_test(machine)
    # JW_test(machine)
    # h_test(machine)
    # qcl_circuits(machine)
    
    input = np.random.randn(4)
    weights = np.random.randn(4, 6)
    qlist = qAlloc_many(4)
    def build_circuit(weights, qubits):
        cir = pq.QCircuit()
        for i in range(len(qubits)):
            cir.insert(pq.RZ(qubits[i], weights[i][0]))
            cir.insert(pq.RY(qubits[i], weights[i][1]))
            cir.insert(pq.RZ(qubits[i], weights[i][2]))
        for d in range(3, 6):
            for i in range(len(qubits) - 1):
                cir.insert(pq.CNOT(qubits[i], qubits[i + 1]))
            cir.insert(pq.CNOT(qubits[len(qubits) - 1], qubits[0]))

            for i in range(len(qubits)):
                cir.insert(pq.RY(qubits[i], weights[i][d]))
        return cir
    weights = weights.reshape([4, 6])
    cir = pq.QCircuit()
    cir.insert(AmplitudeEmbeddingCircuit(input, qlist))
    prog = build_circuit(weights, qlist)
    cir.insert(prog)
    prob = machine.prob_run_dict(cir, qlist[0], -1)
    print(cir)
