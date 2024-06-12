import numpy as np
from math import pi
import pyqpanda as pq
from pyqpanda import *
from functools import partial
import matplotlib.pyplot as plt

machine = pq.init_quantum_machine(pq.QMachineType.CPU)

def get_ccsd_n_term(qn, en):
    if en > qn:
        assert False
    return int((qn-en)*en+(qn-en)*(qn-en-1)*en*(en-1)/4)

# 构建费米子哈密顿量
def get_ccsd_var(qubit_number, electron_number, para_list):
    if electron_number > qubit_number:
        assert False
    if electron_number == qubit_number:
        return VarFermionOperator()
    if get_ccsd_n_term(qubit_number, electron_number) != len(para_list):
        assert False

    cnt = 0

    var_fermion_op = VarFermionOperator()
    for i in range(electron_number):
        for ex in range(electron_number, qubit_number):
            var_fermion_op += VarFermionOperator(str(ex) + '+ ' + str(i), para_list[cnt])
            cnt += 1
    return var_fermion_op

def get_fermion_jordan_wigner(fermion_item):
    pauli = PauliOperator('', 1)
    for i in fermion_item:
        op_qubit = i[0]
        op_str = ''
        for j in range(op_qubit):
            op_str += 'Z' + str(j) + ' '
        op_str1 = op_str + 'X' + str(op_qubit)
        op_str2 = op_str + 'Y' + str(op_qubit)
        pauli_map = {}
        pauli_map[op_str1] = 0.5
        if i[1]:
            pauli_map[op_str2] = -0.5j
        else:
            pauli_map[op_str2] = 0.5j
        pauli *= PauliOperator(pauli_map)
    return pauli

def JordanWignerTransformVar(var_fermion_op):
    data = var_fermion_op.data()
    var_pauli = VarPauliOperator()
    for i in data:
        one_pauli = get_fermion_jordan_wigner(i[0][0])
        for j in one_pauli.data():
            var_pauli += VarPauliOperator(j[0][1], complex_var(
                i[1].real()*j[1].real-i[1].imag()*j[1].imag,
                i[1].real()*j[1].imag+i[1].imag()*j[1].real))
    return var_pauli
def JW_transform(fermion_op):
        data = fermion_op.data()
        pauli = PauliOperator()
        for term in data:
            pauli += get_fermion_jordan_wigner(term[0][0])
        return pauli


def cc_to_ucc_hamiltonian_var(cc_op):
    pauli = VarPauliOperator()
    for i in cc_op.data():
        pauli += VarPauliOperator(i[0][1], complex_var(var(-2)*i[1].imag(), var(0)))
    return pauli

def perpareInitialState(qlist, en):
    circuit = QCircuit()
    if len(qlist) < en:
        return circuit
    for i in range(en):
        circuit << X(qlist[i])
    return circuit

def simulate_one_term_var(qlist, hamiltonian_term, coeff, t):
    vqc = VariationalQuantumCircuit()
    if len(hamiltonian_term) == 0:
        return vqc
    tmp_qlist = []
    for q, term in hamiltonian_term.items():
        if term == 'X':
            vqc << H(qlist[q])
        elif term == 'Y':
            vqc << RX(qlist[q], pi/2)
        tmp_qlist.append(qlist[q])
    size = len(tmp_qlist)
    if size == 1:
        vqc << VariationalQuantumGate_RZ(tmp_qlist[0], 2 * coeff * t)
    elif size > 1:
        for i in range(size - 1):
            vqc << CNOT(tmp_qlist[i], tmp_qlist[size - 1])
        vqc << VariationalQuantumGate_RZ(tmp_qlist[size - 1], 2 * coeff * t)
        for i in range(size - 1):
            vqc << CNOT(tmp_qlist[i], tmp_qlist[size - 1])
    # dagger
    for q, term in hamiltonian_term.items():
        if term == 'X':
            vqc << H(qlist[q])
        elif term == 'Y':
            vqc << RX(qlist[q], -np.pi/2)
    return vqc

def simulate_hamiltonian_var(qubit_list,var_pauli,t,slices=3):
    vqc = VariationalQuantumCircuit()
    for i in range(slices):
        for j in var_pauli.data():
            term = j[0][0]
            vqc.insert(simulate_one_term_var(qubit_list, term, j[1].real(), t/slices))
    return vqc

def GradientDescent(mol_pauli, n_qubit, n_en, iters):
    n_para = get_ccsd_n_term(n_qubit, n_en)
    
    para_vec = []
    var_para = []
    for i in range(n_para):
        var_para.append(var(0.5, True))
        para_vec.append(0.5)
    print(n_qubit, n_en)
    fermion_cc = get_ccsd_var(n_qubit, n_en, var_para)
    pauli_cc = JordanWignerTransformVar(fermion_cc)
    print(pauli_cc)
    ucc = cc_to_ucc_hamiltonian_var(pauli_cc)
    
    machine = init_quantum_machine(QMachineType.CPU)
    qlist = machine.qAlloc_many(n_qubit)

    vqc = VariationalQuantumCircuit()
    vqc << perpareInitialState(qlist, n_en)
    vqc << simulate_hamiltonian_var(qlist, ucc, 1.0, 3)

    loss = qop(vqc, mol_pauli, machine, qlist)
    gd_optimizer = MomentumOptimizer.minimize(loss, 0.1, 0.9)
    leaves = gd_optimizer.get_variables()

    min_energy = float('inf')
    for i in range(iters):
        gd_optimizer.run(leaves, 0)
        loss_value = gd_optimizer.get_loss()
        print(loss_value)
        if loss_value < min_energy:
            min_energy = loss_value
            for m, n in enumerate(var_para):
                para_vec[m] = eval(n, True)[0][0]
    
    
    return min_energy

if __name__ == '__main__':
    energies = []
    # fermion_op = VarFermionOperator({'1': complex_var(var(2)),'2+ 1': complex_var(var(2)), '3+ 1': complex_var(var(2)), '4+ 1': complex_var(var(2)), })
    # fermion_op = FermionOperator({'2+ 0': 1, '3+ 0': 1, '2+ 1': 1, '3+ 1': 1, '3+ 2+ 1 0': 1, })
    # pauil_op = JW_transform(fermion_op)
    pauil_op = PauliOperator({"" : 0.250000,
"Z2" : 0.125000,
"Z0 Z2" : -0.125000,
"Z1 Z2" : -0.125000,
"Z0 Z1 Z2" : -0.125000,
"Z3" : 0.250000,
"Z2 Z3" : 0.125000,
"Z0 Z2 Z3" : -0.125000,
"Z1 Z2 Z3" : -0.125000,
"Z0 Z1 Z2 Z3" : -0.125000})
    # n_qubit = pauil_op.getMaxIndex()
    print(pauil_op)
    energies.append(GradientDescent(pauil_op, 4, 1, 30))
    print(energies)
