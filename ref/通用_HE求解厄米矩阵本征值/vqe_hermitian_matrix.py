#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pyqpanda as pq
import numpy as np


# In[2]:


# Prepare Hardware-Efficient ansatz
def prepare_HE_ansatz(qlist, para):  
    '''
    prepare Hardware-Efficient ansatz, return a QCircuit
    
    Args:
        qlist(QVec): qubit list
        para(list[float64]): initial parameter.The number of parameters is four times qubit number.
    Return:
        quantum circuit(QCircuit)
    
    ''' 
    circuit = pq.QCircuit()
    qn = len(qlist)
    for i in range(qn):
        circuit.insert(pq.RZ(qlist[i], para[4*i]))  
        circuit.insert(pq.RX(qlist[i], para[4*i+1]))
        circuit.insert(pq.RZ(qlist[i], para[4*i+2]))
        
    for j in range(qn-1):
        ry_control = pq.RY(qlist[j+1], para[4*j+3]).control(qlist[j])
        circuit.insert(ry_control)
    
    ry_last = pq.RY(qlist[0], para[4*qn-1]).control(qlist[qn-1])
    circuit.insert(ry_last)
    #print(circuit)   
    return circuit


# In[3]:


def transform_base(qlist,component):
    '''
    choose measurement basis, it means rotate all axis to z-axis
    
    Args:
        qlist(QVec): qubit list
        component(List[Tuple[Dict[int,str],float]): one term paulioperator and coefficient
                                  e.g.({0: 'Y', 1: 'Z', 2: 'X'}, 1.0)
    Return:
        quantum circuit(QCircuit)
    
    '''
    circuit = pq.QCircuit()
    for i, j in component[0].items():
        if j=='X':
            circuit.insert(pq.H(qlist[i]))
        elif j=='Y':
            circuit.insert(pq.RX(qlist[i],-np.pi/2))
        elif j=='Z':
            pass
        else:
            assert False
    return circuit


# In[4]:


def parity_check(state, paulidict):
    '''
    parity check, that is, to check how many "1" in the state after measurement. 
    
    Args:
        state(str):the state after measurement
        paulidict(dict):paulioperator dictionary. e.g.{0: 'Y', 1: 'Z', 2: 'X'}
    Return:
        bool:True for odd count, False for even count
        
    '''
    check=0
    state=state[::-1]  #要将顺序进行翻转，因为量子比特默认低位在右，而python slice index是从左开始数起
    for i in paulidict:  #默认的测量是对所有量子比特进行测量，但我们只关心component[0]中待测量的量子位
        if state[i]=='1':
            check+=1
    
    return check%2


# In[5]:


def get_expectation(machine_type,qn,component,para):
    '''
    get expectation of one term of hermitian matrix.
    
    Args:
      machine_type(QQMachineType):QMachineType.CPU,QMachineType.CPU_SINGLE_THREAD,QMachineType.GPU,QMachineType.NOISE
      qn(int):qubit number
      component(List[Tuple[Dict[int,str],float]]): paulioperator and coefficient, e.g.({0: 'Y', 1: 'Z', 2: 'X'}, 1.0)
      para(list[float64]): initial parameter. The number of parameters is four times qubit number.
    Return:
      expectation value of one component of hermitian matrix (float64)
    '''
    machine = pq.init_quantum_machine(machine_type)
    qlist=pq.qAlloc_many(qn)
    
    prog=pq.QProg()
    HE_circuit = prepare_HE_ansatz(qlist, para) #准备拟设线路
    prog.insert(HE_circuit)  #添加拟设线路
    prog.insert(pq.BARRIER(qlist))
    # 添加测量线路
    if component[0]!='':
        prog.insert(transform_base(qlist,component)) #如果测量的基中泡利项不为零，则进行转基操作
    #print(prog)
    result = machine.prob_run_dict(prog,qlist,-1)
    expectation=0
    for i in result:
        if parity_check(i, component[0]):
            expectation-=result[i]
        else:
            expectation+=result[i]       
    return expectation*component[1] 


# In[6]:


def get_eigenvalue(machine_type,qn,matrix,para):
    '''
    get eigenvalue of a hermitian matrix.
    
    Args:
        machine_type(QQMachineType):QMachineType.CPU,QMachineType.CPU_SINGLE_THREAD,QMachineType.GPU,QMachineType.NOISE
        qn(int):qubit number
        matrix(List[Tuple[Dict[int,str],float]]): matrix expressed by paulioperator e.g.:[({0: 'X', 2: 'Y'}, 2.0), ({1: 'Y', 2: 'Z', 3: 'X'}, 1.0)]
        para(list[float64]): initial parameter. The number of parameters is four times qubit number.
    Return:
        eigenvalue of a hermitian matrix(float64)
    '''
    expectation=0
    for component in matrix:
        expectation+=get_expectation(machine_type=machine_type,qn=qn,component=component,para=para)
    expectation=float(expectation.real)
    return expectation


# In[7]:


import scipy.optimize as opt

def func(x):
    return get_eigenvalue(machine_type,qn, Hf, x)
    
def opt_scipy():
    res = opt.minimize(func,init_para,
            method = "SLSQP",
            options = {"maxiter":100},
            tol=1e-9)
    print(res)


# In[8]:


# 使用pyqpanda实现厄密矩阵的分解(这个是临时接口，后续接口形式会改为f = pq.matrix_decompose_hamiltonian(mat))
mat = np.array([[2,1,4,2],[1,3,2,6],[4,2,2,1],[2,6,1,3]])
f = pq.PauliOperator()
machine = pq.CPUQVM()
machine.init_qvm()
pq.matrix_decompose_hamiltonian(machine, mat, f)
print(f)


# In[9]:


if __name__ == "__main__":
    
    Hf = f.to_hamiltonian(True)
    machine_type = pq.QMachineType.CPU
    qn = 2
    init_para = np.random.rand(qn*4)
    opt_scipy()


# In[13]:


# 使用numpy进行验证
import numpy as np
mat = np.array([[2,1,4,2],[1,3,2,6],[4,2,2,1],[2,6,1,3]])
np.min(np.linalg.eigvals(mat))


# In[ ]:




