from datetime import datetime
from pathlib import Path
import time
from multiprocessing import Process
from timeit import default_timer as timer
from multiprocessing import Pool, cpu_count
import random 
from random import sample
from qiskit import QuantumCircuit
import qiskit as qiskit
import qiskit.visualization
from numpy import sqrt 
from numpy import transpose 
from numpy import conj
import scipy
from numpy import exp
from numpy import log
import numpy as np
import random as rand
import shutil

from qiskit import QuantumCircuit
from qiskit import Aer, transpile
from qiskit.tools.visualization import plot_histogram, plot_state_city
import qiskit.quantum_info as qi
from qiskit.visualization import plot_histogram

from qiskit import QuantumCircuit, transpile
from qiskit.providers.basicaer import QasmSimulatorPy
from qiskit import Aer
from qiskit import QuantumCircuit

from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.test.reference_circuits import ReferenceCircuits
from qiskit_ibm_runtime import QiskitRuntimeService

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.visualization import plot_histogram
from qiskit.providers.basicaer import QasmSimulatorPy

from qiskit import QuantumCircuit, transpile
from qiskit.providers.aer import QasmSimulator
from qiskit.visualization import plot_histogram
from numpy import zeros
from numpy.random import rand
from numpy.linalg import qr
from numpy import linalg
from numpy.linalg import inv
from numpy.linalg import eig
from numpy.linalg import matrix_power

from numpy import matmul
from numpy import divide
from numpy import diagonal
from numpy import floor
from numpy import copy
from qiskit import quantum_info as qinfo

from qiskit.quantum_info import Statevector
from numpy import math
from numpy import pi


import numpy as np
from qiskit import(
  QuantumCircuit,
  execute,
  Aer)

from matplotlib import pyplot as plt

#service = QiskitRuntimeService()

#program_inputs = {'iterations': 1}
#options = {"backend_name": "ibmq_qasm_simulator"}
#job = service.run(program_id="hello-world",
#                options=options,
#                inputs=program_inputs
#                )
#print(f"job id: {job.job_id}")
#result = job.result()
#print(result)

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, transpile
from qiskit.tools.visualization import circuit_drawer
from qiskit.quantum_info import state_fidelity
from qiskit import BasicAer

from numpy import log
from numpy import exp

import numpy as np
#from numpy import *
from joblib import Parallel, delayed
from multiprocessing import Pool
import numpy as np
#from time import clock     
from time import process_time
#import qiskit fg
import matplotlib
import numpy as np
from random import randrange

import multiprocessing

import matplotlib.pyplot as plt
import sympy
#from sympy import *
import itertools
from IPython.display import display
#init_printing()
import math
from tempfile import TemporaryFile
#qiskit.__qiskit_version__
import numpy as np
import tensorflow as tf
import pandas as pd
from pandas import read_csv
from matplotlib import pyplot
import csv
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import os, fnmatch
import sys, getopt

from tensorflow.keras.layers import Dense, Activation, Dropout, Input, LSTM, Reshape, Lambda, RepeatVector
#from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, TimeDistributed, SimpleRNN
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from scipy.optimize import minimize
from scipy.linalg import logm
from qiskit.compiler import assemble



def denMatCostDiag(lamSq, sx, sy, sz):
    legendre = lamSq[-1];
    mu = [[1/2*(1+sz), 1/2*(sx-1j*sy)], [1/2*(sx+1j*sy), 1/2*(1-sz)]]
    mueigval, mueigvec = eig(mu)
    #a=denVec[0]; b=denVec[1]; c=denVec[2]; d=denVec[3];
    print("mueigval = ", mueigval)
    Lambda = sum(np.multiply(np.multiply(lamSq[:-1], lamSq[:-1])-\
                             mueigval[:], np.multiply(lamSq[:-1], lamSq[:-1])-\
                             mueigval[:]))-legendre*(sum(np.multiply(lamSq[:-1], lamSq[:-1]))-1)
    
    return Lambda

def denMat(denVec):
    a=denVec[0]; b=denVec[1]; c=denVec[2]; d=denVec[3];
    rho = np.divide([[a**2+b**2+c**2, d*(b-1j*c)], [d*(b+1j*c), d**2]], \
                    (a**2+b**2+c**2+d**2))
    return rho






#def denMatCost(denVec, sx, sy, sz):
def denMatCost(denVec, sx, sy, sz):
    #print("denMat = ", denMat(denVec))
#    sx=-.1;sy=.5;sz=.5;
    a=denVec[0]; b=denVec[1]; c=denVec[2]; d=denVec[3];
    #print("denVec=", denVec)
    norm=(a**2+b**2+c**2+d**2);
    #cost = (1-a)**2+2*a
    #(2*b*d-sx)**2+(2*c*d-sy)**2
    cost1 = (2*b*d-norm*sx)**2/(norm*(2*b*d)) + \
    (2*c*d-norm*sy)**2/(norm*(2*c*d)) + \
    (a**2+b**2+c**2-d**2-norm*sz)**2/(norm*(a**2+b**2+c**2-d**2))
    #derivCost1da = (norm*sx-2*b*d)*(a*norm*sx+2*b*d*a)/(b*d*norm**2)+ \
    #(norm*sx-2*c*d)*(a*norm*sx+2*c*d*a)/(c*d*norm**2)+\
    #2*(2*a-2*a*sz)*(a**2+b**2+c**2-d**2-norm*sz)/(norm*(a**2+b**2+c**2-d**2))+\
    #-2*a*(a**2+b**2+c**2-d**2-norm*sz)**2/(norm**2*(a**2+b**2+c**2-d**2))+\
    #-(a**2+b**2+c**2-d**2-norm*sz)**2/(norm*(a**2+b**2+c**2-d**2)**2)
    
    
    cost2 = (2*b*d-norm*sx)**2/(norm**2*(sx)) + \
    (2*c*d-norm*sy)**2/(norm**2*sy) + \
    (a**2+b**2+c**2-d**2-norm*sz)**2/(norm**2*sz)
    #derivCost2da = 
    
    cost=cost1
    #derivCost=[derivCost1da, derivCost1db, derivCost1dc, derivCost1dd]
    
    #derivCost=[derivCost2da, derivCost2db, derivCost2dc, derivCost2dd]
    
    #print("cost = ", cost)
    #cost = (2*a**2+2*b**2+2*c**2-1-sz)**2
    #return cost1, cost2
    return cost

def denMatDerCost(denVec, sx, sy, sz):
    #print("denMat = ", denMat(denVec))
#    sx=-.1;sy=.5;sz=.5;
    a=denVec[0]; b=denVec[1]; c=denVec[2]; d=denVec[3];
    #print("denVec=", denVec)
    norm=(a**2+b**2+c**2+d**2);
    #cost = (1-a)**2+2*a
    #(2*b*d-sx)**2+(2*c*d-sy)**2
    derivCost1da=-2*a*sx*(2*b*d-norm*sx)/(b*d*norm) - a*(2*b*d-norm*sx)**2/(b*d*norm**2)\
    -2*a*sx*(2*c*d-norm*sx)/(c*d*norm) - a*(2*c*d-norm*sx)**2/(c*d*norm**2)\
    +2*(2*a-2*a*sz)*(a**2+b**2+c**2-d**2-norm*sz)/((a**2+b**2+c**2-d**2)*norm)\
    -2*a*(a**2+b**2+c**2-d**2-norm*sz)**2/((a**2+b**2+c**2-d**2)*(norm**2))\
    -2*a*(a**2+b**2+c**2-d**2-norm*sz)**2/((a**2+b**2+c**2-d**2)**2 * norm);
    
    derivCost1db=(2*d-2*b*sx)*(2*b*d-norm*sx)/(b*d*norm) - (2*b*d-norm*sx)**2/(d*norm**2)\
    -(2*b*d-norm*sx)**2/(2* b**2 *d*norm) - 2*b*sx*(2*c*d-norm*sx)/(c*d*norm)\
    -b*(2*c*d-norm*sx)**2/(c*d*norm**2)\
    +2*(2*b-2*b*sz)*(a**2+b**2+c**2-d**2-norm*sz)/((a**2+b**2+c**2-d**2)*(norm))\
    -2*b*(a**2+b**2+c**2-d**2-norm*sz)**2/((a**2+b**2+c**2-d**2) * norm**2)\
    -2*b*(a**2+b**2+c**2-d**2-norm*sz)**2/((a**2+b**2+c**2-d**2)**2 * norm);    

    derivCost1dc=-2*c*sx*(2*b*d-norm*sx)/(b*d*norm) - c*(2*b*d-norm*sx)**2/(b*d*norm**2)\
    +(2*d-2*c*sx)*(2*c*d-norm*sx)/(c*d*norm) - (2*c*d-norm*sx)**2/(d*norm**2)\
    -(2*c*d-norm*sx)**2/(2*c**2*d*norm)\
    +2*(2*c-2*c*sz)*(a**2+b**2+c**2-d**2-norm*sz)/((a**2+b**2+c**2-d**2)*(norm))\
    -2*c*(a**2+b**2+c**2-d**2-norm*sz)**2/((a**2+b**2+c**2-d**2) * norm**2)\
    -2*c*(a**2+b**2+c**2-d**2-norm*sz)**2/((a**2+b**2+c**2-d**2)**2 * norm);    

    derivCost1dd=(2*b-2*d*sx)*(2*b*d-norm*sx)/(b*d*norm) - (2*b*d-norm*sx)**2/(b*norm**2)\
    -(2*b*d-norm*sx)**2/(2*b*d**2*norm) + (2*c-2*d*sx)*(2*c*d-norm*sx)/(c*d*norm)\
    -(2*c*d-norm*sx)**2/(c*norm**2) - (2*c*d-norm*sx)**2/(2*c*d**2*norm)\
    +2*(-2*d-2*d*sz)*(a**2+b**2+c**2-d**2-norm*sz)/((a**2+b**2+c**2-d**2)*(norm))\
    -2*d*(a**2+b**2+c**2-d**2-norm*sz)**2/((a**2+b**2+c**2-d**2) * norm**2)\
    +2*d*(a**2+b**2+c**2-d**2-norm*sz)**2/((a**2+b**2+c**2-d**2)**2 * norm);    

    derivCost=[derivCost1da, derivCost1db, derivCost1dc, derivCost1dd]
        
    return derivCost

def maxLikelihoodDen(sx, sy, sz, methodML):
    #print("[sx, sy, sz] = ", [sx, sy, sz])
    Delta=1/4*(1-sx**2-sy**2-sz**2)
    M11=1/2*(1-sz);
    #print("sqrt(Delta/M11) = ", sqrt(Delta/M11))
    #print("M11 = ", M11)
    #print("Delta = ", Delta)
    #print("1-sz = ", 1-sz)
    #print("D/M = ", Delta/M11)
    T0 =np.array([[sqrt(Delta/M11+0j), 0], \
           [(sx+1j*sy)/(sqrt(2*(1-sz))), \
            sqrt(1/2*(1-sz))]])
    #print("T0 = ", T0)
    #denMat0=np.dot(np.transpose(conj(T0)), T0)
    #print("denMat0 = ", denMat0)
    #print("T0[1, 0] = ", np.real(complex(T0[1, 0])))
    #print("real T0[1, 0] = ", np.real(T0[1, 0]))
    denVec0=[np.real(complex(T0[0, 0])), np.real(complex(T0[1, 0])), \
             np.imag(complex(T0[1, 0])), np.real(complex(T0[1, 1]))]

    #a0=denMat0[0];    b0=denMat0[1];    c0=denMat0[2];    d0=denMat0[3];
    
    #denMatParam = [a, b, c, d];
    
    #res = minimize(fun, (2, 0), method='SLSQP') #, bounds=bnds,\
               #constraints=cons)
    #bnds = ((-float('inf'), float('inf')), (0, None))        
    #res = minimize(denMatCost, denVec0, args=(sx, sy, sz), \
    #               method='SLSQP', jac=None)
    
    res = minimize(denMatCost, denVec0, args=(sx, sy, sz), \
                   method=methodML, jac=None)
    
    #res = minimize(denMatCost, denVec0, args=(sx, sy, sz), \
    #               method='TNC', jac=None)

    
    physDenMat = denMat(res.x)
    
    #print("density mat from TNC = ", denMat(res.x))
    #res = minimize(denMatCost, denVec0, args=(sx, sy, sz), \
    #method='nelder-mead', options={'xatol': 1e-8, 'disp': True})
    #res = minimize(denMatCost, denVec0, args=(sx, sy, sz), \
    #method='Powell', jac=None)
    #res = minimize(denMatCost, denVec0, args=(sx, sy, sz),\
    #               method='Newton-CG', jac=denMatDerCost) #, hess=None, hessp=None) 
    #print("density mat from Newton = ", denMat(res.x))
    #print("res Newton = ", res)
    return physDenMat, res.x, res


def denMat3param(denVec):
    
    a=denVec[0]; b=denVec[1]; c=denVec[2];
    d=sqrt(1-a**2-b**2-c**2);
    rho = [[a**2+b**2+c**2, d*(b-1j*c)], [d*(b+1j*c), d**2]]
    
    return rho

def denMat3paramCost(denVec, sx, sy, sz):
    a=denVec[0]; b=denVec[1]; c=denVec[2]; 
    norm=1;d=sqrt(1-a**2-b**2-c**2); print("d=", d);
    cost1 = (2*b*d-norm*sx)**2/(norm*(2*b*d)) + \
    (2*c*d-norm*sy)**2/(norm*(2*c*d)) + \
    (a**2+b**2+c**2-d**2-norm*sz)**2/(norm*(a**2+b**2+c**2-d**2))
        
    cost2 = (2*b*d-norm*sx)**2/(norm**2*(sx)) + \
    (2*c*d-norm*sy)**2/(norm**2*sy) + \
    (a**2+b**2+c**2-d**2-norm*sz)**2/(norm**2*sz)
    
    cost=cost2
    
    #return cost1, cost2
    return cost


def maxLikelihoodDen3Param(sx, sy, sz):
    Delta=1/4*(1-sx**2-sy**2-sz**2)
    M11=1/2*(1-sz);
    T0 =np.array([[sqrt(Delta/M11), 0], \
           [(sx+1.0j*sy)/(sqrt(2*(1-sz))), \
            sqrt(1/2*(1-sz))]])
    print("T0=", T0)
    denMat0=np.matmul(np.transpose(conj(T0)), T0)
    print("denMat0 = ", denMat0)
    #print("T0[1, 0] = ", np.real(complex(T0[1, 0])))
    #print("real T0[1, 0] = ", np.real(T0[1, 0]))
    denVec0=[np.real(complex(T0[0, 0])), np.real(complex(T0[1, 0])), \
             np.imag(complex(T0[1, 0])), np.real(complex(T0[1, 1]))]
    
    #a0=denMat0[0];    b0=denMat0[1];    c0=denMat0[2];    d0=denMat0[3];
    
    #denMatParam = [a, b, c, d];
    
    #res = minimize(fun, (2, 0), method='SLSQP') #, bounds=bnds,\
               #constraints=cons)
    #bnds = ((-float('inf'), float('inf')), (0, None)) 
    print("denVec0 = ", denVec0)
    res = minimize(denMat3paramCost, denVec0[0:3], args=(sx, sy, sz), \
                   method='SLSQP', jac=None)
    print("density mat from SLSQP = ", denMat3param(res.x))
    #res = minimize(denMatCost, denVec0, args=(sx, sy, sz), \
    #method='nelder-mead', options={'xatol': 1e-8, 'disp': True})
    #res = minimize(denMatCost, denVec0, args=(sx, sy, sz), \
    #method='Powell', jac=None)
    #res = minimize(denMat3paramCost, denVec0, args=(sx, sy, sz),\
    #               method='Newton-CG', jac=denMatDerCost) #, hess=None, hessp=None) 
    #print("density mat from Newton = ", denMat(res.x))
    #print("res Newton = ", res)
    return res.x, res

#def maxLikelihoodDenDiag(sx, sy, sz):
#    return res.x, res
#A=[1, 2, 3]; A[0:2]; print("A = ", A)
#sx=0.5; sy=0.2; sz=0.9
#res = maxLikelihoodDen3Param(sx, sy, sz)
#T0=np.array([[0.707106781186548*1j, 0],
# [1.1180339887499 + 0.447213595499958*1j, 0.223606797749979]])
#print("T0 = ", np.matmul(T0, T0))
#A = np.array([[17.+0.j, -3.+0.j],
#              [-7.+0.j,  1.+0.j]])

#B = np.array([[ 60.+0.j,  -4.+0.j],
#              [-12.+0.j,   0.+0.j]])
#print("T0 = ", np.matmul(A, B))



def entanglement(rho):
    eigval = eig(rho)[0]
    EE = -np.sum(np.dot(np.log2(eigval), eigval))
    
    return EE



def HadamardMidQbit(circ, nqbit):
    circ.h(int(np.floor(nqbit-1)/2), nqbit-1)
    return circ
def CNOTWithLastQbit(circ, nqbit):
    circ.cx(int(np.floor(nqbit-1)/2), nqbit-1)
    return circ


def HaarGen(dim, randseed):
    #print("randseed = ", randseed)
    random.seed(randseed);    
    realrand = [[random.random() for e in range(dim)] for e in range(dim)]
    imagrand = [[1j*random.random() for e in range(dim)] for e in range(dim)]    

    A = np.add(realrand, imagrand);
    #print("shape A = ", np.shape(A))    
    #A = random.random(dim, dim)+1j*rand(dim, dim)
    
    A = A/np.sqrt(2)    
    #print("A = ", A)    
    Q, R = linalg.qr(A)
    #print("R = ", R)
    D = np.diag(np.diag(R))
    #np.diag(np.diag(x))    
    #print("D = ", D)    
    #print("np.abs(D) = ", np.abs(D))
    #PhaseVec = np.divide(D, np.abs(D))
    PhaseVec = divide(diagonal(D), abs(diagonal(D)))    
    #print("PhaseVec = ", PhaseVec)
    PhaseArr = np.diag(PhaseVec)
    Rprime = matmul(inv(PhaseArr), R)
    #print("shape Q = ", np.shape(Q))
    #print("PhaseArr = ", np.shape(PhaseArr))    
    Qprime = matmul(Q, PhaseArr)
    eigval, eigvec = eig(Qprime)
    #print("eigval = ", eigval)
    #QprimeCl1 = [[1, 0, 0, 0],[0, 1, 0, 0], [0, 0, 0, 1],[0, 0, 1, 0]] #np.identity(4) 
    #QprimeCl2 = np.divide([[1, 1, 0, 0], [1, -1, 0, 0], [0, 0, 1, 1], [0, 0, 1, -1]],sqrt(2))
    
    #if rand()>0.5:
    #    Qprime=QprimeCl1
    #else:
    #    Qprime=QprimeCl2
    
    return Qprime, A, eigval


#QR = LinearAlgebra.QRCompactWY{Float64, Matrix{Float64}}
def angHaar(eigvalues):
    ang = np.zeros(len(eigvalues), 1)
    ang = [np.angle(eigvalues[i]) for i in range(len(eigvalues))]
    return ang

def traceVecExceptLast(state, nqbit):
    stateTr1 = qinfo.partial_trace(state, np.arange(0, nqbit))

def statHaar(dim, Nsamp, Nbin):  
    min = -1;
    max = +1;
    binVec = LinRange(min, max, Nbin+1)
    deltaBin = (max-min)/Nbin    
    prob = zeros(Nbin)
    #display(prob)
    for n in range(Nsamp):
        if n%500==0:
            print("n = ", n)        
        Qprime, A, eig = HaarGen(dim)
        ang = angHaar(eig)
        #display("ang")        
        #display(ang)        
        for n in range(dim):
            #display(ang[n])
            #display((ang[n]-min)/deltaBin)
            numBin = Int(floor((ang[n]-min)/deltaBin))+1
            #display("numBin")
            #display(numBin)
            prob[numBin] = prob[numBin]+1
        #display("prob")
        #display(prob)
    return prob

def genCircConfig(nqbit, circDepth, p, randseed, ifsave=0):
    nNodes=nqbit*circDepth
    nUnitary=int(nNodes/2)
    #lock = threading.Lock()
    
    #with lock:
    #print("randseed = ", randseed)
    random.seed(randseed);
    #print("random.seed(randseed) = ", random.seed(randseed))
    measureVec=[(random.random()<p) for i in range(nNodes)]        

    measureArr=np.zeros((circDepth, nqbit))
    for t in range(circDepth):
        measureArr[t, :]=[measureVec[i+t*nqbit] for i in range(nqbit)]
    
    #print("measureArr = ", measureArr)
    unitaryArr=np.zeros((nUnitary, 4, 4))+0j
    dim=4;
    for i in range(nUnitary):
        Q, A, eig = HaarGen(dim, randseed)
        unitaryArr[i, :, :] = np.copy(Q)
        #print("Q = ", np.real(Q))
        #print("Q*Q^{\dagger}=", matmul(Q, np.conj(np.transpose(Q))))
        
    #circ={};circ[0]=measureArr;circ[1]=unitaryArr;
    #print("circ = ", circ)
    #if ifsave==1:
    #    df = pd.DataFrame(circ);
    #    df.to_csv("Circ-nq{}-depth{}-p{}-seed{}.csv".format(nq, \
    #        circDepth, p, randseed))

    #circ = genCircConfig(nq, d, p, seed, 1);
    path = Path('~/Desktop/Hafezi/Codes/Haar/').expanduser()

    #print("path = ", path)
    path.mkdir(parents=True, exist_ok=True)

    #lb,ub = -1,1;
    #num_samples = 5;
    #x = np.random.uniform(low=lb,high=ub,size=(1,num_samples));
    #y = x**2 + x + 2;
    if ifsave==1:
        np.save(path/"measure-nq{}-depth{}-p{}-seed{}".format(nqbit, \
            circDepth, p, randseed), measureArr)
        np.save(path/"unitary-nq{}-depth{}-p{}-seed{}".format(nqbit, \
            circDepth, p, randseed), unitaryArr)
        
            
    #try:
    #    circIndArr = globals()["circIndArrL{}P{}".format(L, Prob)][:, PT]
    #    except KeyError:    
    
    """
    with open("Circ-nq{}-depth{}-p{}-seed{}.csv".format(nq, \
        circDepth, p, randseed), newline='') as csvfile:
                    #spamreader = csv.reader(csvfile, delimiter=',', quotechar=',')
        csv_reader = csv.reader(csvfile, delimiter=',')
        rowc = 0                
        for row in csv_reader: 
            if rowc==1:
                measureArrLoad = row[1][1:-1];
                #print("measureArr = ", measureArr)
                #print("np.shape(measureArr) = ", np.shape(measureArr))
                arr = measureArrLoad.split(' ')
                print("arr = ", arr)                
            #print("rowc = ", rowc)
            #print("row = ", row)
            #print("row[1] = ", row[1])
            rowc += 1       
            
        #for i in range(10):
        #    circIndArr[i, rowc-1] = int(float(row[i]))
        #    rowc+=1       
    """
    
    return measureArr, unitaryArr


#def saveCircConfig():
    
def timeEvolveAncilla(nqbit, circDep, p, initStateLabel, circConfig, renyiInd, refQbitAxis, Nshots):
    nNodes=nqbit*circDep;
    nUnitary=int(nNodes/2);
    measureArr=np.zeros((circDep, nqbit));
    measureVec = np.zeros(nNodes);
    if circConfig=="None":
        measureVec=[(rand(1)[0]<p)+0.0 for i in range(nNodes)];
        for t in range(circDep):
            measureArr[t, :]=[measureVec[i+t*nqbit] for i in range(nqbit)];
            
    
    unitaryArr=np.zeros((nUnitary, 4, 4))+0j;
    dim=4;
    if circConfig!="None":
        measureArr=circConfig[0];
        unitaryArr=circConfig[1];       
        for t in range(circDep):
            #print("measureArr = ", np.shape(measureArr))
            #print("measureVec = ", np.shape(measureVec))            
            measureVec[t*nqbit:(t+1)*nqbit] = measureArr[t, :]

    if refQbitAxis=="X":
        print()
        #print("measureVec = ", measureVec)            
        #print("measureArr = ", measureArr)
    #print("measureVec = ", measureVec)
    
    # The ancilla qubit is put at the  of the string of the qubits. This way we tensor product the states
    # from left to right. The ancilla qbit is entangled to the qbit at the middle of the string at Ind=floor(nqbit/2)
    
    n2qbit = int(floor(nqbit/2))
    state = Statevector.from_int(0, 2**(nqbit+1))
    
    A=rand(2, 2)
    unitCirc = QuantumCircuit(nqbit+1, 2)
    midind = int(floor((nqbit-1)/2))
    # Add a H gate on qubit 0    
    #circuit.h(n-1)
    #circuit.cx(n-1, midind)

    #U = Operator(circuit)    
    #identity = Matrix(one(eltype(A))I, size(A,1), size(A,1))
    identityMat = np.identity(np.shape(A)[0])
    if initStateLabel=="mixed":
        for t in range(circDep):
            if t%2==1:                
                for ngate in range(n2qbit):
                    Qprime, A, qprimEig = HaarGen(dim)                    
                    #print("Qprime = ", Qprime)
                    #print("2*ngate = ", 2*ngate)
                    gate2x2 = unitCirc.unitary(Qprime, [2*ngate, 2*ngate+1])
                    
            elif t%2==0:
                #U = identityMat
                for ngate in range(n2qbit-1):
                    print("t = ", t)
                    Qprime, A, qprimEig = HaarGen(dim)
                    #print("Qprime = ", Qprime)
                    #print("2*ngate = ", 2*ngate)
                    gate2x2 = unitCirc.unitary(Qprime, [2*ngate+1, 2*(ngate+1)])
                    
                #U = kron(U, identityMat)   

    state = state.evolve(unitCirc)
    #state.__dir__()
    BellGateCirc = QuantumCircuit(nqbit+1)
    #initState = Statevector.from_int(0, 2**(nqbit+1))        
    #q = QuantumRegister(nqbit)
    #cbit = ClassicalRegister(nqbit)
    #qc = QuantumCircuit(q, cbit)
    
    #qc.initialize(mixedState, [q[0],q[1]])
    BellGateCirc.initialize(state)
    
    hGate = BellGateCirc.h(midind)   # Hadamard gate on the last qubit. 
    cnotgate = BellGateCirc.cx(midind, nqbit)  # CNOT between the mid qubit and the last qubit. 
    state = state.evolve(BellGateCirc)
    #backend_sim = Aer.get_backend('qasm_simulator')

    # Execute the circuit on the qasm simulator.
    # We've set the number of repeats of the circuit
    # to be 1024, which is the default.
    #job = backend_sim.run(transpile(qc, backend_sim), shots=1024)
    
    qreg  = QuantumRegister((nqbit+1)) #
    qregX  = QuantumRegister((nqbit+1)) #
    qregY  = QuantumRegister((nqbit+1)) #
    qregZ  = QuantumRegister((nqbit+1)) #
    if refQbitAxis=="None": # In this case we measure the state of the reference qubit at the final step.
        cr  = ClassicalRegister((nqbit)*circDep)
        hybCirc = QuantumCircuit(qreg,cr)        
    else:
        cr  = ClassicalRegister((nqbit)*circDep+1)
        crX  = ClassicalRegister((nqbit)*circDep+1)
        crY  = ClassicalRegister((nqbit)*circDep+1)
        crZ  = ClassicalRegister((nqbit)*circDep+1)
        
        hybCirc = QuantumCircuit(qreg,cr)
        hybCircX = QuantumCircuit(qregX,crX)
        hybCircY = QuantumCircuit(qregY,crY)
        hybCircZ = QuantumCircuit(qregZ,crZ)   
        
    hybCirc.initialize(state)
    if refQbitAxis=="All":
        hybCircX.initialize(state)
        hybCircY.initialize(state)    
        hybCircZ.initialize(state)    
    Qprime=np.zeros((dim, dim))+0.0j
    for t in range(circDep):
        ### Measurement
        #cntMeasure = np.count_nonzero(measureArr)
        #qreg  = QuantumRegister(nqbit+1) # 
        #cr  = ClassicalRegister(nqbit+1)
        if t%2==0:              
            for ngate in range(n2qbit):
                if circConfig=="None":                
                    Qprime, A, qprimEig = HaarGen(dim)
                else:
                    #unitInd=ngate+int(t/2)*nqbit
                    unitInd=ngate+t*n2qbit
                    Qprime=np.copy(unitaryArr[unitInd, :, :])
                #print("unitInd = ", unitInd)
                gate2x2=hybCirc.unitary(Qprime, [2*ngate, 2*ngate+1])
                if refQbitAxis=="All":                
                    gate2x2X=hybCircX.unitary(Qprime, [2*ngate, 2*ngate+1])
                    gate2x2Y=hybCircY.unitary(Qprime, [2*ngate, 2*ngate+1])
                    gate2x2Z=hybCircZ.unitary(Qprime, [2*ngate, 2*ngate+1])                
        elif t%2==1:            
            for ngate in range(1, n2qbit+1):    
                if circConfig=="None":
                    Qprime, A, qprimEig = HaarGen(dim)
                else:
                    unitInd=ngate-1+t*n2qbit
                #print("unitInd = ", unitInd)    
                if ngate!=n2qbit:                    
                    gate2x2 = hybCirc.unitary(Qprime, [2*ngate-1, 2*ngate])
                    if refQbitAxis=="All":                                    
                        gate2x2X = hybCirc.unitary(Qprime, [2*ngate-1, 2*ngate])                    
                        gate2x2Y = hybCirc.unitary(Qprime, [2*ngate-1, 2*ngate])                    
                        gate2x2Z = hybCirc.unitary(Qprime, [2*ngate-1, 2*ngate])                                        
                else: 
                    gate2x2 = hybCirc.unitary(Qprime, [2*ngate-1, 0])                    
                    if refQbitAxis=="All":                                    
                        gate2x2X = hybCirc.unitary(Qprime, [2*ngate-1, 0])                    
                        gate2x2Y = hybCirc.unitary(Qprime, [2*ngate-1, 0])                    
                        gate2x2Z = hybCirc.unitary(Qprime, [2*ngate-1, 0])                                        
                    
                    
        for m in range(nqbit):
            if measureArr[t, m]:
                hybCirc.measure(qreg[m], cr[m+nqbit*t])                
                if refQbitAxis=="All":                                                    
                    hybCircX.measure(qregX[m], crX[m+nqbit*t])
                    hybCircY.measure(qregY[m], crY[m+nqbit*t])
                    hybCircZ.measure(qregZ[m], crZ[m+nqbit*t])
    if refQbitAxis=="Z":
        hybCirc.measure(qreg[nqbit], cr[(nqbit)*circDep])
        #hybCircZ.measure(qreg[nqbit], cr[(nqbit)*circDep])        
    elif refQbitAxis=="X":
        hybCirc.h(nqbit)
        hybCirc.measure(qreg[nqbit], cr[(nqbit)*circDep])
    elif refQbitAxis=="Y":
        hybCirc.u(np.pi/2, np.pi/2, np.pi, nqbit);
        hybCirc.measure(qreg[nqbit], cr[(nqbit)*circDep])
    elif refQbitAxis=="All":
        hybCircZ.measure(qregZ[nqbit], crZ[(nqbit)*circDep])                
        #hybCircX.u(np.pi/2, np.pi/2, np.pi, nqbit);
        hybCircX.measure(qregX[nqbit], crX[(nqbit)*circDep])        
        hybCircY.u(np.pi/2, np.pi/2, np.pi, nqbit);
        hybCircY.measure(qregY[nqbit], crY[(nqbit)*circDep])

        
    #backend_sim = Aer.get_backend('qasm_simulator')
    backend_sim = Aer.get_backend('statevector_simulator')
    
        # Execute the circuit on the qasm simulator.
        # We've set the number of repeats of the circuit
        # to be 1024, which is the default.
    if refQbitAxis=="None":
        job = backend_sim.run(transpile(hybCirc, backend_sim), shots=1)

    elif refQbitAxis=="All":        
        jobX = backend_sim.run(transpile(hybCircX, backend_sim), shots=Nshots)
        jobY = backend_sim.run(transpile(hybCircY, backend_sim), shots=Nshots)
        jobZ = backend_sim.run(transpile(hybCircZ, backend_sim), shots=Nshots)                
    else:
        job = backend_sim.run(transpile(hybCirc, backend_sim), shots=Nshots)        
        my_qobj = assemble(hybCirc)
        #my_qobj = assemble(c)
        #result = simulator.run(my_qobj).result()
        
        #backend = BasicAer.get_backend('statevector_simulator')
        #job = backend.run(transpile(qc, backend))
        #job=qiskit.execute(qc,backend,shots=500)
    #my_qobj = assemble(c)
    #result = simulator.run(my_qobj).result()
    
    counts = job.result().get_counts(hybCirc)
    if refQbitAxis=="All":            
        countsX = job.result().get_counts(hybCircX)
        countsY = job.result().get_counts(hybCircY)
        countsZ = job.result().get_counts(hybCircZ)
    
    #print("measureArr = ", measureArr)
    #convCounts = np.copy(counts);
    convCounts = {}
    #convKVec = #np.zeros((Nshots, circDep*nqbit+1))
    convKVec = []
    
    if refQbitAxis!="None" and refQbitAxis!="All":
        #print("counts = ", counts)
        #print("countsX = ", countsX)
        #print("countsY = ", countsY)        
        #for kx,vx in countsX.items():
        #    print("kx[::-1] = ", kx[::-1])
        #    print("vx = ", vx)

        #for ky,vy in countsY.items():
        #    print("ky[::-1] = ", ky[::-1])
        #    print("vy = ", vy)

        
        for k,v in counts.items():            
            #print("k[::-1] = ", k[::-1])
            #print("v = ", v)
            
            # k inverse of measureVec
            # tempInvK inverse of k => tempInvK aligned with measureVec
            tempInvK = k[::-1]
            #print("tempInvK = ", tempInvK)
            
            tempInvKArr = list(tempInvK)
            tempInvKConvArr = [int(i) for i in tempInvKArr]
            #print("tempInvKConvArr = ", tempInvKConvArr)
            #print("Dtype = ", tempInvK.dtype)
            #print("tempInvK[:-1] = ", tempInvK[:-1])
            #print("measureVec = ", measureVec)
            
            convK = 2*np.multiply(tempInvKConvArr[:-1], measureVec) - measureVec
            #convK = -1*np.multiply(tempInvKConvArr[:-1], measureVec) + 2*measureVec
            
            if tempInvKConvArr[-1]==0:
                convK=np.append(convK, [0]);
            elif tempInvKConvArr[-1]==1:
                convK=np.append(convK, [1]);
            for repetition in range(v):
                convKVec = np.append(convKVec, convK, axis=0);
                
            convKStr = ''.join(str(int(x)) for x in convK);
            
            convCounts.update({convKStr: v});
        
        #print("(convKVec) = ", np.shape(convKVec))
        
        return measureArr, convKVec, counts
    
            
    if refQbitAxis=="None":
        finalState = job.result().get_statevector(hybCirc);
        finStVec = finalState.data;
        finStVec = finStVec.reshape(len(finStVec), 1);
        finalDMArr = matmul(finStVec, conj(transpose(finStVec)));
        rhoFinDM = qinfo.DensityMatrix(finalDMArr);
        
        traceVec = np.arange(0, nqbit);
        traceState = rhoFinDM.copy();
        for i in range(nqbit):
            #traceState = qinfo.partial_trace(traceState, [nqbit-i])
            traceState = qinfo.partial_trace(traceState, [0]);
        
        rhoAncilla = np.copy(traceState);        
        #print("rhoAconvKVecncilla = ", rhoAncilla)
        #print("purity = ", np.trace(matrix_power(rhoAncilla, renyiInd)))
        eigRho, eigVecRho = eig(rhoAncilla);
        #print("eigRho = ", eigRho)
        sigmax=[[0, 1],[1, 0]]; sigmay=[[0, -1j],[1j, 0]]; sigmaz=[[1, 0],[0, -1]]; 
        rx = np.trace(matmul(rhoAncilla, sigmax));
        ry = np.trace(matmul(rhoAncilla, sigmay));
        rz = np.trace(matmul(rhoAncilla, sigmaz));        
        rhoAncillaZ = [[1/2*(1+rz), 0], [0, 1/2*(1-rz)]]
        
        rvec=[rx, ry, rz];        
        rvecNorm = np.linalg.norm(rvec);
        rvecNormZ = np.linalg.norm(rz);
        
        eps = (1+1j)*1e-16;
        #ancillaDenMat = traceMatExceptLast(rho)+eps*Matrix(I, 2, 2)
        #println("ancillaDenMat = ", ancillaDenMat)
        eigvalsDen = [1/2-rvecNorm/2, 1/2+rvecNorm/2];
        
        eigvalsDenZ = [1/2-rvecNormZ/2, 1/2+rvecNormZ/2];
        
        if renyiInd==1:
            try:
                renyiEnt = -np.real(sum([eigvalsDen[i]*np.log2((eigvalsDen[i])) for i in range(2)]))                
                renyiEntZ = -np.real(sum([eigvalsDenZ[i]*np.log2((eigvalsDenZ[i])) for i in range(2)]))                
            except y:
                if isa(y, DomainError):
                    println("domainError")
                    println("eigvalsDen = ", eigvalsDen)
                    println("ancillaDenMat = ", ancillaDenMat)                    
        else:
            renyiEnt = 1/(1-renyiInd) * np.log2(np.trace(matrix_power(rhoAncilla, renyiInd)))
            renyiEnt = np.real(renyiEnt)

            renyiEntZ = 1/(1-renyiInd) * np.log2(np.trace(matrix_power(rhoAncillaZ, renyiInd)))
            renyiEntZ = np.real(renyiEntZ)
            
        #print("renyiEnt = ", renyiEnt)
        if renyiEnt==1:
            renyiEnt=1-1e-8;
        if renyiEntZ==1:
            renyiEntZ=1-1e-8;            
        return state, rhoAncilla, renyiEnt, measureArr, renyiEntZ
    

def learning(nqbit, depth, p, measureArr, convKVec, countDic, NofNTR, NTotal, 
             delNNT, nnt1, lightCone, deltaLDim, NTest = [], testConvMeasure = [], new=[]):
    #print("NTest = ", NTest)
    #delNNT: jump in the number of training samples
    #measureArr: Array of measured qubits
    #countDic: Dictionary of measured outcomes with the repetition of a classical string.
    #delNNT: delta in the increase in the number of training samples
    #NofNTR: number of delNNT such that the total number of training samples is NofNTR*delNNT
    #convKVec: 
    
    nshots = NTotal
    nshotVec = range(nshots)
    measureRes = np.zeros((NTotal, depth, nqbit))
    ancilla = np.zeros(NTotal)
    
    #DecimalP = (str(p).replace('0.',''))
    #Prob = "0p{}".format(DecimalP)
    
    #Training data:
    # creating the measure array:
    
    for nsh in range(nshots):    
        for t in range(depth):
            
            measureRes[nsh, t, :] = np.copy(convKVec[nsh*(nqbit*depth+1)+t*nqbit:
                                                     nsh*(nqbit*depth+1)+(t+1)*nqbit])            
            
        #print("(nsh+1)*nqbit*depth+nqbit = ", (nsh+1)*nqbit*depth+nqbit)        
        ancilla[nsh] = convKVec[(nsh+1)*(nqbit*depth+1)-1]
        #print("ancilla[nsh] = ", ancilla[nsh])
        #print("\n")
        #nsh*(nqbit*depth+1)+(depth)*nqbit+nqbit
        
    #print("measureRes = ", measureRes[:10, :, :])
    #print("ancilla = ", ancilla[0:20])
    nnt = nnt1 + int(np.floor(delNNT*(NofNTR)))
    #print("nnt1 = ", nnt1)
    #print("nnt = ", nnt)    
    NSampVec = np.arange(nnt1, nnt, delNNT)
    #print("NSampVec = ", NSampVec);
    #fileNameArr = [];
    NRepLearn = 1;
            
    Ncircuit=1;
    n = 0;
    realT = depth
    if lightCone:
        if nqbit/2-1<realT+1:
            LDimOfArrays = nqbit;
        else:
            LDimOfArrays = 2*(realT+1)+2*deltaLDim;
    else:
        LDimOfArrays = nqbit;
        
    refInd = nqbit+1;
    #middleInd = 2*L; #In the new version of the time evolution code for ancillas, reference qubit is entangled to the middle qubit which is at the "end" of the chain
    middleInd = int(floor(nqbit/2)); #In the old version of the time evolution code for ancillas, reference qubit is entangled to the middle qubit which is at the "middle" of the chain
                        
    halfLDim = int(float(LDimOfArrays/2))
    
    #file = open(fileNameArr[-1], "r")
    #for t in range(realT):
                    #print("t = ", t)
    #    if not(lightCone):
    #        for l in range(nqbit):
                #measureArr[n-shiftInData, t, l] = int(x[4*L*t+2*l])
    #            if measureArr[n, t, l] == 2:
    #                measureArr[n, t, l] = -1                        
    #            if testAncillaState[n] == 2:
    #                testAncillaState[n] = 0
    #        else:
    #            break
    
    scoresArr = np.zeros(NofNTR);
    if NTest==[]:
        NTest = int(min(abs(NTotal-max(NSampVec)), 400));    
    
    #print("NTest in learning = ", NTest)
    #print("NSampVec in learning = ", NSampVec)
    
    sigmaPredict = np.zeros((NofNTR, NTest))
    for ntr in range(NofNTR):
        NSamples = int(NSampVec[ntr])
        if NSamples > 2000:
            epoch = 400
        else:
            epoch = 200
        
        measureResTrain = np.zeros((NSamples, depth, nqbit));
        ancillaTrain = np.zeros((NSamples));
        measureResTest = np.zeros((NTest, depth, nqbit));
        ancillaTest = np.zeros(NTest);
        
        vecSamples = sample(nshotVec, NSamples);
        
        trainInd = vecSamples;
        testInd = [i for i in nshotVec if i not in trainInd]
        
        for n in range(NSamples):
            measureResTrain[n, :, :] = measureRes[trainInd[n], :, :]
            ancillaTrain[n] = ancilla[trainInd[n]]
        
        for n in range(NTest):       
            measureResTest[n, :, :] = measureRes[testInd[n], :, :]
            ancillaTest[n] = ancilla[testInd[n]]
            
        #print("measureResTest = ", measureResTest[:10, :, :])
        
        convMeasure = np.zeros((NSamples, realT, LDimOfArrays, 1));
        convAncilla = np.zeros((NSamples, 1));
        
        testConvAncilla = np.zeros((NTest, 1));
                
        convMeasure[:, :, :, 0] = np.copy(measureResTrain);
        convAncilla = np.copy(ancillaTrain);
        
        if np.size(testConvMeasure)==0:
            testConvMeasure = np.zeros((NTest, realT, LDimOfArrays, 1));
            testConvMeasure[:, :, :, 0] = np.copy(measureResTest);            
            #print("testConvMeasure = ", testConvMeasure[:10, :, :, 0]);
        
        testConvAncilla = np.copy(ancillaTest);
        
        #print("testConvMeasure = ", testConvMeasure[:5, :5, :5, 0]);
        
        #print("testConvMeasure = ", testConvMeasure[:20, :10, :10, 0]);
        
        #print("testConvAncilla[:10] = ", testConvAncilla[:10]);      
        
        nnn = 512*(1 + 2*int(NSamples//1000))
        
        for nl in range(NRepLearn):
            #print("inside nrep")
            model = Sequential()
            #testConvMeasure = np.zeros((NTest, realT, LDimOfArrays, 1))                
            #model.add(Conv2D(L, (4, 4), activation='relu', kernel_initializer='he_uniform', padding='same', 
            model.add(Conv2D(int(float(LDimOfArrays/2)), (4, 4), activation='relu', kernel_initializer='he_uniform', padding='same', 
                                
            #input_shape=(realT, 2*L, 1)))
            input_shape=(realT, LDimOfArrays, 1)))
            #print("realT = ", realT)   
            #print("LDimOfArrays = ", LDimOfArrays)               
            #model.add(Conv2D(2*L, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
            model.add(Conv2D(LDimOfArrays, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
            
            #print("");
            if realT>=1:
                if realT>1:
                    model.add(MaxPooling2D((2, 2)))            
                model.add(Dropout(0.2))    

                model.add(Flatten())
                model.add(Dense(nnn, activation='relu'))
                model.add(Dropout(0.2))
                model.add(Dense(1, activation='sigmoid'))
               
                if NSamples > 20000:
                    n_epoch = 800
                elif NSamples <= 20000 and NSamples > 1000:
                    n_epoch = 600
                else:
                    n_epoch = 400

                lrate = 0.01;
                #decay = .9;    

                #sgd = SGD(lr=lrate, momentum=1, decay=decay, nesterov=False);
                sgd = SGD(learning_rate=lrate, momentum=1, nesterov=False);
                #optimizer = tf.keras.optimizers.Adam(0.001);
                kwargs = 'clipnorm';
                optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, \
                        nesterov=False, name="SGD")                                                    
                #            nesterov=False, name="SGD", **kwargs)
                model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy']);
                                
                AccHist = [];
                valAccHist = [];
                histLen = 100
                histLen1 = 10
                #print("convMeasure = ", convMeasure)
                #print("convAncilla = ", convAncilla)           
                for epoch in range(n_epoch):
                    history = model.fit(convMeasure, convAncilla, epochs = 1, batch_size = 100, validation_split=0.1, 
                    verbose=0)
                    if epoch%100==0:
                        a=1;
                        #print("acc = ", epoch, history.history['val_accuracy'][-1], history.history['accuracy'][-1])
                    
                    if epoch > histLen1:
                        if (history.history['val_accuracy'][-1]<.55 and history.history['accuracy'][-1]>.8) or \
                        (history.history['val_accuracy'][-1] > .98 and history.history['accuracy'][-1] > .98):
                            break

                    AccHist.append(history.history['accuracy'][-1])                
                    valAccHist.append(history.history['val_accuracy'][-1])                                                
                    if epoch > histLen:
                        if np.average(AccHist[-histLen:-1])-AccHist[-histLen]<0.001 :
                            a=1;
                            #print(np.average(AccHist[-histLen:-1]), AccHist[-histLen])
                            #print("no increase")
                            break
                        elif np.average(valAccHist[-histLen:-1])-valAccHist[-histLen]<0.001:
                            a=1;
                            #print(np.average(valAccHist[-histLen:-1]), valAccHist[-histLen])
                            #print("no increase")
                            break;
                sys.stdout.flush()           
                sys.stdout.flush()
                
                scores = model.evaluate(testConvMeasure, testConvAncilla, verbose=0);
                
                #print("scores[1] = ", scores[1], "NRepLearn = ", NRepLearn)
                #print("scores = ", scores)
                #scoresArr[i-2, 0] = scores[0]
                #print("c = ", c," ntr = ", ntr, " nl = ", nl)
                scoresArr[ntr] += scores[1]/NRepLearn;
                
                #print("testConvMeasure = ", testConvMeasure[:5, 1, :5, 0])
                
                predict=model.predict(testConvMeasure); #input1.reshape(1, d, nq, 1)) 
                
                #print("predict = ", predict)
                
                classes=np.argmax(predict,axis=1);
                tempSigma = 1*(predict)-(np.ones(np.shape(predict))-predict);
                
                if scores[1]>.96:
                    break
            
            if scoresArr[ntr]>.96:
                print("break");
                break
                
        sigmaPredict[ntr, 0:NTest] = np.copy(np.array(tempSigma[:, 0]))
        
        #print("scoresArr = ", scoresArr)                
        #df = pd.DataFrame(scoresArr)
        #if TequalsPT == 0:
        #    df.to_csv("accuracy-L{}-p{}-c1_{}-c2_{}-Nc{}-PT{}-nti{}-ntf{}-delNNT{}-NLrn{}.csv".format(L, p, c1, c2, Nc, PT, nnt1, nnt2, delNNT, NRepLearn))
        #if TequalsPT == 1:
        #    df.to_csv("accuracy-L{}-p{}-TeqPT-c1_{}-c2_{}-Nc{}-PT{}-nti{}-ntf{}-delNNT{}-NLrn{}.csv".format(L, p, c1, c2, Nc, PT, nnt1, nnt2, delNNT, NRepLearn))            
        
    #print("model = ", model)
    #print("sigmaPredict = ", sigmaPredict)
    return model, sigmaPredict, NTest, testConvMeasure

def genTrajLearnPredParallel(nrep, nq, d, learningT1, learnDelT, hundredp, trajNum, circ, delNNT, nshots, seed, \
                             lightCone=0, deltaLDim=0, ifsave=0,  nnt1=[], NTest=[]):
    print("NTest = ", NTest)
    learningT2=d+1;
    learnT = np.arange(learningT1, learningT2, learnDelT);
    #print("learnT = ", learnT)
    totalee=np.zeros(len(learnT));
    totalexactee=np.zeros(len(learnT));
    totaleeZ=np.zeros(len(learnT));
    totalexacteeZ=np.zeros(len(learnT));

    for i in range(nrep):
        print("i in nrep = ", i)
        #seed = seed+i                
        print("seed in parallel = ", seed+i)
        aveEeVec, aveEeZVec, exactEE, exactEEZ = genTrajLearnPred(nq, d, learningT1, learnDelT, hundredp, trajNum, 
                            circ, delNNT, nshots, seed, lightCone, deltaLDim, ifsave, nnt1, NTest);                              
        #print("totalee = ", totalee);
        #print("eeVec = ", eeVec);
        totalee = np.add(eeVec, totalee);
        totalexactee = np.add(tempexactee, totalexactee);    
    
    print("totalee = ", totalee)
    print("totalexactee = ", totalexactee)
    return totalee, totalexactee
    
    
def genTrajLearnPred(nq, d, learningT1, learnDelT, hundredp, trajNum, circ, delNNT, nshots, seed, \
                     lightCone=0, deltaLDim=0, ifsave=0, nnt1=[], NTest=[]):
    #print("NTest = ", NTest)
    #d=4; learningT=d;p=0.1;circ=0; delNNT=500;nshots=5000;lightCone=0;deltaLDim=0;
    
    ### Args:
    # nq = number of qubits
    # d = depth of the circuit
    # learningT1 = the initial time for obtaining S_Q(t)
    # learnDelT = steps in the time vector of S_Q(t)
    # hundredp = p*100; p= measurement rate
    # trajNum = Number of quantum trajectories per each circuit. 
    # circ = circuit data including the unitary matrices and the measurement locations
    # delNNT = steps in the number of samples of the input data for the neural networks. 
    # nshots = number of shots of quantum trajectories used for learning
    # seed = the random seed for creating circuit configurations
    # lightCone = Boolean whether or not use the light cone data
    # deltaLDim = offset of the light cone box if lightCone==True
    # ifsave = Boolean whether or not save the circuit configurations
    
    p = hundredp/100;
    #print("p = ", p)
    if circ=="None" or circ==0:
        circ = genCircConfig(nq, d, p, seed);
    
    learningT2 = d+1;
    if delNNT==[]:
        delNNT=.05*nshots;nnt1=.9*nshots;
        
    NofNTR = int(floor((nshots-nnt1)/(delNNT)))-1;
    learnT = np.arange(learningT1, learningT2, learnDelT);
    #print("learnT = ", learnT);
    lightCone=0; deltaLDim=0;
    middleInd=int(float(nq/2));
    
    aveEeVec = np.zeros(len(learnT));
    exactEE = np.zeros(len(learnT));    
    EERes = np.zeros((2, len(learnT)));

    aveEeZVec = np.zeros(len(learnT));
    exactEEZ = np.zeros(len(learnT));    
    EERes = np.zeros((2, len(learnT)));
    EEZRes = np.zeros((2, len(learnT)));
    
    eeVec = np.zeros(len(learnT)); 
    eeZVec = np.zeros(len(learnT));     
    svecAve = np.zeros((3, len(learnT))); 
    
    for trj in range(trajNum):   
        eeVec = np.zeros(len(learnT));
        for t in range(len(learnT)):
            timel = learnT[t];
            
            marrX, convKX, countsX = timeEvolveAncilla(nq, timel, p, "prod", circ, 2, "X", nshots);
            marrY, convKY, countsY = timeEvolveAncilla(nq, timel, p, "prod", circ, 2, "Y", nshots);
            marrZ, convKZ, countsZ = timeEvolveAncilla(nq, timel, p, "prod", circ, 2, "Z", nshots);
            
            state, rhoAncilla, exactEE[t], measureArr, exactEEZ[t] = timeEvolveAncilla(nq, timel, p, "prod", circ, 2, "None", nshots);  
            # We use the test quantum trajectory produced in circuit X, and use it in the second two circuits.
            #print("shape convKX = ", np.shape(convKX))
            #print("convKX = ", convKX[:20])
            
            modelx, sxvec, NTest, testMeasX = learning(nq, timel, p, marrX, convKX, countsX, NofNTR, nshots, \
                             delNNT, nnt1, lightCone, deltaLDim, NTest);
            modely, syvec, NTest, testMeasY = learning(nq, timel, p, marrZ, convKZ, countsZ, NofNTR, nshots, \
                             delNNT, nnt1, lightCone, deltaLDim, NTest, testMeasX);
            modelz, szvec, NTest, testMeasZ = learning(nq, timel, p, marrY, convKY, countsY, NofNTR, nshots, \
                             delNNT, nnt1, lightCone, deltaLDim, NTest, testMeasX);   
                
            svec=[sxvec, syvec, szvec];
            
            svecAve[0, t] = np.mean(sxvec); 
            svecAve[1, t] = np.mean(syvec);
            svecAve[2, t] = np.mean(szvec);
            
            ee=0;
            eeZ=0;
            for i in range(NTest):             
                #print("inside for NTest")
                sx=sxvec[0, i]
                sy=syvec[0, i]
                sz=szvec[0, i]
        
                method="TNC"; 
                physden, tempresx, tempres=maxLikelihoodDen(sx, sy, sz, method)        
                #print("den = ", physden)
                if np.isnan(physden[0, 0]) or \
                np.isnan(physden[1, 0]) or np.isnan(physden[0, 1]) or \
                np.isnan(physden[1, 1]):            
                    #print("sx = ", sx, ", sy = ", sy, ", sz = ", sz)
                    #print("method =  nelder-mead")
            
                    method="nelder-mead"
                    physden, tempresx, tempres=maxLikelihoodDen(sx, sy, sz, method)
                    #print("physden = ", physden)
                    if np.isnan(physden[0, 0]) or \
                    np.isnan(physden[1, 0]) or np.isnan(physden[0, 1]) or \
                    np.isnan(physden[1, 1]):
                        method="SLSQP";
                        physden, tempresx, tempres=maxLikelihoodDen(sx, sy, sz, method);
                        if np.isnan(physden[0, 0]) or \
                        np.isnan(physden[1, 0]) or np.isnan(physden[0, 1]) or \
                        np.isnan(physden[1, 1]):
                        #print("continue")
                            continue
            
                partialEE = -np.trace(np.multiply(logm(physden),physden));
                eigval = eig(physden)[0];
                if eigval[0]==0:
                    eigval[0]=1e-16;
                if eigval[1]==0:
                    eigval[1]=1e-16;
        
                partialee = -np.sum(np.dot(np.log2(eigval), eigval));
                if np.isnan(partialee):
                    continue
                    
                ee = ee+np.real(partialee/(NTest));
                #print("partialee/NTest = ", partialee/NTest);                
                rvecNormZ = abs(sz)
                eigvalsDenZ = [1/2-rvecNormZ/2, 1/2+rvecNormZ/2];        
                rhoAncillaZ = [[1/2*(1+sz), 0], [0, 1/2*(1-sz)]];                
                renyiInd=1;
                if renyiInd==1:
                    try:
                        partialeeZ = -np.real(sum([eigvalsDenZ[i]*np.log2((eigvalsDenZ[i])) for i in range(2)]))                
                    except y:
                        if isa(y, DomainError):
                            println("domainError")
                            println("eigvalsDen = ", eigvalsDen)
                            println("ancillaDenMat = ", ancillaDenMat)                    
                        else:
                            partialeeZ = 1/(1-renyiInd) * np.log2(np.trace(matrix_power(rhoAncillaZ, renyiInd)))
                            partialeeZ = np.real(partialeeZ) 
                        
                    if partialeeZ==1:
                        partialeeZ=1-1e-8;
                eeZ = eeZ+np.real(partialeeZ/(NTest));
                

            eeVec[t] = ee;            
            eeZVec[t] = eeZ;            
            #print("eeVec inside test = ", eeVec);
            #print("eeVecZ inside test = ", eeZVec);            
            aveEeVec[t] = aveEeVec[t]+eeVec[t];
            aveEeZVec[t] = aveEeZVec[t]+eeZVec[t];            
        #print("aveEeZVec = ",aveEeZVec);
    
    aveEeVec=[aveEeVec[i]/trajNum for i in range(len(aveEeVec))];
    aveEeZVec=[aveEeZVec[i]/trajNum for i in range(len(aveEeZVec))];    
    
    print("aveEeVec after = ",aveEeVec);
    print("aveEeZVec after = ",aveEeZVec);    
            #print("eeVec inside t = ", eeVec);
    print("exact EE = ", np.real(exactEE));
    print("exact EEZ = ", np.real(exactEEZ));    
    
    EERes[0, 0:len(learnT)] = aveEeVec[0:len(learnT)];
    EERes[1, 0:len(learnT)] = exactEE[0:len(learnT)];

    EEZRes[0, 0:len(learnT)] = aveEeZVec[0:len(learnT)];
    EEZRes[1, 0:len(learnT)] = exactEEZ[0:len(learnT)];
    
    print("EERes = ", EERes);
    print("EEZRes = ", EEZRes);
    #seconds = time.time()
    #print("Seconds since epoch =", seconds)
    
    df = pd.DataFrame(EERes);
    #dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    #print("time =", dt_string[-8:])

    #fileOpen = open("HaarEERes-nq{}-p{}-ti{}-tf{}-delT{}-lightCone{}-nshots{}-seed{}.csv".format(nq, \
    #    p, learningT1, learningT2, learnDelT, lightCone, nshots, seed, dt_string[-8:]), "a")
    #s = fileOpen.read()
    #print("fileOpen = ", fileOpen)
                
    #fileOpen.write(EERes)
    with open("HaarEERes-nq{}-p{}-ti{}-tf{}-delT{}-lightCone{}-nshots{}-seed{}.csv".format(nq, \
        p, learningT1, learningT2, learnDelT, lightCone, nshots, seed), 'a') as f:
        csv.writer(f, delimiter=' ').writerows(EERes)    
    
    with open("HaarSpinRes-nq{}-p{}-ti{}-tf{}-delT{}-lightCone{}-nshots{}-seed{}.csv".format(nq, \
        p, learningT1, learningT2, learnDelT, lightCone, nshots, seed), 'a') as f:
        csv.writer(f, delimiter=' ').writerows(svecAve);
    
    #df.to_csv("HaarEERes-nq{}-p{}-ti{}-tf{}-delT{}-lightCone{}-nshots{}-seed{}-time{}.csv".format(nq, \
    #    p, learningT1, learningT2, learnDelT, lightCone, nshots, seed, dt_string[-8:]))
        
    return aveEeVec, aveEeZVec, exactEE, exactEEZ




def genTrajLearnPredArgs(argv):
#    (nq, d, p, circ, delNNT, nshots, lightCone=0, deltaLDim=0):

#genTrajLearnPred(nq, d, learningT1, learnDelT, hundredp, trajNum, circ, delNNT, nshots, seed, \
#                     lightCone=0, deltaLDim=0, ifsave=0, nnt1=[], NTest=[]):
    print("learningARGS")    
    print('Argument List:', str(argv))
    opts, args = getopt.getopt(argv, "hi:o:")

    nq = int(args[0]);
    d = int(args[1]);
    p = float(args[2]);
    learningT1 = int(args[3]);
    learnDelT = int(args[4]);    
    #hundredp = float(args[5]);    
    #p = hundredp/100;
    trajNum = int(args[5]);    
    circ = int(args[6]);
    if circ==0:
        circ = "None"; 
        
    delNNT=int(args[7]); nshots=int(args[8]); learningT2 = d+1;
    learnT = np.arange(learningT1, learningT2, learnDelT);
    
    eeVec = np.zeros(len(learnT));
    
    seed = float(args[9]);
    lightCone=int(args[10]); deltaLDim=int(args[11]);    
    ifsave=int(args[12]);    
    nnt1=int(args[13]);    
    NTest=int(args[14]);
    
    if circ=="None":
        if seed==0:
            seed = random.random();
            seed = round(seed, 5)                        
        circ = genCircConfig(nq, d, p, seed, ifsave);
    
    #nshots=1000;
    #(nq, d, learningT1, learnDelT, hundredp, trajNum, circ, delNNT, nshots, seed=0, \
    #                 lightCone=0, deltaLDim=0, ifsave=0):
    
    #delNNT=.05*nshots;nnt1=.9*nshots;
    #NofNTR = int(floor((nshots-nnt1)/(delNNT)))-1;
    
    NofNTR=1;
    print("NofNTR = ", NofNTR);
    print("nshots = ", nshots);
    
    lightCone=0; deltaLDim=0;
    middleInd=int(float(nq/2));
    svec=[0, 0, 0];
    aveEeVec = np.zeros(len(learnT));
    aveEeZVec = np.zeros(len(learnT));    
    exactEE = np.zeros(len(learnT));
    exactEEZ = np.zeros(len(learnT));
    
    #print("eeVec = ", eeVec)
    EERes = np.zeros((2, len(learnT)));
    EEZRes = np.zeros((2, len(learnT)));
    svecAve = np.zeros((3, len(learnT))); 
    
    #NTest = 200;
    for trj in range(trajNum):   
        eeVec = np.zeros(len(learnT));
        eeZVec = np.zeros(len(learnT));
        for t in range(len(learnT)):    #if True:         
            #print("t = ", t)    
            timel = learnT[t];
                                
            marrX, convKX, countsX = timeEvolveAncilla(nq, timel, p, "prod", circ, 2, "X", nshots);
            marrY, convKY, countsY = timeEvolveAncilla(nq, timel, p, "prod", circ, 2, "Y", nshots);
            marrZ, convKZ, countsZ = timeEvolveAncilla(nq, timel, p, "prod", circ, 2, "Z", nshots);
            
            state, rhoAncilla, exactEE[t], measureArr, exactEEZ[t] = timeEvolveAncilla(nq, timel, p, "prod", circ, 2, \
                                                                          "None", nshots);                    
            # We use the test quantum trajectory produced in circuit X, and use it in the second two circuits.
            #print("shape convKX = ", np.shape(convKX))

            modelx, sxvec, NTest, testMeasX = learning(nq, timel, p, marrX, convKX, countsX, NofNTR, nshots, \
                             delNNT, nnt1, lightCone, deltaLDim, NTest);
            modely, syvec, NTest, testMeasY = learning(nq, timel, p, marrZ, convKZ, countsZ, NofNTR, nshots, \
                             delNNT, nnt1, lightCone, deltaLDim, NTest, testMeasX);
            modelz, szvec, NTest, testMeasZ = learning(nq, timel, p, marrY, convKY, countsY, NofNTR, nshots, \
                             delNNT, nnt1, lightCone, deltaLDim, NTest, testMeasX);
            
            svecAve[0, t] = np.mean(sxvec);
            svecAve[1, t] = np.mean(syvec);
            svecAve[2, t] = np.mean(szvec);
                
            svec=[sxvec, syvec, szvec];

            ee=0;
            eeZ=0;
            
            for i in range(NTest):             
                #print("inside for NTest")
                sx=sxvec[0, i];
                sy=syvec[0, i];
                sz=szvec[0, i];
        
                method="TNC"; 
                physden, tempresx, tempres=maxLikelihoodDen(sx, sy, sz, method)        
                if np.isnan(physden[0, 0]) or \
                np.isnan(physden[1, 0]) or np.isnan(physden[0, 1]) or \
                np.isnan(physden[1, 1]):            
                    method="nelder-mead"
                    physden, tempresx, tempres=maxLikelihoodDen(sx, sy, sz, method)
                    #print("physden = ", physden)
                    if np.isnan(physden[0, 0]) or \
                    np.isnan(physden[1, 0]) or np.isnan(physden[0, 1]) or \
                    np.isnan(physden[1, 1]):
                        method="SLSQP";
                        physden, tempresx, tempres=maxLikelihoodDen(sx, sy, sz, method);
                        if np.isnan(physden[0, 0]) or \
                        np.isnan(physden[1, 0]) or np.isnan(physden[0, 1]) or \
                        np.isnan(physden[1, 1]):
                        #print("continue")
                            continue

                            
                rhoAncillaZ = [[1/2*(1+sz), 0], [0, 1/2*(1-sz)]];                            
                rvecNormZ = np.linalg.norm(sz);        
                eps = (1+1j)*1e-16;
                #ancillaDenMat = traceMatExceptLast(rho)+eps*Matrix(I, 2, 2)
                #println("ancillaDenMat = ", ancillaDenMat)
                #eigvalsDen = [1/2-rvecNorm/2, 1/2+rvecNorm/2];        
        
                partialEE = -np.trace(np.multiply(logm(physden),physden))
                eigval = eig(physden)[0]
                eigvalZ = [1/2-sz/2, 1/2+sz/2];
                
                if eigval[0]==0:
                    eigval[0]=1e-16;
                if eigval[1]==0:
                    eigval[1]=1e-16;

                if eigvalZ[0]==0:
                    eigvalZ[0]=1e-16;
                if eigvalZ[1]==0:
                    eigvalZ[1]=1e-16;
                    
                partialee = -np.sum(np.dot(np.log2(eigval), eigval))                    
                partialeeZ = -np.sum(np.dot(np.log2(eigvalZ), eigvalZ))
                        
                if np.isnan(partialee):
                    continue
                if np.isnan(partialeeZ):
                    continue
            
                ee = ee+np.real(partialee/(NTest));
                eeZ = eeZ+np.real(partialeeZ/(NTest));
                #print("ee = ", ee);
                #print("partialee/NTest = ", partialee/NTest)

            eeVec[t] = ee;
            eeZVec[t] = eeZ;
            #print("eeVec inside test = ", eeVec);                
            aveEeVec[t] = aveEeVec[t]+eeVec[t];
            aveEeZVec[t] = aveEeZVec[t]+eeZVec[t];            
        print("aveEeVec = ",aveEeVec)        
            
    aveEeVec=[aveEeVec[i]/trajNum for i in range(len(aveEeVec))];
    aveEeZVec=[aveEeZVec[i]/trajNum for i in range(len(aveEeZVec))];    
    
    aveEeVec = ["{:.8e}".format(aveEeVec[i]) for i in range(len(aveEeVec))]
    aveEeZVec = ["{:.8e}".format(aveEeZVec[i]) for i in range(len(aveEeZVec))]    
    
    exactEE = ["{:.8e}".format(exactEE[i]) for i in range(len(exactEE))]
    exactEEZ = ["{:.8e}".format(exactEEZ[i]) for i in range(len(exactEEZ))]    

    print("aveEeVec after = ",aveEeVec)       
            #print("eeVec inside t = ", eeVec)
    print("exact EE = ", np.real(exactEE))
    
    EERes[0, 0:len(learnT)] = aveEeVec[0:len(learnT)];
    EERes[1, 0:len(learnT)] = exactEE[0:len(learnT)];
    
    EEZRes[0, 0:len(learnT)] = aveEeZVec[0:len(learnT)];
    EEZRes[1, 0:len(learnT)] = exactEEZ[0:len(learnT)];
    
    print("EEZRes = ", EEZRes)
    
    df = pd.DataFrame(EERes);
    with open("HaarEERes-nq{}-p{}-ti{}-tf{}-delT{}-lightCone{}-nshots{}-seed{}.csv".format(nq, \
        p, learningT1, learningT2, learnDelT, lightCone, nshots, seed), 'a') as f:
        csv.writer(f, delimiter=' ').writerows(EERes)    

    df = pd.DataFrame(EEZRes);
    with open("HaarEEZRes-nq{}-p{}-ti{}-tf{}-delT{}-lightCone{}-nshots{}-seed{}.csv".format(nq, \
        p, learningT1, learningT2, learnDelT, lightCone, nshots, seed), 'a') as f:
        csv.writer(f, delimiter=' ').writerows(EEZRes)    
        
    with open("HaarSpinRes-nq{}-p{}-ti{}-tf{}-delT{}-lightCone{}-nshots{}-seed{}.csv".format(nq, \
        p, learningT1, learningT2, learnDelT, lightCone, nshots, seed), 'a') as f:
        csv.writer(f, delimiter=' ').writerows(svecAve);
        
        
    return circ, svec, eeVec, eeZVec







def genTrajExactPred(nq, d, T1, DelT, hundredp, trajNum, circ, nshots, seed, ifsave=0):
    
    #d=4; learningT=d;p=0.1;circ=0; delNNT=500;nshots=5000;lightCone=0;deltaLDim=0;
    
    ### Args:
    # nq = number of qubits
    # trajNum = Number of quantum trajectories per each circuit. 

    
    p = hundredp/100;
    #print("p = ", p)
    if circ=="None" or circ==0:
        circ = genCircConfig(nq, d, p, seed);
    
    T2 = d+1;
    delNNT=.05*nshots;nnt1=.9*nshots;
    NofNTR = int(floor((nshots-nnt1)/(delNNT)))-1;
    learnT = np.arange(T1, T2, DelT);
    #print("learnT = ", learnT);
    lightCone=0; deltaLDim=0;
    middleInd=int(float(nq/2));
    

    exactEE = np.zeros(len(learnT));    
    EERes = np.zeros((1, len(learnT)));

    exactEEz = np.zeros(len(learnT));    
    EEResZ = np.zeros((1, len(learnT)));
    
    for trj in range(trajNum):   
        eeVec = np.zeros(len(learnT));
        for t in range(len(learnT)):    #if True:         
            #print("t = ", t)    
            timel = learnT[t];
            
        
            state, rhoAncilla, tempEE, measureArr, tempEEz = timeEvolveAncilla(nq, timel, p, "prod", circ, 2, "None", nshots);        
            exactEE[t] = exactEE[t] + tempEE; 
            exactEEz[t] = exactEEz[t] + tempEEz; 
        print("exactEE = ",exactEE)        
        print("exactEEz = ",exactEEz)                    
        
    exactEE=[exactEE[i]/trajNum for i in range(len(exactEE))];
    exactEEz=[exactEEz[i]/trajNum for i in range(len(exactEEz))];
    
    print("exact EE = ", np.real(exactEE))
    print("exact EEz = ", np.real(exactEEz))    

    #return circ, svec, eeVec
        
    EERes[0, 0:len(learnT)] = exactEE[0:len(learnT)];
    EEResZ[0, 0:len(learnT)] = exactEEz[0:len(learnT)];    
    #EERes[1, 0:len(learnT)] = exactEE[0:len(learnT)];
    
    #print("len(EERes) = ", len(EERes))
    #seconds = time.time()
    #print("Seconds since epoch =", seconds)	
    
    df = pd.DataFrame(EERes);
    with open("HaarExactEERes-nq{}-p{}-ti{}-tf{}-delT{}-nshots{}-seed{}.csv".format(nq, \
        p, T1, T2, DelT, nshots, seed), 'a') as f:
        csv.writer(f, delimiter=' ').writerows(EERes)   
        
    df = pd.DataFrame(EERes);        
    with open("HaarExactEEResZ-nq{}-p{}-ti{}-tf{}-delT{}-nshots{}-seed{}.csv".format(nq, \
        p, T1, T2, DelT, nshots, seed), 'a') as f:
        csv.writer(f, delimiter=' ').writerows(EEResZ)   
    
    
    return exactEE, exactEEz



def genTrajExactPredParallel(nrep, nq, d, T1, DelT, hundredp, trajNum, circ, nshots, seed, \
                             lightCone=0, deltaLDim=0, ifsave=0):
    T2=d+1;
    learnT = np.arange(T1, learningT2, learnDelT);
    #print("learnT = ", learnT)
    totalee=np.zeros(len(learnT));
    totalexactee=np.zeros(len(learnT));
        
    for i in range(nrep):
        print("i in nrep = ", i)
        #seed = seed+i                
        print("seed in parallel = ", seed+i)
        eeVec, tempexactee = genTrajLearnPred(nq, d, learningT1, learnDelT, hundredp, trajNum, 
                            circ, delNNT, nshots, seed, lightCone, deltaLDim, ifsave, delNNT, nnt1, NTest);                              
        #print("totalee = ", totalee);
        #print("eeVec = ", eeVec);
        totalee = np.add(eeVec, totalee);
        totalexactee = np.add(tempexactee, totalexactee);    
    
    print("totalee = ", totalee)
    print("totalexactee = ", totalexactee)
    return totalee, totalexactee

def genTrajExactPredArgs(argv):
#    (nq, d, p, circ, delNNT, nshots, lightCone=0, deltaLDim=0):
    print("learningARGS")    
    print('Argument List:', str(argv))
    opts, args = getopt.getopt(argv, "hi:o:")

    nq = int(args[0]);
    d = int(args[1]);
    p = float(args[2]);
    T1 = int(args[3]);
    DelT = int(args[4]);    
    #hundredp = float(args[5]);    
    #p = hundredp/100;
    trajNum = int(args[5]);    
    circ = int(args[6]);
    if circ==0:
        circ = "None"; 
        
    nshots=int(args[7]);
    T2 = d+1;
    learnT = np.arange(T1, T2, DelT);
    
    eeVec = np.zeros(len(learnT))
    
    seed = float(args[8]);
    ifsave=int(args[9]);    
    
    if circ=="None":
        if seed==0:
            seed = random.random();
            seed = round(seed, 5)                        
        circ = genCircConfig(nq, d, p, seed, ifsave);

    print("nshots = ", nshots)
    
    for trj in range(trajNum):   
        eeVec = np.zeros(len(learnT));
        for t in range(len(learnT)):    #if True:         
            #print("t = ", t)    
            timel = learnT[t];
            
        
            state, rhoAncilla, tempEE, measureArr, tempEEz = timeEvolveAncilla(nq, timel, p, "prod", circ, 2, "None", nshots);        
            state, rhoAncilla, tempEE, measureArr, tempEEz = timeEvolveAncilla(nq, timel, p, "prod", circ, 2, "None", nshots);                    

            exactEE[t] = exactEE[t] + tempEE; 
            exactEEz[t] = exactEEz[t] + tempEEz;
            
        print("exactEE = ",exactEE)        
        print("exactEEz = ",exactEEz)                
            
    exactEE=[exactEE[i]/trajNum for i in range(len(exactEE))];
    exactEEz=[exactEEz[i]/trajNum for i in range(len(exactEEz))];    
    print("exact EE = ", np.real(exactEE))
    print("exact EEz = ", np.real(exactEEz))
    
    #return circ, svec, eeVec
        
    EERes[0, 0:len(learnT)] = exactEE[0:len(learnT)];
    EEResZ[0, 0:len(learnT)] = exactEEz[0:len(learnT)];
    #EERes[1, 0:len(learnT)] = exactEE[0:len(learnT)];
    
    print("len(EERes) = ", len(EERes))
    #seconds = time.time()
    #print("Seconds since epoch =", seconds)
    
    df = pd.DataFrame(EERes);
    with open("HaarExactEERes-nq{}-p{}-ti{}-tf{}-delT{}-nshots{}-seed{}.csv".format(nq, \
        p, T1, T2, DelT, nshots, seed), 'a') as f:
        csv.writer(f, delimiter=' ').writerows(EERes)   
    
    df = pd.DataFrame(EEResZ);
    with open("HaarExactEEResZ-nq{}-p{}-ti{}-tf{}-delT{}-nshots{}-seed{}.csv".format(nq, \
        p, T1, T2, DelT, nshots, seed), 'a') as f:
        csv.writer(f, delimiter=' ').writerows(EEResZ)   

        
    return exactEE, exactEEz





if __name__ == '__main__':  
    nprocess = int(float(sys.argv[-1]))
    print("nprocess + 2 = ", nprocess+2)
    pool = Pool(processes=nprocess)
    """
    cvec = []
    if sys.argv[-2]== "False":
        print("cvec = False.")
        
    else:
        print("else cvec==False")
        cvecstr = sys.argv[-2].split(',')
    
        for i in cvecstr:
            print("i = ", i)
            cvec.append(int(float(i)))
    print("main cvec = ", cvec)            
    """  
    
    argsarr = [];
    for i in range(nprocess):
        argsarr.append(sys.argv[1:-1])
        #argsarr.append(sys.argv[1:-2])
        #argsarr[-1].append(str(cvec[i]))
    
    print("argsarr = ", argsarr)
    pool.map(genTrajLearnPredArgs, argsarr)    
   

    
"""

if __name__ == "__main__":  # confirms that the code is under main function
    names = ['America', 'Europe', 'Africa']
    procs = []
    proc = Process(target=print_func)  # instantiating without any argument
    procs.append(proc)
    proc.start()

    # instantiating process with arguments
    for name in names:
        # print(name)
        proc = Process(target=print_func, args=(name,))
        procs.append(proc)
        proc.start()

    # complete the processes
    for proc in procs:
        proc.join()
"""

#print(tf.__version__)



def createMeasureSHFiles(nq, d, T1, DelT, hundredp, trajNum, circ, nshots, delNNT, seedmin, \
    seedmax, deltaseed, fileLabel, account, hour, minutes, nproc, \
    directory, nnt1, NTest, ifsave=0, circInd=False):
    
    p=hundredp/100;
    DecimalP = str(p).replace('0.','');
    prob = "0p{}".format(DecimalP);
    rowc=0;
    os.chdir(directory);
    cwdir = os.getcwd();
    print(cwdir);
    
    seedvec = np.linspace(seedmin, seedmax, nseed);  #   np.arange(seedmin, seedmax, deltaseed);
    seedvec = [np.round(seed, 2) for seed in seedvec];
    #print("seedvec = ", seedvec);
    filepath = os.path.join('/Users/hosseindehghani/Desktop/Hafezi/Codes/Haar/', 'haar.sh') 


    for s in range(len(seedvec)):
        seed = seedvec[s];
        print("seed = ", seed)
        #n = int(seed);
        if seed!=0:
            try:
                #print('{}/{}P{}nq{}d{}s{}.sh'.format(directory, fileLabel, prob, nq, d, seed))
                shutil.copy2('/Users/hosseindehghani/Desktop/Hafezi/Codes/Haar/haar.sh',
                    '{}/{}P{}nq{}d{}s{}.sh'.format(directory, fileLabel, prob, nq, d, seed));
            except shutil.SameFileError:
                print("Source and destination represents the same file.")
            except PermissionError:
                print("Permission denied.")
            except:
                print("Error occurred while copying file.")
                    

    count = 0;
    filePattern = '*.sh';
    for path, dirs, files in os.walk(os.path.abspath(directory)):        
        for filename in fnmatch.filter(files, filePattern):            
            #print("fileName = ", filename)
            if '{}P{}nq{}d{}s'.format(fileLabel, prob, nq, d) in filename:
                seedSH = filename.replace('{}P{}nq{}d{}s'.format(fileLabel, prob, nq, d),'')
                seed = seedSH.replace('.sh','')
                #print("seed = ", seed)
            else:
                continue                                
            filepath = os.path.join(path, filename);                        
            
            with open(filepath) as f:
                s = f.read();
                find1 = "#SBATCH -t 8:00:00"
                find2 = "python HaarRandomPyHPC.py 8 8 .1 1 1 1 0 100 5000 1.00 0 0 1 20"
        
                replaceTime = "#SBATCH -t {}:{}:00".format(hour, minutes)

                replace2 = "python HaarRandomPyHPC.py {} {} {} {} {} {} {} {} {} {} 0 0 {} {} {} 20".format(nq, \
                    d, p, T1, DelT, trajNum, circ, delNNT, nshots, seed, ifsave, nnt1, NTest);

                learningT2 = d+1;
                stringMeasure = s.replace(find1, replaceTime);
                stringMeasure = s.replace(find2, replace2);
                #print("s modified = ", s);
                #print("filepath = ", filepath);
                with open(filepath, "w") as f:
                    #print("s2 = ", stringMeasure);
                    f.write(stringMeasure);
                
                
            
                
    return 1
        
    
