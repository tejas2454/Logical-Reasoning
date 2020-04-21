import matplotlib.pyplot as plt
from copy import deepcopy
import csv
import pandas as pd
import numpy as np
import sys

with open(sys.argv[2]) as fin, open('sample_input2.csv', 'w') as fout:
    o=csv.writer(fout)
    for line in fin:
        o.writerow(line.split())
p=open(sys.argv[1])
f=pd.DataFrame(p)

S=int(f[0][0],10)
n=int(f[0][1],10)
m=int(f[0][2],10)
t=float(f[0][3])


def interfact(KB,fact):
    if fact not in KB:
            KB.append(fact)
            flag = True
# Euclidean Distance Caculator
def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)

#converting to list

def zipped(input_2,n,m):
    a=[]
    for i in range(n+m+1):
        ola=input_2[i].values
        a.append(ola)
        i+=1
    X=np.array(a).T
    return(X)


input_2=pd.read_csv('sample_input2.csv',header=None)
X=zipped(input_2,n,m)

def neighbour(n,m,t,C,KB):
    for i in range(len(C)):
        for j in range(1,len(C)):
            if (C[i][n+m]-C[j][n+m]<t):
                #return (i,j)
                v=["NH",i,j]
                KB.append(v)          
    return(KB)

def disease(S,n,C,KB):
    for i in range(0,S):
        for j in range(0,n):
            if C[i][j]>0.7:
                KB.append(["D",i+1,j+1])              
    return(KB)

def habit_disease_rel(pf):
    habit_disease=[]
    for i in range(len(pf[4:])):
        l=pf[4:][i]
        list_k=l.split()
        for i in range(0, len(list_k)): 
            list_k[i] = int(list_k[i],10) 
        habit_disease.append(list_k)
    return(habit_disease)

def habit(S,n,C,KB):
    for i in range(0,S):
        for j in range(n,n+m):
            if C[i][j]>0.5:
                v=["H",i+1,j-1]
                KB.append(v)
                #q.append(j)
                #print(p)
                #print(q)     
      
    return(KB)


C=np.random.rand(S,n+m+1)

C_prev = np.zeros(C.shape)
ob_clusters = np.zeros(len(X))

# Error function distance between new centroids and old centroids
error = dist(C, C_prev, None)
#print(error)
#print(len(X))
# Loop will run till the error becomes zero
while error != 0:
    # Assigning each value to its closest cluster
    for i in range(len(X)):
        distances = dist(X[i], C)
        cluster = np.argmin(distances)
        ob_clusters[i] = cluster
    # Storing the old centroid values
    C_prev = deepcopy(C)
    # Finding the new centroids by taking the average value
    for i in range(S):
        points = [X[j] for j in range(len(X)) if ob_clusters[j] == i]
        C[i] = np.mean(points, axis=0)
    
    error = dist(C, C_prev, None)

np.savetxt('output.txt',C,delimiter=' ',fmt='%.2f')

print(np.around(C,decimals=2))

KB=[]
KB1=neighbour(n,m,t,C,KB)
lol=disease(S,n,C,KB)
p=open(sys.argv[1],'r')
pf=p.read()
pf=pf.split('\n')
habit_info=pf[4:]
habit_lst=habit_disease_rel(pf)
habit_lst=np.array(habit_lst)
shape=(m,n)
habit_lst=habit_lst.reshape(shape)
for i in range(m):
    for j in range(n):
        if habit_lst[i][j]==1:
            hd=["HLD", i+1,j+1]
            KB.append(hd)

lol1=habit(S,n,C,KB)


flag = True        
while flag:
    flag = False
    for A in KB:            
        ## I(s,d)
        if A[0] == 'D':
            interfact(KB,['D',A[1],A[2]])
            ## N(s,s1) and D(s1,d) 
        if A[0] == 'NH':
            A_d = [f for f in KB if f[0] == 'D']
            for idx_d in A_d:
                if A[2] == idx_d[1]:
                    interfact(KB,['D',A[1],idx_d[2]])

        ## I(s,h) and L(h,d)
        if A[0] == 'H':
            A_h = [h for h in KB if h[0] == 'HLD']
            for idx_h in A_h:
                if A[2] == idx_h[1]:
                    interfact(KB,['D',A[1],idx_h[2]])

new_KB=[i for i in KB if i[0]=='D']
new_KB.sort()


file_out=open('output.txt','a')
#np.savetxt(file_out,[0], delimiter=' ',fmt='%d')
for i in range(1,S+1,1):
    tempo=[]
    for j in new_KB:
        tempo = [j[2] for j in new_KB if j[1]==i]
    
    tempo = np.array(tempo)
    print(tempo)
    #print(type(tempo))
    np.savetxt(file_out, [tempo], delimiter=' ',fmt='%d')
file_out.close()


