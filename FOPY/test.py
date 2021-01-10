import numpy as np
import math
import numpy.linalg as LA
import matplotlib.pyplot as plt

def det(i):
    ans = i[0,0]*i[1,1]*i[2,2] + i[0,1]*i[1,2]*i[2,0]+i[0,2]*i[1,0]*i[2,1] - i[0,2]*i[1,1]*i[2,0] - i[0,0]*i[1,2]*i[2,1] - i[0,1]*i[1,0]*i[2,2]
    return ans

def def_re(i):
    print('mathans:' + str(det(i)))
    print('pyans  :' + str(LA.det(i)))

a = np.array([[1,2,3],
              [4,5,6],
              [7,8,9]])

b = np.array([[1,1,3],
              [4,78,6],
              [7,8,-4]])

print(a,type(a))



def_re(b)
def_re(a@b)
def_re(a)


print(LA.det(a))
print("#####")
print(det(a))

b = np.array([[1,2,3]])

print(b,type(b))

print(a@b.T)
# print(a@b) Undifined
