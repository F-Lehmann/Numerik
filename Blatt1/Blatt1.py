import numpy as np
import time

print("a)")
#Das ist ein Kommentar

print("b)")
b = np.random.randint(100, size=(1, 4))
print(b)

print("c)")
c = np.random.randint(100, size=(5,1))
print(c)

print("d)")
d = np.zeros((3,4),dtype=int)
print(d)

print("e)")
e = np.random.randint(100, size=(4,3))
print(e)
print(e[1])

print("f)")
f = np.random.randint(100, size=(4,4))
print(f)
print(f[:,2])

print("g)")
g = np.random.randint(100, size=(2,5))
print(g)
print(g.transpose())

print("h)")
h1 = np.random.randint(100, size=(4,4))
h2 = np.random.randint(100, size=(4,4))
print(h1)
print(h2)
print(h1*h2)
print(np.matmul(h1,h2))

print("i)")
i1 = np.random.randint(100, size=(4,4))
i2 = np.random.randint(100, size=(4,4))
print(i1)
print(i2)
print(np.vstack((i1,i2)))
print(np.hstack((i1,i2)))

print("j)")
j = np.random.randint(10, size=(2,5))
print(j)
print(j.shape)

print("k)")
k = np.random.randint(100, size=(8,7))
print(k)
print(k.reshape((14,4)))

print("l)")
l = np.random.randint(100, size=(3,1))
print(l)
print(np.tile(l,10000))

print("m)")
m = np.random.randint(low= -100, high = 100, size=(3,4))
print(m)
print(m.clip(min=0))

print("n)")
n = np.arange(100, step=7)
print(n)

print("o)")
o = np.random.randint(low=1, high=100, size=100)
print(o)
o[1::2][:]=0
print(o)

print("p)")
p = np.random.randint(100, size=100)
print(p)
p = p[::2]
print(p)

print("q)")
#Aufgabenstellung unklar 100x3 oder 1000x3?
q1 = np.random.randint(100, size=(1000,3))
q2 = np.random.randint(100, size=(1000,3))
print(q1)
print(q2) 
#mit schleife
t0 = time.time()
for v1 in q1:
    for v2 in q2:
        sp = 0
        for i in range(len(v1)):
            sp += v1[i] * v2[i]
t1 = time.time() 
#ohne schleife
t2 = time.time()
for v1 in q1:
    for v2 in q2:
        np.dot(v1,v2)
t3 = time.time() 
print("mit Schleife: {0}".format(t1-t0))
print("ohne Schleife: {0}".format(t3-t2))
print("Oh Wunder, ohne schleife ist schneller")

print("r)")
#nicht jede zufällige Matrix ist invertierbar, diese bleiben stattdessen unverändert.
r = np.random.randint(100, size=(1000,4))
print(r)
r = np.array([(1/(m[0]*m[3]-m[1]*m[2])*np.array([m[3],-m[1],-m[2],m[0]])) if (m[0]*m[3]-m[1]*m[2])!=0 else m for m in r])
print(r)
