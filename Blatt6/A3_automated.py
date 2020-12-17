import numpy as np
import array_to_latex as a2l
to_tex = lambda A : a2l.to_ltx(A, frmt = '{:.3f}', arraytype = 'pmatrix', mathform=True)

A = np.array([[-2,-2,-2],[-2,-1,-1],[1,0,-1]])
A1 = A
a1 = A1[:,0].reshape((3,1))
v1 = a1 + (np.sign(A1[0,0]) * np.linalg.norm(a1) * np.eye(3)[:,0]).reshape((3,1))
H1 = np.eye(3) - 2 * (v1@v1.T)/(v1.T@v1)

A2 = (H1@A1)[1:,1:]
a2 = A2[:,0].reshape((2,1))
v2 = a2 + (np.sign(A2[0,0]) * np.linalg.norm(a2) * np.eye(2)[:,0]).reshape((2,1))
H2 = np.eye(2) - 2 * (v2@v2.T)/(v2.T@v2)

A3 = (H2@A2)[1:,1:]
a3 = A3[:,0].reshape((1,1))
v3 = a3 + (np.sign(A3[0,0]) * np.linalg.norm(a3) * np.eye(1)[:,0]).reshape((1,1))
H3 = np.eye(1) - 2 * (v3@v3.T)/(v3.T@v3)

print('$')
print('A=')
to_tex(A)
print('\\\\')

print('A_1=')
to_tex(A1)
print('\\\\')

print('a_1=')
to_tex(a1)
print('\\\\')

print('||a_1||=\\sqrt{(-2)^2+(-2)^2+1^2}=\\sqrt{9}=3')
print('\\\\')

print('v_1=a_1+sign(A_{11})||a_1||e_1=')
to_tex(a1)
print('-3\\times')
to_tex(np.eye(3)[:,0].reshape((3,1)))
print('=')
to_tex(v1)
print('\\\\')

print('H_1=I-2\\cdot\\frac{v_1\\cdot v_1^T}{v_1^T\\cdot v_1}=')
to_tex(np.identity(3))
print('-\\frac{2}{30}\\cdot')
to_tex(v1)
print('\\cdot')
to_tex(v1.T)
print('=')
to_tex(H1)
print('\\\\')

print('A_2=')
to_tex(A2)
print('\\\\')

print('a_2=')
to_tex(a2)
print('\\\\')

print('||a_2||=\\sqrt{0.6^2+(-0.8^2}=\\sqrt{1}=1')
print('\\\\')

print('v_2=a_2+sign(A_{22})||a_2||e_2=')
to_tex(a2)
print('+1\\times')
to_tex(np.eye(2)[:,0].reshape((2,1)))
print('=')
to_tex(v2)
print('\\\\')

print('H_2=I-2\\cdot\\frac{v_2\\cdot v_2^T}{v_2^T\\cdot v_2}=')
to_tex(np.identity(2))
print('-\\frac{2}{80}\\cdot')
to_tex(v2)
print('\\cdot')
to_tex(v2.T)
print('=')
to_tex(H2)
print('\\\\')

print('A_3=')
to_tex(A3)
print('\\\\')

print('a_3=')
to_tex(a3)
print('\\\\')

print('||a_3||=\\sqrt{(-\\frac{2}{3})^2}=\\sqrt{\\frac{4}{9}}=\\frac{2}{3}')
print('\\\\')

print('v_3=a_3+sign(A_{33})||a_3||e_3=')
to_tex(a3)
print('-\\frac{2}{3}\\times')
to_tex(np.eye(1)[:,0].reshape((1,1)))
print('=')
to_tex(v3)
print('\\\\')

print('H_3=I-2\\cdot\\frac{v_3\\cdot v_3^T}{v_3^T\\cdot v_3}=')
to_tex(np.identity(1))
print('-\\frac{2}{144}\\cdot')
to_tex(v3)
print('\\cdot')
to_tex(v3.T)
print('=')
to_tex(H3)
print('\\\\')

id3 = np.identity(3)
id3[1:,1:] = H2
H2 = id3

id3 = np.identity(3)
id3[2:,2:] = H3
H3 = id3

R = H3@H2@H1@A
Q = H1@H2@H3

print('R=H_3H_2H_1A=')
to_tex(H3)
print('\\times')
to_tex(H2)
print('\\times')
to_tex(H1)
print('\\times')
to_tex(A)
print('=')
to_tex(R)
print('\\\\')

print('Q=H_1H_2H_3=')
to_tex(H1)
print('\\times')
to_tex(H2)
print('\\times')
to_tex(H3)
print('=')
to_tex(Q)
print('\\\\')

print('$')