import numpy as np
from hw4_seccure_comput import *

a_vec, x_vec = [], []
num = 0
while num != NUMBER:            # NUMBER=2,
    a_vec.append(np.random.randint(0,4))
    x_vec.append(np.random.randint(0,4))
    num += 1

a1 = a_vec[0]
a2 = a_vec[1]
x1 = x_vec[0]
x2 = x_vec[1]

print(f"a vector = [a1 = {a1}, a2 = {a2}]")
print(f"x vector = [x1 = {x1}, x2 = {x2}]")



dealer = Dealer()
circuit_A, u_A, v_A, w_A, r_A, mult_gates_A = dealer.RandA()
circuit_B, u_B, v_B, w_B, r_B, mult_gates_B = dealer.RandB()

Alice = Alice(circuit=circuit_A, x=x_vec, u_A=u_A, v_A=v_A, w_A=w_A, r_A=r_A, number_mult_gates=mult_gates_A)
Bob = Bob(circuit=circuit_B, y=a_vec, u_B=u_B, v_B=v_B, w_B=w_B, r_B=r_B, number_mult_gates=mult_gates_B)

d = Alice.ReceiveR_BAndSendD(r_B=Bob.SendR_B())
Bob.ReceiveD(d=d)

d = Bob.ReceiveR_AAndSendD(r_A = Alice.SendR_A())
Alice.ReceiveD(d=d)

while Alice.has_output() == False:
  MULTS = Alice.send()
  Bob.receive(MULTS=MULTS)
  z_B = Bob.send()
  Alice.receive(z_B=z_B)

z = Alice.return_output()

if a1*x1+a2*x2 >= 4 and z == 0:
    print(f"z = {z} error.")
elif a1*x1+a2*x2 < 4 and z == 1:
    print(f"z = {z} error.")
else:
    print(f"z = {z} passed.")
