import numpy as np
from hw3 import *


a_vec, x_vec = [], []
num = 0
while num != NUMBER:
    a_vec.append(np.random.randint(0,2))
    x_vec.append(np.random.randint(0,2))
    num += 1

# calc a1, a2, x1, x_2
a1 = a_vec[0]*2 + a_vec[1]
a2 = a_vec[2]*2 + a_vec[3]

x1 = x_vec[0]*2 + x_vec[1]
x2 = x_vec[2]*2 + x_vec[3]

print(f"a vector = [a1 = {a1}, a2 = {a2}]")
print(f"x vector = [x1 = {x1}, x2 = {x2}]")

dealer = Dealer()

u_A, v_A, w_A = dealer.rand_a()
u_B, v_B, w_B = dealer.rand_b()
circuit = dealer.circuit

Alice = Alice(x_vec, circuit, u_A, v_A, w_A)
Bob = Bob(a_vec, circuit, u_B, v_B, w_B)


Bob.receive_alice_share(Alice.send_bob_share())
Alice.receive_bob_share(Bob.send_alice_share())

while Alice.has_output() == False:
    ands = Alice.send()
    Bob.receive(ands)
    z_B = Bob.send()
    Alice.receive(z_B)

z = Alice.output()
if a1*x1+a2*x2 >= 4 and z == 0:
    print(f"z = {z} error.")
elif a1*x1+a2*x2 < 4 and z == 1:
    print(f"z = {z} error.")
else:
    print(f"z = {z} passed.")

