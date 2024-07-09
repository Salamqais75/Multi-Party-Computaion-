from source import *


dealer = Dealer()
passed = True
for input_x in range(4):
    for input_a in range(4):
        alice = Alice(input_x, dealer.rand_a())

        bob = Bob(input_a, dealer.rand_b())

        # Communication between Alice and Bob
        bob.receive(alice.send())
        alice.receive(bob.send())

        # Output
        z = alice.out_put()
        print(f"the output of f_{input_a}({input_x}) = {z} : ", end=" ")
        if (input_x*input_a < 4 and z) or (input_x * input_a >= 4 and z==0):
            print("error")
        print("passed")




