import numpy as np

number_of_AND_gates = 21
curr_layer = 0
layers = 14
number_of_AND = 0
NUMBER = 4
d = 0
e = 0


class Dealer:
    # the dealer will get the Beaver triples as the number of the AND gates because
    # we will use 1 set of the Beaver triples to calculate the and gate

    def __init__(self):
        self.u = []
        self.v = []
        self.u_A = []
        self.v_A = []
        self.w_A = []

        c = 0
        while c != number_of_AND_gates:
            self.u.append(np.random.randint(0, 2))
            self.v.append(np.random.randint(0, 2))
            c += 1
        # calc the w from the v and u 
        self.w = np.bitwise_and(self.u, self.v)

        # here we are getting the shares for bob and alice from the Beaver triples
        # as that _A going to stand for alice share and _B for BOB share
        c = 0
        while c != number_of_AND_gates:
            self.u_A.append(np.random.randint(0, 2))
            self.v_A.append(np.random.randint(0, 2))
            self.w_A.append(np.random.randint(0, 2))
            c += 1

        self.u_B = np.bitwise_xor(self.u, self.u_A)
        self.v_B = np.bitwise_xor(self.v, self.v_A)
        self.w_B = np.bitwise_xor(self.w, self.w_A)

        # in this array we will define our circuit
        #     gate is X if the gate is xor and A if the gate is an and gate.
        #     i is 1 if wire2 is a const number and 0 otherwise.
        #     wire1 is the index of the first wire 
        #     wire2 is the second wire of the gate: can be a const number (0 or 1) or the index of the second wire.
        self.circuit = np.array(
            [[('A', 0, 0, 4), ('A', 0, 0, 5), ('A', 0, 1, 4), ('A', 0, 1, 5), ('A', 0, 2, 6), ('A', 0, 2, 7),
              ('A', 0, 3, 6), ('A', 0, 3, 7)],
             [('X', 1, 0, 0), ('A', 0, 1, 2), ('X', 0, 1, 2), ('X', 1, 3, 0), ('X', 1, 4, 0), ('A', 0, 5, 6),
              ('X', 0, 5, 6), ('X', 1, 7, 0)],
             [('A', 0, 0, 1), ('X', 0, 0, 1), ('X', 1, 2, 0), ('X', 1, 3, 0), ('A', 0, 4, 5), ('X', 0, 4, 5),
              ('X', 1, 6, 0), ('X', 1, 7, 0)],
             [('A', 0, 0, 4), ('X', 0, 0, 4), ('A', 0, 1, 5), ('X', 0, 1, 5), ('A', 0, 2, 6), ('X', 0, 2, 6),
              ('A', 0, 3, 7), ('X', 0, 3, 7)],
             [('X', 1, 0, 0), ('X', 1, 1, 0), ('X', 1, 2, 0), ('X', 1, 3, 0), ('X', 1, 4, 0), ('A', 0, 5, 6),
              ('X', 0, 5, 6), ('X', 1, 7, 0)],
             [('X', 1, 0, 0), ('X', 1, 1, 0), ('X', 1, 2, 0), ('X', 1, 3, 0), ('X', 0, 4, 5), ('X', 1, 6, 0),
              ('X', 1, 7, 0)],
             [('X', 1, 0, 0), ('X', 1, 1, 0), ('X', 1, 2, 0), ('A', 0, 3, 4), ('X', 0, 3, 4), ('X', 1, 5, 0),
              ('X', 1, 6, 0)],
             [('X', 1, 0, 0), ('X', 1, 1, 0), ('X', 0, 2, 3), ('X', 1, 4, 0), ('X', 1, 5, 0), ('X', 1, 6, 0)],
             [('X', 1, 0, 0), ('A', 0, 1, 2), ('X', 0, 1, 2), ('X', 1, 3, 0), ('X', 1, 4, 0), ('X', 1, 5, 0)],
             [('X', 0, 0, 1), ('X', 1, 2, 0), ('X', 1, 3, 0), ('X', 1, 4, 0), ('X', 1, 5, 0)],
             [('A', 0, 0, 1), ('X', 0, 0, 1), ('X', 1, 2, 0)],
             [('X', 0, 0, 1), ('X', 1, 2, 0)],
             [('A', 0, 0, 1), ('X', 0, 0, 1)],
             [('X', 0, 0, 1)]], dtype=object)

    # this two function gets the parameters for Alice and BOB (_a means alice, _b bob)
    # each one will get the circuit and his shares
    def rand_a(self):
        return self.u_A, self.v_A, self.w_A

    def rand_b(self):
        return self.u_B, self.v_B, self.w_B


class Alice:
    # getting the params (u_A, v_A, w_A) which is the shared Beaver triples for alice
    def __init__(self, alice_input, circuit, u_A, v_A, w_A):
        self.x = alice_input

        self.message_a = np.zeros((2 * NUMBER), dtype=int)
        for i in range(NUMBER):
            self.message_a[i] = np.random.randint(0, 2)
            
        self.circuit = circuit
        self.u_A = u_A
        self.v_A = v_A
        self.w_A = w_A

        self.sync = -1
        self.z_Bob = -1

    # send the share to bob of alice input
    def send_bob_share(self):
        x_B = np.zeros(NUMBER, dtype=int)
        for i in range(0, NUMBER):
            x_B[i] = np.bitwise_xor(self.x[i], self.message_a[i])
        return x_B

    # get the share of bob input (bob_input_a, bob_input_b) <- Share(B, bob_input).
    def receive_bob_share(self, bob_a):
        for i in range(NUMBER, 2 * NUMBER):
            self.message_a[i] = bob_a[i - 2 * NUMBER]

    # receive list of the result that bob computes 
    def receive(self, z_Bob):
        global d, e
        global curr_layer
        global number_of_AND

        self.z_Bob = z_Bob

        for index in range(len(self.circuit[curr_layer])):
            # check if there is need to sync for this gate
            if self.sync[index] != -1:
                # getting the information about the current AND gate
                current_sync_index = self.sync[index]
                num_current_and = self.message_a[current_sync_index][2]

                # do XOR and AND operations to update message_a for the current AND gate
                xor_1 = np.bitwise_xor(self.w_A[num_current_and], np.bitwise_and(
                    e[current_sync_index],
                    np.bitwise_xor(self.message_a[current_sync_index][0], self.u_A[num_current_and])))
                xor_2 = np.bitwise_and(d[current_sync_index], np.bitwise_xor(
                    self.message_a[current_sync_index][1], self.v_A[num_current_and]))

                self.message_a[current_sync_index] = np.bitwise_xor(xor_1, xor_2)
                d[current_sync_index] = -1
                e[current_sync_index] = -1
                self.sync[index] = -1

        curr_layer += 1

    def xor_X_Y(self, x, y):
        return np.bitwise_xor(x, y)

    def xor_X_C(self, x, c):
        return np.bitwise_xor(x, c)

    def and_X_Y(self, x, y):
        global number_of_AND
        z_A = [np.bitwise_xor(self.message_a[x], self.u_A[number_of_AND]),
               np.bitwise_xor(self.message_a[y], self.v_A[number_of_AND]),
               number_of_AND]
        return z_A

    def and_X_C(self, x, c):
        return np.bitwise_and(self.message_a[x], c)

    def send(self):
        global number_of_AND

        z_A = []
        ands = []
        self.sync = []

        for _ in range(len(self.circuit[curr_layer])):
            z_A.append(0)
            ands.append(-1)
            self.sync.append(-1)

        idx = 0
        for i in range(len(self.circuit[curr_layer])):
            if self.circuit[curr_layer][i][0] == 'X':  # XOR
                op_type = self.circuit[curr_layer][i][1]
                if op_type == 0:  # XOR -> XOR([x],[y])
                    z_A[i] = self.xor_X_Y(self.message_a[self.circuit[curr_layer][i][2]], self.message_a[self.circuit[curr_layer][i][3]])
                else:  # XOR -> XOR([x],c)
                    z_A[i] = self.xor_X_C(
                        self.message_a[self.circuit[curr_layer][i][2]], self.circuit[curr_layer][i][3])
            else:  # AND
                if self.circuit[curr_layer][i][1] == 0:  # AND([x],[y])
                    self.sync[idx] = i
                    idx += 1
                    z_A[i] = self.and_X_Y(self.circuit[curr_layer][i][2], self.circuit[curr_layer][i][3])
                    ands[i] = z_A[i]
                    number_of_AND += 1
                else:  # AND([x],c)
                    z_A[i] = self.and_X_C(self.circuit[curr_layer][i][2], self.circuit[curr_layer][i][3])

        self.message_a = z_A
        return ands

    def has_output(self):
        if curr_layer < layers:
            return False
        return True

    # return the output
    def output(self):
        return np.bitwise_xor(self.z_Bob[0], self.message_a[0])


class Bob:
    # getting the params (u_B, v_B, w_B) which is the shared Beaver triples for Bob

    def __init__(self, bob_input, circuit, u_B, v_B, w_B):
        self.y = bob_input
        self.circuit = circuit

        self.u_B = u_B
        self.v_B = v_B
        self.w_B = w_B

        self.message_b = np.zeros((2 * NUMBER), dtype=int)


    def send_alice_share(self):
        y_A = np.zeros((NUMBER), dtype=int)
        for i in range(NUMBER, 2 * NUMBER):
            self.message_b[i] = np.random.randint(0, 2)
            y_A[i - NUMBER] = np.bitwise_xor(self.y[i - NUMBER], self.message_b[i])
        return y_A


    # get the share of alice input (alice_input_a, alice_input_b) <- Share(A, alice_input).
    def receive_alice_share(self, x_b):

        for i in range(0, NUMBER):
            self.message_b[i] = x_b[i]

    def xor_X_Y(self, x, y):
        return np.bitwise_xor(x,y)

    def and_X_C(self, x, c):
        return np.bitwise_and(x, c)

    # receive list of the result that alice computes ands[i] = [x_A xor u_A, y_A xor v_A, #AND].
    def receive(self, ands):
        global d, e
        z_Bob = np.zeros(len(self.circuit[curr_layer]), dtype=int)
        d = np.full(len(self.circuit[curr_layer]), -1, dtype=int)
        e = np.full(len(self.circuit[curr_layer]), -1, dtype=int)

        number_of_gates_layer = len(self.circuit[curr_layer])
        for i in range(number_of_gates_layer):
            gate_type, gate_param = self.circuit[curr_layer][i][0], self.circuit[curr_layer][i][1]

            if gate_type == 'X':  # XOR
                x_idx, y_idx = self.circuit[curr_layer][i][2], self.circuit[curr_layer][i][3]
                z_Bob[i] = self.xor_X_Y(self.message_b[self.circuit[curr_layer][i][2]], self.message_b[self.circuit[curr_layer][i][3]]) if gate_param == 0 else self.message_b[x_idx]
            else:  # AND
                x_idx, y_idx = self.circuit[curr_layer][i][2], self.circuit[curr_layer][i][3]
                and_info = ands[i]
                number_of_current_AND = and_info[2]

                d[i] = np.bitwise_xor(and_info[0],
                                      np.bitwise_xor(self.message_b[x_idx], self.u_B[number_of_current_AND]))
                e[i] = np.bitwise_xor(and_info[1],
                                      np.bitwise_xor(self.message_b[y_idx], self.v_B[number_of_current_AND]))

                res1 = np.bitwise_xor(self.w_B[number_of_current_AND], np.bitwise_and(e[i], self.message_b[x_idx]))
                res2 = np.bitwise_xor(np.bitwise_and(d[i], self.message_b[y_idx]), np.bitwise_and(e[i], d[i]))
                z_Bob[i] = np.bitwise_xor(res1, res2) if gate_param == 0 else self.and_X_C(self.message_b[x_idx], y_idx)

        self.message_b = z_Bob

    def send(self):
        z_Bob = self.message_b
        return z_Bob



