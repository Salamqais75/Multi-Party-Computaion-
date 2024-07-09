import numpy as np

# Global parameters

prime = 41
num_parties = 2
NUMBER = 2
list_of_mults = [(False, 0, 2), (False, 1, 3)]
num_greater_than = 4
number_of_curr_Mult = 0
d_array = []
e_array = []


# Defining the AMC clss:
class MAC:
    def __init__(self, x, key, tag):
        self.value = x
        self.key = key
        self.tag = tag


################################### NEW CLASS ###################################


# Now we will define the circuit class,
# But before that we have to define the Node class, because the circuit is constructed by nodes
class Node:
    def __init__(self, op, const, parent1, parent2):
        self.value = None
        self.constant = const  # It is False when the operation is between x and y. And it is True when the operation is between x and a const
        self.operation = op             # contains "+" or "*"
        self.parent1 = parent1
        self.parent2 = parent2
        self.sons_list = []
        self.output = False     # it is True if this is leaf in the circuit. And it is False otherwise
        self.tag = -1
        self.key = -1


################################### NEW CLASS ###################################


# And now we can define the Circuit class
class Circuit:
  def __init__(self, n, input, Mults, num, p):
    self.num_of_mult_gates = 0
    self.num_parites = n
    self.input = input
    self.Mults = Mults
    self.prime = p
    self.num = num
    self.results_greater_compare = []

  def Mul(self, parent1, parent2, const):
    self.num_of_mult_gates += 1
    mul_node = Node("*", const, parent1, parent2)
    parent1.sons_list.append(mul_node)
    # We add the result of the multiplication to the array of the children nodes of the parent1
    if const == False:
      parent2.sons_list.append(mul_node)
        # It means that the second parent is not a const, so we add the result of the multiplication to the array of the children nodes of the parent2
    return mul_node

# It is similar to the Mul function
  def Add(self, parent1, parent2, const):
    add_node = Node("+", const, parent1, parent2)
    parent1.sons_list.append(add_node)
    if const == False:
      parent2.sons_list.append(add_node)
    return add_node

  def Not(self, parent):
    node = self.Add(parent, -1, True)
    node_not = self.Mul(node, -1, True)
    return node_not

  def sum(self, parent1, parent2, constant):
    sum = self.Add(parent1, parent2, False)
    self.results_greater_compare.append(self.greater_compare(sum))
    return sum


# This function repeatedly does a binary operation between pairs of nodes from the visited_list list.
# It does that until one node remains, and then it returns it.
  def do_until_one_node(self, visited_list, op, constant_flag):
    temp_visited_list = []
    while len(visited_list) > 1:
      for i in range(0, len(visited_list), 2):   # A pair of operations each time
        if (i + 1) >= len(visited_list):
          temp_visited_list.append(visited_list[i])
        else:
          temp_visited_list.append(op(visited_list[i], visited_list[i + 1], constant_flag))
      visited_list = temp_visited_list.copy()
      temp_visited_list = []
    return visited_list[0]


  def greater_compare(self, parent):
    results_of_power = []

    for i in range(self.num):
      node = self.Add(parent1=parent, parent2=np.negative(i), const=True)
      results_of_power.append(node)
      for j in range(self.prime - 2):
        node_son = self.Mul(parent1=results_of_power[i], parent2=node, const=False)
        results_of_power[i] = node_son

    result = self.do_until_one_node(results_of_power, self.Mul, False)
    return result


  def create_circuit(self):
    nodes_list = np.zeros((2 * self.num_parites), dtype=object)
    visited = []
    new_visited = []
    results_mul_list = []
    results_greater_compare_list = []
    results_not_list = []
    for i in range(len(self.input)):
        nodes_list[i] = Node(None, False, None, None)
        nodes_list[i].value = self.input[i]
    for mult in self.Mults:
        const = mult[0]
        parent1 = nodes_list[mult[1]]
        if const == False:
            parent2 = nodes_list[mult[2]]
        else:
            parent2 = mult[2]
        mul = self.Mul(parent1, parent2, const)
        results_mul_list.append(mul)
    for mul in results_mul_list:
        results_greater_compare_list.append(self.greater_compare(mul))

    self.results_greater_compare = []
    result = self.do_until_one_node(results_mul_list, self.sum, False)
    results_greater_compare_list = self.results_greater_compare
    results_greater_compare_list.append(self.greater_compare(result))

    for node in results_greater_compare_list:
      results_not_list.append(self.Not(node))

    result = self.do_until_one_node(results_not_list, self.Mul, False)
    result = self.Not(result)
    result.output = True
    return nodes_list, self.num_of_mult_gates


################################### NEW CLASS ###################################


class Dealer:
    # We have some parameters such as:
    # circuitA represents an arithmetic circuit of Alice, and similar for Bob
    # num_of_mult_gates it is the num of multiplications gates in the circuit
    # u, v and r, are arrays of random samples between 0 and prime-1, and w is the result
    # u_A is an array of MACs between 0 and prime-1, such that u=(u_a.value + u_B.value)%prime
    # similar for all the variables: u_B, v_A, v_B...

    def __init__(self):

        input_list = np.zeros((2 * num_parties), dtype=int)
        circuit = Circuit(num_parties, input_list, list_of_mults, num_greater_than, prime)
        self.circuitA, self.number_of_mult_gates = circuit.create_circuit()
        self.circuitB, self.number_of_mult_gates = circuit.create_circuit()

        alphaA = np.random.randint(0, prime)
        alphaB = np.random.randint(0, prime)

        self.u = [np.random.randint(0, prime) for i in range(self.number_of_mult_gates)]
        self.v = [np.random.randint(0, prime) for i in range(self.number_of_mult_gates)]
        self.w = np.multiply(self.u, self.v)

        self.u_A = [np.random.randint(0, prime) for i in range(self.number_of_mult_gates)]
        self.v_A = [np.random.randint(0, prime) for i in range(self.number_of_mult_gates)]
        self.w_A = [np.random.randint(0, prime) for i in range(self.number_of_mult_gates)]

        self.u_B = np.subtract(self.u, self.u_A) % prime
        self.v_B = np.subtract(self.v, self.v_A) % prime
        self.w_B = np.subtract(self.w, self.w_A) % prime

        self.u_A, self.u_B = self.create_MACs(inputA=self.u_A, inputB=self.u_B, alphaA=alphaA, alphaB=alphaB)
        self.v_A, self.v_B = self.create_MACs(inputA=self.v_A, inputB=self.v_B, alphaA=alphaA, alphaB=alphaB)
        self.w_A, self.w_B = self.create_MACs(inputA=self.w_A, inputB=self.w_B, alphaA=alphaA, alphaB=alphaB)

        self.r = [np.random.randint(0, prime) for i in range(2 * num_parties)]
        self.r_A = [np.random.randint(0, prime) for i in range(2 * num_parties)]
        self.r_B = np.subtract(self.r, self.r_A) % prime

        self.r_A, self.r_B = self.create_MACs(inputA=self.r_A, inputB=self.r_B, alphaA=alphaA, alphaB=alphaB)

    def RandA(self):
        # for Alice
        # it returns the variables that we have explained about in  beginning of the dealer class above

        return self.circuitA, self.u_A, self.v_A, self.w_A, self.r_A, self.number_of_mult_gates

    def RandB(self):
        # for Bob

        return self.circuitB, self.u_B, self.v_B, self.w_B, self.r_B, self.number_of_mult_gates

    def create_MACs(self, inputA, inputB, alphaA, alphaB):
        # inputA is an array of random variables between 0 and prime-1 of Alice..

        macs_arrayA = np.zeros(len(inputA), dtype=object)
        macs_arrayB = np.zeros(len(inputB), dtype=object)
        for i in range(len(inputA)):
            x_A = inputA[i]
            x_B = inputB[i]
            beta_A = np.random.randint(0, prime)
            beta_B = np.random.randint(0, prime)
            tag_A = (alphaB * x_A + beta_B) % prime
            tag_B = (alphaA * x_B + beta_A) % prime
            macs_arrayA[i] = MAC(x_A, (alphaA, beta_A), tag_A)
            macs_arrayB[i] = MAC(x_B, (alphaB, beta_B), tag_B)
        return macs_arrayA, macs_arrayB


################################### NEW CLASS ###################################


class Alice:

    def __init__(self, circuit, x, u_A, v_A, w_A, r_A, number_mult_gates):
      self.x = x
      self.circuit = circuit
      self.number_mult_gates = number_mult_gates

      self.u_A = u_A
      self.v_A = v_A
      self.w_A = w_A

      self.r_A = r_A

      self.mults_list = []       # a temporary array that has the indexes of mult gate in a certain layer
      self.visited_list = []     # a list that has all the nodes in the next layer
      self.occurred_nodes_list = []

      self.z_A = None       # an array that has the leaf node in Alice's circuit
      self.z_B = None       # similar to Alice, but he also sends it to Alice


    def OpenTo(self, node_x_A, node_x_B):
        # x_A and x_B are nodes

        if type(node_x_A) is np.ndarray and type(node_x_B) is np.ndarray:
            x = np.zeros((len(node_x_A)), dtype=int)
            for i in range(len(node_x_A)):
                x[i] = np.add(node_x_A[i].value, node_x_B[i].value) % prime
                if not self.verify(node_x_B[i].value, node_x_A[i].key, node_x_B[i].tag):
                    return False        # it doesn't verify
        else:
            x = np.add(node_x_A.value, node_x_B.value) % prime
            if not self.verify(node_x_B.value, node_x_A.key, node_x_B.tag):
                return False        # it doesn't verify
        return x


    def SendR_A(self):

        return self.r_A[num_parties:]     # we return an array of MACs between 0 and prime-1



    def ReceiveR_BAndSendD(self, r_B):
        # r_B is an array of MAC's between 0 and prime-1

        r = self.OpenTo(self.r_A[:num_parties], r_B)

        d = np.zeros((num_parties), dtype=int)
        for i in range(0, num_parties):
            d[i] = self.x[i] - r[i]
            self.circuit[i].value, self.circuit[i].key, self.circuit[i].tag = self.Add_with_const(self.r_A[i], d[i])
        return d        # we return an array d, such that d[i] = x[i] - r[i]


    def ReceiveD(self, d):
        # d is the parameter that d[i] = y[i] - r[i]

        for i in range(num_parties, 2 * num_parties):
            self.circuit[i].value, self.circuit[i].key, self.circuit[i].tag = self.Add_with_const(self.r_A[i], d[i - num_parties])

        for i in range(len(self.circuit)):
            for son in self.circuit[i].sons_list:
                self.visited_list.append(son)
            self.occurred_nodes_list.append(self.circuit[i])
        return


    def Add_with_const(self, node_x, c):

        key_x = node_x.key

        z_A = (node_x.value + c) % prime
        k_Az = (key_x[0] % prime, key_x[1] % prime)
        t_Az = node_x.tag % prime
        return z_A, k_Az, t_Az

    def Add(self, node_x, node_y):

        key_x = node_x.key
        key_y = node_y.key

        z_A = (node_x.value + node_y.value) % prime
        k_Az = (key_x[0] % prime, (key_x[1] + key_y[1]) % prime)
        t_Az = (node_x.tag + node_y.tag) % prime
        return z_A, k_Az, t_Az


    def Mult_with_const(self, node_x, c):

        key_x = node_x.key

        z_A = (node_x.value * c) % prime
        k_Az = (key_x[0] % prime, (key_x[1] * c) % prime)
        t_Az = (node_x.tag * c) % prime
        return z_A, k_Az, t_Az      # z_A=x_A * c, k_Az=(alpha_x, beta_x * c), and t_Az=t_x * c

    def Mult(self, node_x, node_y, u_A, v_A, w_A, d, e):
        # u_A, v_A ans w_A are MAC
        # d:=OpenTo(d_A, d_B) and e:=OpenTo(e_A, e_B)

        key_x = node_x.key
        key_y = node_y.key

        z_mul1, k_mul1, t_mul1 = self.Mult_with_const(u_A, e)
        z_mul2, k_mul2, t_mul2 = self.Mult_with_const(v_A, d)

        mac1 = MAC(z_mul1, k_mul1, t_mul1)
        mac2 = MAC(z_mul2, k_mul2, t_mul2)

        z_add1, k_add1, t_add1 = self.Add(mac1, mac2)
        mac3 = MAC(z_add1, k_add1, t_add1)

        z_add2, k_add2, t_add2 = self.Add(w_A, mac3)
        mac4 = MAC(z_add2, k_add2, t_add2)

        z_A, k_Az, t_Az = self.Add_with_const(mac4, e * d)
        return z_A, k_Az, t_Az      # z_A=x_A*y_A, k_Az=(alpha_x, beta_new), and  t_Az=t_new


    def receive(self, z_B):
        # z_B is an array of the leaf in Bob's circuit

        global d_array, e_array

        self.z_B = z_B

        for mult_node in self.mults_list:
            node_x_A = mult_node.parent1
            node_y_A = mult_node.parent2
            number_of_current_MULT = mult_node.value
            mult_node.value, mult_node.key, mult_node.tag = self.Mult(node_x_A, node_y_A,
                                                                      self.u_A[number_of_current_MULT],
                                                                      self.v_A[number_of_current_MULT],
                                                                      self.w_A[number_of_current_MULT], d_array[0],
                                                                      e_array[0])
            del d_array[0]
            del e_array[0]
        d_array = []
        e_array = []
        self.mults_list.clear()


    # Here we return an array that holds the indexes of mult gates in a specific layer
    def send(self):

        global number_of_curr_Mult

        MULTS = []
        new_visited = []

        if len(self.visited_list) == 1:
            self.z_A = self.visited_list[0]

        for node in self.visited_list:
            node_x_A = node.parent1
            if node.constant == False:
                node_y_A = node.parent2
            else:
                c = node.parent2

            if node.value is None:
                if node.operation == "+" and node.constant:
                    node.value, node.key, node.tag = self.Add_with_const(node_x_A, c)

                if node.operation == "+" and not node.constant:
                    node.value, node.key, node.tag = self.Add(node_x_A, node_y_A)

                if node.operation == "*" and node.constant:
                    node.value, node.key, node.tag = self.Mult_with_const(node_x_A, c)

                if node.operation == "*" and not node.constant:
                    self.mults_list.append(node)
                    d_A = np.subtract(node_x_A.value, self.u_A[number_of_curr_Mult].value) % prime
                    e_A = np.subtract(node_y_A.value, self.v_A[number_of_curr_Mult].value) % prime
                    node.value = number_of_curr_Mult
                    MULTS.append([d_A, e_A, number_of_curr_Mult])
                    number_of_curr_Mult += 1

                self.occurred_nodes_list.append(node)
                if len(node.sons_list) != 0:
                    for son in node.sons_list:
                        if son.parent1 in self.occurred_nodes_list and (son.constant or son.parent2 in self.occurred_nodes_list):
                            new_visited.append(son)

        self.visited_list = new_visited.copy()
        new_visited.clear()
        return MULTS


    # Here we want to verify according to MAC structure, with the value, key and a tag
    def verify(self, x, k, t):

        if (k[0] * x + k[1]) % prime == t:
            return True
        else:
            return False


    # It is a method the returns boolean value, wethter ALice has an input
    def has_output(self):

        if self.z_B is not None:
            return True
        return False


    # A method that returns Alice's output
    def return_output(self):

        z = self.OpenTo(self.z_A, self.z_B[0])
        return z


################################### NEW CLASS ###################################


# Bob's class is similar to Alice's class.
# I mean the functions and the parameters in general are similar.
class Bob:

    def __init__(self, circuit, y, u_B, v_B, w_B, r_B, number_mult_gates):
        self.y = y
        self.circuit = circuit
        self.number_mult_gates = number_mult_gates

        self.u_B = u_B
        self.v_B = v_B
        self.w_B = w_B

        self.r_B = r_B

        self.visited = []
        self.occurred_nodes = []

        self.z_B = None



    def OpenTo(self, node_x_A, node_x_B):

        if type(node_x_A) is np.ndarray and type(node_x_B) is np.ndarray:
            x = np.zeros((len(node_x_A)), dtype=int)
            for i in range(len(node_x_A)):
                x[i] = np.add(node_x_A[i].value, node_x_B[i].value) % prime
                if not self.verify(node_x_A[i].value, node_x_B[i].key, node_x_A[i].tag):
                    return False        # It doesn't verify
        else:
            x = np.add(node_x_A.value, node_x_B.value) % prime
            if not self.verify(node_x_A.value, node_x_B.key, node_x_A.tag):
                return False            # It doesn't verify
        return x



    def SendR_B(self):

        return self.r_B[:num_parties]     # returns an array of MACs between 0 and prime-1


    def ReceiveD(self, d):
        # d:=d[i] = x[i] - r[i]

        for i in range(0, num_parties):
            self.circuit[i].value, self.circuit[i].key, self.circuit[i].tag = self.Add_with_const(self.r_B[i], d[i])


    def ReceiveR_AAndSendD(self, r_A):
        # r_A an array of MACs between 0 and prime-1

        r = self.OpenTo(r_A, self.r_B[num_parties:])

        d = np.zeros((num_parties), dtype=int)
        for i in range(0, num_parties):
            d[i] = self.y[i] - r[i]
            self.circuit[i + num_parties].value, self.circuit[i + num_parties].key, self.circuit[i + num_parties].tag = self.Add_with_const(
                self.r_B[i + num_parties], d[i])

        for i in range(len(self.circuit)):
            for son in self.circuit[i].sons_list:
                self.visited.append(son)
            self.occurred_nodes.append(self.circuit[i])
        return d        # an array such that d[i] = y[i] - r[i]


    def Add_with_const(self, node_x, c):

        key_x = node_x.key

        z_B = (node_x.value) % prime
        k_Bz = (key_x[0] % prime, (key_x[1] - c * key_x[0]) % prime)
        t_Bz = node_x.tag % prime
        return z_B, k_Bz, t_Bz
        # z_B: x_B + c, k_Bz: (alpha_x, beta_x - c * alpha_x), and t_Bz: t_x


    def Add(self, node_x, node_y):

        key_x = node_x.key
        key_y = node_y.key

        z_B = (node_x.value + node_y.value) % prime
        k_Bz = (key_x[0] % prime, (key_x[1] + key_y[1]) % prime)
        t_Bz = (node_x.tag + node_y.tag) % prime
        return z_B, k_Bz, t_Bz
        # z_B: x_B + y_B, k_Bz: (alpha_x, beta_x + beta_y), and t_Bz: t_x + t_y


    def Mult_with_const(self, node_x, c):

        key_x = node_x.key

        z_B = (node_x.value * c) % prime
        k_Bz = (key_x[0] % prime, (key_x[1] * c) % prime)
        t_Bz = (node_x.tag * c) % prime
        return z_B, k_Bz, t_Bz
        # z_B: x_B * c, k_Bz: (alpha_x, beta_x * c), and t_Bz: t_x * c


    def Mult(self, node_x, node_y, u_B, v_B, w_B, d, e):
        # u_B, v_B and w_B are MACs
        # d=OpenTo(d_A, d_B), and e=OpenTo(e_A, e_B)

        key_x = node_x.key
        key_y = node_y.key

        z_mul1, k_mul1, t_mul1 = self.Mult_with_const(u_B, e)
        z_mul2, k_mul2, t_mul2 = self.Mult_with_const(v_B, d)

        mac1 = MAC(z_mul1, k_mul1, t_mul1)
        mac2 = MAC(z_mul2, k_mul2, t_mul2)

        z_add1, k_add1, t_add1 = self.Add(mac1, mac2)
        mac3 = MAC(z_add1, k_add1, t_add1)

        z_add2, k_add2, t_add2 = self.Add(w_B, mac3)
        mac4 = MAC(z_add2, k_add2, t_add2)

        z_B, k_Bz, t_Bz = self.Add_with_const(mac4, e * d)
        return z_B, k_Bz, t_Bz
        # z_B: x_B * y_B,  k_Bz: (alpha_x, beta_new), and t_Bz: t_new


    # It is a method that receives what send method from Alice send
    # And they are similar in the construction
    def receive(self, MULTS):
        # Mults an array that has the indexes of multiplications gates in a certain layer

        global d_array, e_array

        new_visited = []

        for node in self.visited:
            node_x_B = node.parent1
            if node.constant == False:
                node_y_B = node.parent2
            else:
                c = node.parent2

            if node.value is None:
                if node.operation == "+" and node.constant:
                    node.value, node.key, node.tag = self.Add_with_const(node_x_B, c)

                if node.operation == "+" and not node.constant:
                    node.value, node.key, node.tag = self.Add(node_x_B, node_y_B)

                if node.operation == "*" and node.constant:
                    node.value, node.key, node.tag = self.Mult_with_const(node_x_B, c)

                if node.operation == "*" and not node.constant:
                    number_of_current_MULT = MULTS[0][2]
                    d_A = MULTS[0][0]
                    d_B = np.subtract(node_x_B.value, self.u_B[number_of_current_MULT].value) % prime
                    d_array.append(d_A + d_B)

                    e_A = MULTS[0][1]
                    e_B = np.subtract(node_y_B.value, self.v_B[number_of_current_MULT].value) % prime
                    e_array.append(e_A + e_B)

                    del MULTS[0]

                    node.value, node.key, node.tag = self.Mult(node_x_B, node_y_B, self.u_B[number_of_current_MULT],
                                                               self.v_B[number_of_current_MULT],
                                                               self.w_B[number_of_current_MULT], d_array[-1],
                                                               e_array[-1])

                self.occurred_nodes.append(node)
                if len(node.sons_list) != 0:
                    for son in node.sons_list:
                        if son.parent1 in self.occurred_nodes and (
                                son.constant == True or son.parent2 in self.occurred_nodes):
                            new_visited.append(son)
                    self.z_B = new_visited.copy()

        self.visited = new_visited.copy()
        new_visited.clear()


    def send(self):

        if len(self.visited) > 0:
            return None
        else:
            return self.z_B     # returns an array of the leaf in Bob's circuit


    # It verifies the MAC
    def verify(self, x, k, t):

        if (k[0] * x + k[1]) % prime == t:
            return True
        else:
            return False



