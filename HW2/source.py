import numpy as np
import random


class Dealer:
    __Truth_table = np.array([[0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 1, 1],
                             [0, 0, 1, 1]])

    def __init__(self):
        # get random numbers
        self.__r = random.randint(1, 4)
        self.__c = random.randint(1, 4)
        # shift row
        __shifted_col_truth_table = np.roll(self.__Truth_table, self.__r, axis=0)
        self.__shifted_truth_table = np.roll(__shifted_col_truth_table, self.__c, axis=1)

        self.__random_matrixB = np.random.randint(0, 2, size=(4, 4))
        self.__matrixA = np.bitwise_xor(self.__random_matrixB, self.__shifted_truth_table)

    def rand_a(self):
        Alice_out_put = [self.__r, self.__matrixA]
        return Alice_out_put

    def rand_b(self):
        Bob_out_put = [self.__c, self.__random_matrixB]
        return Bob_out_put


class Alice:
    def __init__(self, my_input, massage):
        self.__u = (my_input + massage[0]) % 4
        self.__matrix = massage[1]
        self.__zb = -1
        self.__v = -1

    def send(self):
        return self.__u

    def receive(self, bob_massage):
        self.__zb = bob_massage[1]
        self.__v = bob_massage[0]

    def out_put(self):
        if self.__zb != -1 and self.__v != -1:
            out_put_r = self.__matrix[self.__u][self.__v] ^ self.__zb
            return out_put_r
        raise Exception("haven't received the massage from bob still waiting")


class Bob:
    def __init__(self, my_input, massage):
        self.__v = (my_input + massage[0]) % 4
        self.__matrix = massage[1]
        self.__zb = -1

    def send(self):
        if self.__zb == -1:
            raise Exception("waiting to receive the massage from Alice try again later.")

        massage_send = [self.__v, self.__zb]
        return massage_send

    def receive(self, alice_massage):
        u = alice_massage
        self.__zb = self.__matrix[u][self.__v]


# Main part of the code
