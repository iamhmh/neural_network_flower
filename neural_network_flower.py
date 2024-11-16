import numpy as np
import matplotlib.pyplot as plt

x_entry =  np.array(([3, 1.5], [2, 1], [4, 1.5], [3, 1], [3.5,0.5], [2,0.5], [5.5,1], [1,1], [4,1.5]), dtype=float)
y = np.array(([1], [0], [1],[0],[1],[0],[1],[0]), dtype=float) # 1 = red, 0 = blue

x_entry = x_entry/np.amax(x_entry, axis=0) # scaling the data to 0-1

X = np.split(x_entry, [8])[0] # training data
xPredicted = np.split(x_entry, [8])[1] # testing data

class Neural_Network(object):
    def __init__(self):
        # parameters
        self.inputSize = 2
        self.outputSize = 1
        self.hiddenSize = 3

        # weights
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize) # matrice 2x3 entry neural to hidden neural
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize) # matrice 3x1 hidden neural to ouput neural

        self.loss_history = [] # loss history for the training data

        # propagation function
    def forward(self, X):
        self.z = np.dot(X, self.W1) # multiplication between entry values and W1 weight
        self.z2 = self.sigmoid(self.z) # activation function (sigmoid)
        self.z3 = np.dot(self.z2, self.W2) # multiplication between hidden values and W2 weight

        o = self.sigmoid(self.z3) # activation function for getting the output values
        return o

    def sigmoid(self, s):
        return 1/(1+np.exp(-s)) # activation function w/ sigmoid
    
    def sigmoidPrime(self, s):
        return s * (1 - s)

    # backward function
    def backward(self, X, y, o):

        self.o_error = y - o # error calc
        self.o_delta = self.o_error*self.sigmoidPrime(o) # sigmoidPrime for the error

        self.z2_error = self.o_delta.dot(self.W2.T) # error calc hidden neural
        self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2) # sifmoidPrime for the error

        self.W1 += X.T.dot(self.z2_delta) # add W1 weight
        self.W2 += self.z2.T.dot(self.o_delta) # add W2 weight

    # train function
    def train(self, X, y):
        o = self.forward(X)
        self.backward(X, y, o)
        loss = np.mean(np.square(y - o))
        self.loss_history.append(loss)

    # prediction function
    def prediction(self):

        print("Predicted Data after train : ")
        print("Entry : \n" + str(xPredicted))
        print("Ouput : \n" + str(self.forward(xPredicted)))

        if(self.forward(xPredicted) < 0.5):
            print("The flower is blue ! \n")
        else:
            print("The flower is red ! \n")

NN = Neural_Network()

for i in range(1000):
    print("# " + str(i) + "\n")
    print("Entries values: \n" + str(X))
    print("Actual output value: \n" + str(X))
    print("Predicted output: \n" + str(np.matrix.round(NN.forward(X),2)))
    print("\n")
    NN.train(X,y)

NN.prediction()

plt.plot(NN.loss_history)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss evolution during training')
plt.savefig('charts/loss_evolution.png')
plt.show()