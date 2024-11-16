import os
import numpy as np
import matplotlib.pyplot as plt

if not os.path.exists('images'):
    os.makedirs('images')

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
    
def draw_neural_network(layer_sizes):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axis('off')
    n_layers = len(layer_sizes)
    v_spacing = (1.0 - 0.2) / float(max(layer_sizes))
    h_spacing = 1.0 / float(n_layers - 1) if n_layers > 1 else 1.0

    layer_colors = ['skyblue', 'lightgreen', 'salmon', 'plum', 'khaki']
    
    for i, layer_size in enumerate(layer_sizes):
        layer_top = (1.0 - (layer_size - 1) * v_spacing) / 2.0
        color = layer_colors[i % len(layer_colors)]
        for j in range(layer_size):
            circle = plt.Circle((i * h_spacing + 0.1, layer_top + j * v_spacing), v_spacing / 4,
                                color=color, ec='black', zorder=4)
            ax.add_artist(circle)
            ax.text(i * h_spacing + 0.1, layer_top + j * v_spacing, f'N{j+1}',
                    fontsize=9, ha='center', va='center')
    
    for i, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = (1.0 - (layer_size_a - 1) * v_spacing) / 2.0
        layer_top_b = (1.0 - (layer_size_b - 1) * v_spacing) / 2.0
        for j in range(layer_size_a):
            for k in range(layer_size_b):
                line = plt.Line2D([i * h_spacing + 0.1, (i + 1) * h_spacing + 0.1],
                                  [layer_top_a + j * v_spacing, layer_top_b + k * v_spacing],
                                  c='gray', lw=1)
                ax.add_artist(line)
    
    for i in range(n_layers):
        layer_label = ''
        if i == 0:
            layer_label = 'Input'
        elif i == n_layers - 1:
            layer_label = 'Output'
        else:
            layer_label = f'Hidden {i}'
        ax.text(i * h_spacing + 0.1, 1.05, layer_label, fontsize=12, ha='center', va='bottom')
    
    ax.set_xlim(0, 1.0 + 0.2)
    ax.set_ylim(0, 1.0 + 0.2)
    plt.tight_layout()
    plt.savefig('images/neural_network_structure.png', dpi=300)
    plt.show()

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
plt.savefig('images/loss_evolution.png')
plt.show()

layer_sizes = [NN.inputSize, NN.hiddenSize, NN.outputSize]
draw_neural_network(layer_sizes)