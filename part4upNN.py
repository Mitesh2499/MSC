import numpy as np

class NeuralNetwork:
    def __init__(self):
        # initialize weights
        self.w1 = np.random.randn(2, 2)
        self.w2 = np.random.randn(2, 1)

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, x):
        # forward propagation
        self.z1 = np.dot(x, self.w1)
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.w2)
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def backward(self, x, y):
        # calculate error
        error = y - self.a2
        d2 = error * self.sigmoid_derivative(self.a2)

        # backpropagation
        error_hidden = np.dot(d2, self.w2.T)
        d1 = error_hidden * self.sigmoid_derivative(self.a1)

        # update weights
        self.w2 += np.dot(self.a1.T, d2)
        self.w1 += np.dot(x.T, d1)

    def train(self, x, y, epochs=1000):
        for i in range(epochs):
            self.forward(x)
            self.backward(x, y)

    def predict(self, x_test):
        return self.forward(x_test)[0][0]

# define input and output arrays
x=np.array(([2,9],[1,5],[3,6]),dtype=float)
y=np.array(([92],[86],[89]),dtype=float)

# normalize the input data
x = x/np.amax(x, axis=0)
y = y/100 # max test score is 100

# create the model
nn = NeuralNetwork()

# train the model
nn.train(x, y, epochs=1000)

# predict using the trained model
x_test = np.array([[1, 8]])
predicted_test_score = nn.predict(x_test)

print("Predicted test score: {:.2f}%".format(predicted_test_score*100))
print("input is "+str(x))
print("desire output is "+str(y))
print("loss is "+str(np.mean(np.square(y-nn.forward(x)))))
print("actual output "+str(nn.forward(x)))
