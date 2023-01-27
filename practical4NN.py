import numpy as np
# define the input and desire output matrix
# define here sleeping hours and studing hours
x=np.array(([2,9],[1,5],[3,6]),dtype=float)
# desire output
y=np.array(([92],[86],[89]),dtype=float)

# scale the parameter
x=x/np.amax(x,axis=0)
y=y/100
print(x)
print(y)
class neuralNetwork():
    def __init__(self):
        self.input=2
        self.output=1
        self.hidden=3
        self.w1=np.random.randn(self.input,self.hidden)
        self.w2=np.random.randn(self.hidden,self.output)
       
    def forward(self,x):
        self.z=np.dot(x,self.w1)
        self.z2=self.sigmoid(self.z)
        self.z3=np.dot(self.z2,self.w2)
        output=self.sigmoid(self.z3)
        return output
    
    #activation function
    def sigmoid(self,s,deriv=False):
        if(deriv==True):
            return s*(1-s)
        return 1/(1+(np.exp(-s)))
   
    def backword(self,x,y,output):
        self.output=y-output
        self.output_error_delta=self.output*self.sigmoid(self.output,deriv=True)
       
        self.z2_error=self.output_error_delta*self.sigmoid(self.w2.T)
        self.z2_error_delta=self.z2_error*self.sigmoid(self.z2,deriv=True)
       
        self.w1+=x.T.dot(self.z2_error_delta)
        self.w2+=self.z2.T.dot(self.output_error_delta)
       
    def train(self,x,y):
        output=self.forward(x)
        self.backword(x,y,output)
       
       
NN=neuralNetwork()
for i in range(1000):
    NN.train(x,y)
   
print("input is "+str(x))
print("desire output is "+str(y))
print("loss is "+str(np.mean(np.square(y-NN.forward(x)))))
print("actual output "+str(NN.forward(x)))
