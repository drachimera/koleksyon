# Imports
import torch
import torch.nn as nn
import torch.optim as optim  #optimization algorithms
import torch.nn.functional as F  #relu tanh ect..
import torch.utils.data as tdata  #DataLoader
# pip install torchvision
import torchvision.datasets as datasets  #mnist
import torchvision.transforms as transforms


#SHAPE Nx1x28x28 --- mnist is NOT a good dataset for RNN, typically not used on images txt is better
# but we 'force it' so we don't write a bunch of boilerplate code:
#    ----- here is how -----
#    we have 28 time sequences and each sequence has 28 features
#  here we just learn how to create the RNN
# take one row at a time, and thats what we send in at each time step

# Hyperparameters
input_size = 28
sequence_length = 28
num_layers = 2
hidden_size = 256
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 10


# Create a RNN
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        # Nxtime_seqxfeatures
        self.fc = nn.Linear(hidden_size*sequence_length,num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to_device()
        # Forward Prop
        out, _ = 



# Set device 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')





# Load Data
train_dataset = datasets.MNIST(root='/tmp/dataset/', train=True, transform=transforms.ToTensor(), download=True)  #downloads as numpy, so convert it
train_loader = tdata.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root='/tmp/dataset/', train=False, transform=transforms.ToTensor(), download=True)  #downloads as numpy, so convert it
test_loader = tdata.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Initialize network
model = NN(input_size=input_size, num_classes=num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Check accuracy on training & test to see how good our model is
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval() #change to evaluation mode, lock the network
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            x = x.reshape(x.shape[0], -1)

            scores = model(x)  #64x10
            #print(scores)  # 64 images x 10 scores
            _, predictions = scores.max(1)  # the prediction for a digit (0-9) is the max score over all the digits
            #print(predictions)  #we have 64 images so this is the prediction of the digit for each image as a list
            num_correct += (predictions == y).sum()  #number correct 
            num_samples += predictions.size(0)

            print(f'{num_correct} / {num_samples}, accuracy: {100.0*float(num_correct)/float(num_samples)}' )
    model.train()



# Train Network
for epoch in range(num_epochs): #1 epoch - the network has seen all the images in the dataset
    for batch_idx, (data, targets) in enumerate(train_loader):  #data=images target=correct label for each image
        #put to cuda if we can
        data = data.to(device=device)
        targets = targets.to(device=device)

        ##print(data.shape) # torch.Size([64, 1, 28, 28])  64=#images, 1=channels (black/white), 28x28 pixels
        #flaten the data into a single dimension
        data = data.reshape(data.shape[0], -1)
        ##print(data.shape)

        #forward
        scores = model(data)
        loss = criterion(scores, targets)

        #backward
        optimizer.zero_grad()  #don't store from previous forward props
        loss.backward()

        # gradient decent or adam step or whatever...
        optimizer.step()  #update the weights found in backward

    print("**********************************************")
    check_accuracy(test_loader, model)
    print("**********************************************")




def sanity_test():   
    """
    Just check to see if the neural network functions.
    If this works, output will be: 
    torch.Size([64, 10])
    """ 
    DATA_SIZE = 784                 # 784 is just random data, 28x28pxl
    model = NN(DATA_SIZE, 10)       # 10 is the number of digits, 
    x = torch.randn(64, DATA_SIZE)  # 
    print(model(x).shape)           


#sanity_test()
