# Imports
import torch
import torch.nn as nn
import torch.optim as optim  #optimization algorithms
import torch.nn.functional as F  #relu tanh ect..
import torch.utils.data as tdata  #DataLoader
# pip install torchvision
import torchvision.datasets as datasets  #mnist
import torchvision.transforms as transforms

# Create Fully Connected Neural Network
class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50,50)
        self.fc3 = nn.Linear(50,50)
        self.fc4 = nn.Linear(50, num_classes)
    def forward(self, x):
        x = F.tanh(self.fc1(x))  #or relu
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        x = self.fc4(x)
        return x


class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(CNN, self).__init__()
        F.tanh()


# Set device 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 10

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