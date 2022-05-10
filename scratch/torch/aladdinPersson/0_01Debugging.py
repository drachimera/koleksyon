
from torchvision import models #just an example model
from torchinfo import summary

model = models.vgg16()
summary(model)

#model = ConvNet()
#batch_size = 16
#summary(model, input_size=(batch_size, 1, 28, 28))


