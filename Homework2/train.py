import torch
import torchvision
import torchvision.transforms as transforms

# Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load CIFAR10 data
train_data = torchvision.datasets.CIFAR10(root='./datasets',
                                        train=True,
                                        transform=transforms.ToTensor(),
                                        download=True)

in_channel = 3
max_pool_kernel = 2

class ConvNetStep(torch.nn.Module) :
    def __init__(self, num_classes = 10) :
        super(ConvNetStep, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = in_channel,
                      out_channels = 12,
                      kernel_size = 5,
                      stride = 1,
                      padding = 2),
            torch.nn.BatchNorm2d(num_features = 12),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = max_pool_kernel)
        )
        
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = 12,
                      out_channels = 24,
                      kernel_size = 5,
                      stride = 1, 
                      padding = 2),
            torch.nn.BatchNorm2d(num_features = 24),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = max_pool_kernel)
        )
        
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = 24,
                      out_channels = 48,
                      kernel_size = 5,
                      stride = 1, 
                      padding = 2),
            torch.nn.BatchNorm2d(num_features = 48),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = max_pool_kernel)
        )
        
        self.fc1 = torch.nn.Linear(in_features = 48*4*4, out_features = 128)
        self.fc2 = torch.nn.Linear(in_features = 128, out_features = 32)
        self.fc3 = torch.nn.Linear(in_features = 32, out_features = num_classes) #num_classes = 10
    
    def forward(self, x) :
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = torch.nn.functional.relu(x)
        x = x.reshape(x.size(0), -1)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
model = ConvNetStep()