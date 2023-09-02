import torch
import torchvision
import torchvision.transforms as transforms

# Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_data = torchvision.datasets.CIFAR10(root='./datasets',
                                       train=False,
                                       transform=transforms.ToTensor(),
                                       download=True)

## train에서 직접 만드신 model class 코드가 들어가는 자리입니다
## train.py에서의 model class를 복사해서 붙이시면 됩니다.
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

## 위에 model class 코드를 입력하셨다면 test는 여기서부터 진행하시면 됩니다.

batch_size = 64
learning_rate = 0.005
num_epochs = 7

train_loader = torch.utils.data.DataLoader(dataset = train_data,
                                          batch_size = batch_size,
                                          shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = test_data,
                                          batch_size = batch_size,
                                          shuffle = False) 

test_model = ConvNetStep().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(test_model.parameters(), lr = learning_rate)

total_step = len(train_loader)
total_loss = []
model.train()
for epoch in range(num_epochs):
    epoch_loss = []
    for i, (img, label) in enumerate(train_loader):
        img = img.to(device)
        label = label.to(device)

        outputs = test_model(img)

        loss = criterion(outputs, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss.append(loss.detach().cpu().numpy()) 
        if i % 100 == 0 or (i+1) == len(train_loader):
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, 
                                                                     i+1, len(train_loader), loss.item()))
    total_loss.append(np.mean(epoch_loss))
    print(f"epoch{i} loss: {np.mean(epoch_loss)}")

test_model.load_state_dict(torch.load('model.pth'))
acc_list = []
test_model.eval()