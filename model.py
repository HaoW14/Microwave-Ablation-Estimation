import warnings
warnings.filterwarnings("ignore")
import torch.nn as nn
import torchvision
from tqdm import tqdm
import torch
from torch.utils.data import random_split, DataLoader
CUDA = torch.cuda.is_available()

class Decoder(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(Decoder, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_relu = nn.Sequential(
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat((x1, x2), dim=1)
        x1 = self.conv_relu(x1)
        return x1

class Unet(nn.Module):  # resunet model
    def __init__(self, n_class):
        super().__init__()

        self.base_model = torchvision.models.resnet18(True)
        self.base_layers = list(self.base_model.children())
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),  # input channel = 1
            self.base_layers[1],
            self.base_layers[2])
        self.layer2 = nn.Sequential(*self.base_layers[3:5])
        self.layer3 = self.base_layers[5]
        self.layer4 = self.base_layers[6]
        self.layer5 = self.base_layers[7]
        self.decode4 = Decoder(512, 256 + 256, 256)
        self.decode3 = Decoder(256, 256 + 128, 256)
        self.decode2 = Decoder(256, 128 + 64, 128)
        self.decode1 = Decoder(128, 64 + 64, 64)

        self.decode0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.conv_last = nn.Conv2d(32, n_class, 1)

        self.toseq = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        self.sequence = lstm(9)

    def forward(self, input, text):
        e1 = self.layer1(input)  
        e2 = self.layer2(e1)  
        e3 = self.layer3(e2)  
        e4 = self.layer4(e3)  
        e5 = self.layer5(e4)  

        f = self.toseq(e5)   ##make the map feature to sequence-like features
        f = torch.squeeze(f,2)
        f = torch.squeeze(f, 2) #just for linear operation
        feature = self.fc(f)

        d4 = self.decode4(e5, e4)  
        d3 = self.decode3(d4, e3)  
        d2 = self.decode2(d3, e2)  
        d1 = self.decode1(d2, e1)  
        d0 = self.decode0(d1)  
        out = self.conv_last(d0)  


        inp = torch.cat([feature[:, None, :], text], 2)  #concat for input of lstm
        pred_time = self.sequence(inp)
        return pred_time, out

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)

class lstm(nn.Module):
    def __init__(self, input_size, hidden_size=4, output_size=1,num_layers = 4):
        super(lstm, self).__init__()
        self.layer2 = nn.LSTM(input_size, 4, num_layers, batch_first= True)
        self.layer4 = nn.Linear(4, output_size) 

    def forward(self, x):
        v2, _ = self.layer2(x)
        b, s, h = v2.size()
        v2 = v2.view(s * b, -1)
        v = self.layer4(v2)
        x = v.view(b, s, -1)
        return x

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                #tanh_gain = nn.init.calculate_gain('tanh')
                #nn.init.xavier_uniform_(m.weight.data, gain=tanh_gain)
                nn.init.kaiming_normal_(m.weight.data)

class lstm2(nn.Module):  #model for needle_distance prediction
    def __init__(self, output_size=1, num_layers=4):
        super(lstm2, self).__init__()
        self.layer2 = nn.LSTM(3, 2, num_layers, batch_first=True)
        self.layer4 = nn.Linear(2, output_size)

    def forward(self, x):
        v2, _ = self.layer2(x[:, :, :3])
        v = self.layer4(v2)
        return v

def dice(logits, targets,classindex):# logits:b * c * s * h * w     targets: b  s * h * w
    logits = nn.Softmax(1)(logits)
    b = logits.size()[0]
    smooth = 1
    targets = torch.unsqueeze(targets,1)
    targets = torch.zeros(logits.shape).cuda().scatter_(1, targets, 1)
    input_flat = logits[:,classindex,:,:].view(b,-1) # contiguous使内存地址连续，才能用view
    targets_flat = targets[:,classindex,:,:].view(b,-1)

    intersection = input_flat * targets_flat
    union = input_flat.sum(1) + targets_flat.sum(1)
    total_batch_dice = (2. * intersection.sum(1) + smooth) / (union + smooth)
    mean_batch_dice = total_batch_dice.mean() 
    return mean_batch_dice

 def train(train_loader, val_loader):
    mymodel = Unet(2).cuda()
    mymodel.initialize()
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.MSELoss()
    num_epoch = 500


    scheduler_step = num_epoch // 4
    optimizer = torch.optim.Adam(mymodel.parameters(), lr=0.005)
    #lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, scheduler_step, 0.001)
    maxacc = 0.0
    for epoch in range(num_epoch):
        epoch_start_time = time.time()
        train_loss = 0.0
        val_loss = 0.0
        train_dice = 0.0
        val_dice = 0.0
        mymodel.train()
        for image, mask,text_x, text_y in tqdm(train_loader, total=len(train_loader)):
            image = image.cuda().float()
            print(image.shape)
            target = torch.squeeze(mask).cuda().long()
            text_x, text_y = text_x.cuda().float(), text_y.cuda().float()
            print(text_x.shape)
            optimizer.zero_grad()

            train_time, output = mymodel(image, text_x)
            batch_loss = criterion1(output, target) + criterion2(train_time, text_y)
            train_dice += dice(output, target, 1)

            batch_loss.backward()
            optimizer.step()
            train_loss += batch_loss.item()

        train_loss /= len(train_loader)
        train_dice /= len(train_loader)

        mymodel.eval()  
        correct = 0
        with torch.no_grad(): 
            for image, mask,valtext_x, valtext_y in tqdm(val_loader, total=len(val_loader)):
                image = image.cuda().float()
                target = torch.squeeze(mask.cuda().long())
                valtext_x, valtext_y = valtext_x.cuda().float(), valtext_y.cuda().float()

                pred_time, output = mymodel(image, valtext_x)
                batch_loss = criterion1(output, target) + criterion2(pred_time, valtext_y)


                pred_time, valtext_y = pred_time[:,0,0], valtext_y[:,0,0]
                correct += ((pred_time-valtext_y*0.9) * (valtext_y*1.2-pred_time) > 0).sum()
                val_dice += dice(output, target, 1)
                val_loss += batch_loss.item()

        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        acc = correct / (len(val_loader) * 8)  ##batch=8

        if acc > maxacc:
            torch.save(mymodel.state_dict(), "time.pkl")
            maxacc = acc

        if val_dice > 0.8:
            for name, value in mymodel.named_parameters():
                if name.split(".")[0] != "sequence": 
                    value.required_grad = False

        print('[%03d/%03d] %2.2f sec(s) train_loss: %3.6f train_dice: %3.6f| val_loss: %3.6f val_dice: %3.6f acc: %3.6f'% \
              (epoch + 1, num_epoch, time.time() - epoch_start_time, train_loss, train_dice,
               val_loss, val_dice, acc))
    return mymodel
