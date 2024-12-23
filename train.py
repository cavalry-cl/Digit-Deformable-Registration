import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from network import DeformableRegistrationNet
from dataset import DigitDataset
import torch.nn.functional as F

import loss
import os

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# train_raw_ds = datasets.MNIST(root='.', download=True, train=True,
train_raw_ds = datasets.MNIST(root='.', train=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                   ]))
test_raw_ds = datasets.MNIST(root='.', train=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                   ]))

train_dataloader = DataLoader(train_raw_ds, batch_size=16, shuffle=True, num_workers=1)

target_folder = "targets"
save_dir = "models/targets"

middle_channels = 32
epoches = 150


# model = torch.load("/public/home/CS172/tengzhh2022-cs172/assignments/assignment2/registration/models/DRN64_epoch299_model.bin")
# model = DeformableRegistrationNet(img_channels=2, middle_channels=middle_channels)
model = DeformableRegistrationNet(img_channels=1, middle_channels=middle_channels)
model.to(device=device)

# optimizer = torch.load("/public/home/CS172/tengzhh2022-cs172/assignments/assignment2/registration/models/DRN64_epoch299_opt.bin")
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.00001)
simlossfunc = loss.SimLoss(target_folder).loss_interpolate_fix
log_steps = 250

def save_model(model, optimzer, save_dir, name, epoch):
    torch.save(model,os.path.join(save_dir, f"{name}_epoch{epoch}_model.bin"))
    torch.save(optimzer,os.path.join(save_dir, f"{name}_epoch{epoch}_opt.bin"))

for epoch in range(0, epoches):
    step = 0
    total_sim_loss , total_cls_loss, total_loss = 0, 0, 0
    for img, labels in train_dataloader:
        optimizer.zero_grad()
        img = img.to(device=device)
        labels = labels.to(device=device)
        model_output, logits = model(img)
        sim_loss = simlossfunc(model_output, labels, img)
        cls_loss = F.cross_entropy(logits, labels)
        c = sim_loss.detach()
        l = sim_loss + c * 0.1 * cls_loss

        total_loss += l
        total_sim_loss += sim_loss
        total_cls_loss += cls_loss
        step += 1
        if step % log_steps == 0:
            print(f'step{step} loss={total_loss/log_steps} sim_loss={total_sim_loss/log_steps} cls_loss={total_cls_loss/log_steps}')    
            total_sim_loss , total_cls_loss, total_loss = 0, 0, 0
        l.backward()
        optimizer.step()
    if epoch < 10 or epoch % 25 == 24:
        save_model(model, optimizer, save_dir, f"DRN{middle_channels}", epoch)
