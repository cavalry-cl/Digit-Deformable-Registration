import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import loss

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

test_raw_ds = datasets.MNIST(root='.', train=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                    #    transforms.Normalize((0.1,), (0.3,))
                   ]))
test_dataloader = DataLoader(test_raw_ds)

print('middle=32')
model_path = "models/targets/DRN32_epoch99_model.bin"
target_path = "targets"
model = torch.load(model_path,map_location=lambda storage, loc: storage)
simlossfunc = loss.SimLoss(target_path).loss_interpolate_fix
model.to(device=device)
model.eval()

score = [0 for _ in range(10)]
cnt = [0 for _ in range(10)]

step=0

for img, label in test_dataloader:
    img = img.to(device=device)
    label = label.to(device=device)
    model_output, logits = model(img)
    sim_score = 1 - simlossfunc(model_output, label, img).detach().item()
    score[label[0]] += sim_score
    cnt[label[0]] += 1
    step += 1
    if step % 500 == 0:print(step)
    
for digit in range(10):
    print(f'SSIM score of {digit} = {score[digit]/cnt[digit]}')
ss=0
cc=0
for c in cnt:cc+=c
for s in score:ss+=s
print(f'Total SSIM score = {ss/cc}')