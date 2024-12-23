from pytorch_msssim import ssim
from dataset import TrueDigitImages
import torch


def interpolate(model_output, input):
    N = input.shape[-2]
    M = input.shape[-1]
    pos = torch.tensor([[[[(j+0.5)/(M/2)-1, (i+0.5)/(N/2)-1] for j in range(M)] for i in range(N)]]).to(device=model_output.device) # (1,N,M,2)
    output = model_output + pos
    res = torch.nn.functional.grid_sample(input, grid=output, mode='bilinear', padding_mode='zeros', align_corners=False) 
    return res

class SimLoss:
    def __init__(self, path='targets'):
        self.tdi = TrueDigitImages(path)

    def loss_raw(self, pred, label):
        '''
        output: B * C * N * N
        label: B
        '''
        B = label.shape[0]
        target = torch.stack([self.tdi[label[i]] for i in range(B)]).to(device=pred.device)
        return 1 - ssim(pred, target, data_range=1.0, size_average=True)
    
    def loss_raw_target(self, pred, target):
        '''
        output: B * C * N * N
        '''
        # B = label.shape[0]
        # target = torch.stack([self.tdi[label[i]] for i in range(B)]).to(device=pred.device)
        return 1 - ssim(pred, target, data_range=1.0, size_average=True)

    def loss_interpolate_fix(self, model_output, label, input):
        return self.loss_raw(interpolate(model_output, input), label)

    def loss_interpolate_target(self, model_output, target, input):
        return self.loss_raw_target(interpolate(model_output, input), target)
