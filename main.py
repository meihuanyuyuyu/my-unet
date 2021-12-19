import torch
from torch.nn import parameter
from torch.optim import optimizer
from torch.optim.optimizer import Optimizer
from utils import *
from tqdm import tqdm
from torch.utils.data import DataLoader, dataloader
from model import attention_u_net, init_weight, unet,residual_unet
import warnings
import torch.nn.functional as F
from config import *
from torch.utils.tensorboard import SummaryWriter


writer = SummaryWriter()
warnings.filterwarnings('ignore')
def main(i,re,train_t:list,val_t:list,model:nn.Module,e:int):
    data = EM_mydataset(root_path,[])
    train_set,val_set = kfold_crossval(data,k,i,train_t,val_t)
    train_sampled = DataLoader(train_set,batch_size=batch_size,shuffle=True,num_workers=1)
    val_sampled = DataLoader(val_set,batch_size=batch_size,shuffle=True,num_workers=1)
    net =model().to(device)
    net.apply(init_weight)
    opt = torch.optim.Adam(net.parameters())
    criterion = F.cross_entropy
    if net.__class__.__name__== 'attention_u_net':
        model_dir = f'model_parameters/au-net_model/{i}_{re}_EM_model.pt'
        json_dir = 'result/EMstacks/attention json'
    if net.__class__.__name__ =='unet':
        model_dir = f'model_parameters/unet/{i}_{re}_EM_model.pt'
        json_dir = 'result/EMstacks/unet_json'
    if net.__class__.__name__ =='residual_unet':
        model_dir = f'model_parameters/r_unet/{i}_{re}_EM_model.pt'
        json_dir = 'result/EMstacks/R2 u-net'
    val_f1s = []
    losses = []
    f1s = []
    bar = tqdm(range(e))
    for epoch in bar:
        loss,f1 = train(net,train_sampled,criterion,opt,device=device)
        bar.set_description(f'training loss:{loss},f1 score:{f1}')
        val_f1 = val(i,net,val_sampled,device=device)
        bar.set_description(f'val_f1 {val_f1}')
        losses.append(loss)
        val_f1s.append(val_f1)
        f1s.append(f1)
        if epoch % 100 ==0:
            torch.save(net.state_dict(),model_dir)
    save_result_json(json_dir,i,re,losses,val_f1s,f1s)
    
    



if __name__ == '__main__':
  
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for i in range(3,5):
        main(i,0,[Elastictransform(10,150),Myrandom_resize([1000,1000]),Myrandomcrop((512,512)),Myrandomrotation(),Myrandomflip(),Mynormalize()],
    [Myrandom_resize([1000,1000]),My_center_crop([512,512]),Mynormalize()],residual_unet,800)
