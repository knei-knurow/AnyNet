import yaml
with open("config.yml", "r") as config_file:
    cfg = yaml.safe_load(config_file)

host = cfg["host"]
port = cfg["port"]
debug = cfg["debug"]
sources = cfg["sources"]
path = cfg["path"]
is_remote = cfg["remote"]

with open("rovercamera/config/stereo.yml", "r") as rovercamera_config_file:
    rovercamera_cfg = yaml.unsafe_load(rovercamera_config_file)

import rovercamera
import argparse
import os
from rovercamera import RoverCamera
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import time
from dataloader import listflowfile as lt
from dataloader import SecenFlowLoader as DA
import utils.logger as logger
import models.anynet

parser = argparse.ArgumentParser(description='AnyNet with Flyingthings3d')
parser.add_argument('--maxdisp', type=int, default=192, help='maxium disparity')
parser.add_argument('--loss_weights', type=float, nargs='+', default=[0.25, 0.5, 1., 1.])
parser.add_argument('--maxdisplist', type=int, nargs='+', default=[12, 3, 3])
parser.add_argument('--datapath', default='dataset/',
                    help='datapath')
parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs to train')
parser.add_argument('--train_bsize', type=int, default=6,
                    help='batch size for training (default: 12)')
parser.add_argument('--test_bsize', type=int, default=4,
                    help='batch size for testing (default: 8)')
parser.add_argument('--save_path', type=str, default='results/pretrained_anynet',
                    help='the path of saving checkpoints and log')
parser.add_argument('--resume', type=str, default=None,
                    help='resume path')
parser.add_argument('--lr', type=float, default=5e-4,
                    help='learning rate')
parser.add_argument('--with_spn', action='store_true', help='with spn network or not')
parser.add_argument('--print_freq', type=int, default=5, help='print frequence')
parser.add_argument('--init_channels', type=int, default=1, help='initial channels for 2d feature extractor')
parser.add_argument('--nblocks', type=int, default=2, help='number of layers in each stage')
parser.add_argument('--channels_3d', type=int, default=4, help='number of initial channels of the 3d network')
parser.add_argument('--layers_3d', type=int, default=4, help='number of initial layers of the 3d network')
parser.add_argument('--growth_rate', type=int, nargs='+', default=[4,1,1], help='growth rate in the 3d network')
parser.add_argument('--spn_init_channels', type=int, default=8, help='initial channels for spnet')


args = parser.parse_args()


def main():
    global args

    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    log = logger.setup_logger(args.save_path + '/training.log')
    for key, value in sorted(vars(args).items()):
        log.info(str(key) + ': ' + str(value))

    model = models.anynet.AnyNet(args)
    model = nn.DataParallel(model).cuda()

    log.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    if os.path.isfile(args.resume):
        log.info("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        log.info("=> loaded checkpoint '{}' (epoch {})"
                    .format(args.resume, checkpoint['epoch']))
    else:
        log.error("=> No checkpoint found! ")
        exit()

    camLeft = RoverCamera("left", rovercamera_cfg)
    camRight = RoverCamera("right", rovercamera_cfg)

    imLeft = camLeft.get_frame()
    imRight = camRight.get_frame()
    
    print(imLeft.shape)

    print(test(imLeft, imRight, model))
 
    

def test(imgL, imgR, model):

    stages = 3 + args.with_spn

    model.eval()
    imgL = torch.from_numpy(imgL).float().cuda()
    imgR = torch.from_numpy(imgR).float().cuda()

    with torch.no_grad():
        outputs = model(imgL, imgR)
        return outputs[stages - 1]

        # for x in range(stages):
            # output = torch.squeeze(outputs[x], 1)
            # output = output[:, 4:, :]
    

 

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    main()
