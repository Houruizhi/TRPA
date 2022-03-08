import os
import shutil
from tqdm import tqdm

import torch
import torch.optim as optim
import numpy as np
import tensorboardX

from src.config import get_cfg_defaults, get_args_from_parser
from src.utils import mkdir, load_model, clear_result_dir
from src.metrics import batch_PSNR
from src.models.condrefinenet import CondRefineNetDilated
from src.utils_train import dataloader, loss_function

def main(cfg):
    # make the dataloader
    loader_train = dataloader(cfg, train=True)

    mkdir(cfg.TRAIN.OUT_DIR)
    MIDDLE_RESULT_DIR = os.path.join(cfg.TRAIN.OUT_DIR, 'middle_res')
    mkdir(MIDDLE_RESULT_DIR)

    summary_path = os.path.join(cfg.TRAIN.OUT_DIR, 'tensorboard')

    if not cfg.TRAIN.RESUME:
        clear_result_dir(MIDDLE_RESULT_DIR)
        clear_result_dir(summary_path)

    tensorboard_logger = tensorboardX.SummaryWriter(log_dir=summary_path)

    scoreNet = CondRefineNetDilated(6,6,128, residual=False)
    scoreNet = scoreNet.cuda()
    optimizer = torch.optim.Adam(scoreNet.parameters(), lr=cfg.SOLVER.LEARNING_RATE, weight_decay=0.000,
                            betas=(0.9, 0.999), amsgrad=False)

    epoch_start = 0
    if cfg.TRAIN.RESUME:
        chp = torch.load(os.path.join(cfg.TRAIN.RESUME_CHP_DIR, 'net.pth'))
        scoreNet = load_model(scoreNet, chp['weights'])
        optimizer.load_state_dict(chp['optimizer'])
        epoch_start = chp['epoch']+1
    
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
        list(range(cfg.SOLVER.MILESTONE, cfg.TRAIN.NUM_EPOCHS, cfg.SOLVER.MILESTONE)), gamma = cfg.SOLVER.LEARNING_RATE_DECAY)

    if cfg.TRAIN.PARALLE:
        scoreNet = torch.nn.DataParallel(scoreNet)

    max_sigma = 1/3.
    step = 1
    for epoch in range(epoch_start, cfg.TRAIN.NUM_EPOCHS): 
        avg_loss = 0  
        with tqdm(total=len(loader_train), desc='Epoch: [%d/%d], lr: [%.6f]'%\
            (epoch+1, cfg.TRAIN.NUM_EPOCHS, optimizer.param_groups[0]["lr"]), miniters=1) as t:
            for i, (_, batch) in enumerate(loader_train):
                scoreNet.train()
                scoreNet.zero_grad()
                optimizer.zero_grad()

                loss, denoised, perturbed_samples = loss_function(scoreNet, batch, max_sigma)
                loss.backward()

                optimizer.step()
                
                avg_loss += loss.item()
                tensorboard_logger.add_scalar('loss', loss, global_step=step)
                batch_psnr = 0
                with torch.no_grad():
                    batch_psnr = batch_PSNR(denoised, perturbed_samples)

                t.set_postfix_str("Batch Loss: %.4f, Batch PSNR: %.4f, average loss: %.4f" % (loss.item(), batch_psnr,  avg_loss/(i+1)))
                t.update()
                step += 1
                # break

            scheduler.step()

            checkpoint_path = os.path.join(cfg.TRAIN.OUT_DIR, f'net.pth')
            chp = {
                'epoch': epoch, 
                'weights': scoreNet.state_dict(),
                'optimizer': optimizer.state_dict()
                }
            torch.save(chp, checkpoint_path)

if __name__ == '__main__':
    cfg = get_cfg_defaults()
    args = get_args_from_parser()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in cfg.SYSTEM.GPU_IDS[:cfg.SYSTEM.NUM_GPUS]])
    
    print(cfg)
    main(cfg)