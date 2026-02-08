import algorithms
import os
from utils.validate import *
from utils.args import *
from utils.misc import *
from dataset.data_manager import get_dataset
from tqdm import tqdm
from quan.utils import find_modules_to_quantize, replace_module_by_names
from utils import swa_utils

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import yaml,munch

if __name__ == "__main__":

    args = get_args()
    cfg = setup_cfg(args)
    log_path = os.path.join('./result/fundusaug', cfg.OUTPUT_PATH)
       
########### For Quantization Training
    with open("configs/config_q.yaml") as yaml_file:
        cfg_q = yaml.safe_load(yaml_file)
        cfg_q = munch.munchify(cfg_q)
    print(cfg_q)       


    # # seed
    # random.seed(args.seed)
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # torch.backends.cudnn.deterministic = args.deterministic
    # torch.backends.cudnn.benchmark = not args.deterministic


    # init
    train_loader, val_loader, test_loader, dataset_size = get_dataset(cfg)
    writer = init_log(args, cfg, log_path, len(train_loader), dataset_size)
    algorithm_class = algorithms_new.get_algorithm_class(cfg.ALGORITHM)
    algorithm = algorithm_class(cfg.DATASET.NUM_CLASSES, cfg)
    algorithm.cuda()
    
    # train
    iterator = tqdm(range(cfg.EPOCHS))
    if args.algorithm !='GDRNet_DUAL_MASK':
        scheduler1 = get_scheduler(algorithm.optimizer1, cfg.EPOCHS)
        scheduler2 = get_scheduler(algorithm.optimizer2, cfg.EPOCHS)

    best_performance = 0.0
    for i in iterator:  ## 1
        epoch = i + 1
        
        if args.algorithm=='GDRNet_DUAL_MASK':
            get_scheduler_siam(algorithm.optimizer1, cfg.LEARNING_RATE, epoch)
            get_scheduler_siam(algorithm.optimizer2, cfg.LEARNING_RATE, epoch)

        if epoch == args.q_steps and args.quant==1:            #if quantization
            modules_to_replace = find_modules_to_quantize(algorithm.network1, cfg_q.quan)
            print(modules_to_replace)
            algorithm.network1 = replace_module_by_names(algorithm.network1, modules_to_replace)
            # algorithm.to(device)
            algorithm.cuda()
            swad = None
            if args.swad:
                swad_algorithm = swa_utils.AveragedModel(algorithm)
                swad_cls = getattr(swad_module, hparams["swad"])
                swad = swad_cls(evaluator, **hparams.swad_kwargs)


        loss_avg = LossCounter()
        for image, mask, label, domain, img_index in train_loader:


            algorithm.train()
            minibatch = [image.cuda(), mask.cuda(), label.cuda().long(), domain.cuda().long()]
            loss_dict_iter = algorithm.update(minibatch)   
            loss_avg.update(loss_dict_iter['loss'])

        alpha = algorithm.update_epoch(epoch)
        if args.algorithm=='GDRNet_DUAL_MASK':
            update_writer_siam(writer, epoch,  loss_avg)
        else:
            update_writer(writer, epoch, scheduler1, loss_avg)
            scheduler1.step()
            scheduler2.step()


        # validation
        if epoch % cfg.VAL_EPOCH == 0:
            val_auc, test_auc = algorithm.validate(val_loader, test_loader, writer)
            if val_auc > best_performance and epoch > cfg.EPOCHS * 0.3:
                best_performance = val_auc
                algorithm.save_model(log_path)
    
    algorithm.renew_model(log_path)
    _, test_auc = algorithm.validate(val_loader, test_loader, writer)
    os.mknod(os.path.join(log_path, 'done'))
    writer.close()
