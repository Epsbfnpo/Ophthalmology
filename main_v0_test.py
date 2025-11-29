import algorithms_v0 as algorithms
import os
from utils.validate import *
from utils.args import *
from utils.misc import *
from dataset.data_manager import get_dataset
from tqdm import tqdm
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from torch.cuda.amp import GradScaler, autocast
from torchvision.utils import save_image

if __name__ == "__main__":

    args = get_args()
    cfg = setup_cfg(args)
    log_path = os.path.join('./result/fundusaug', cfg.OUTPUT_PATH)
       
    # init
    train_loader, val_loader, test_loader, dataset_size = get_dataset(cfg)
    writer = init_log_test(args, cfg, log_path, len(train_loader), dataset_size)
    algorithm_class = algorithms.get_algorithm_class(cfg.ALGORITHM)
    algorithm = algorithm_class(cfg.DATASET.NUM_CLASSES, cfg)
    algorithm.cuda()
    
    # train
    # iterator = tqdm(range(cfg.EPOCHS))
    
    # amp_grad_scaler = GradScaler()
    
    # if args.algorithm !='GDRNet_MASK_SIAM1':
    #     if cfg.DG_MODE =='DG':
    #         scheduler = get_scheduler2(algorithm.optimizer, cfg.EPOCHS)
    #     else:
    #         scheduler = get_scheduler(algorithm.optimizer, cfg.EPOCHS)

        
    # best_performance = 0.0
    # for i in iterator:
        
    #     # if step == q_steps and quant==1:            #if quantization
    #     #     modules_to_replace = find_modules_to_quantize(algorithm, args_q.quan)
    #     #     print(modules_to_replace)
    #     #     algorithm = replace_module_by_names(algorithm, modules_to_replace)
    #     #     algorithm.to(device)
    #     #     swad = None
    #     #     if hparams["swad"]:
    #     #         swad_algorithm = swa_utils.AveragedModel(algorithm)
    #     #         swad_cls = getattr(swad_module, hparams["swad"])
    #     #         swad = swad_cls(evaluator, **hparams.swad_kwargs)


    #     epoch = i + 1
        
    #     #  same randommatrix each 10 epoch
    #     if cfg.DG_MODE =='DG':
    #         if (epoch == 1 or epoch % 10 == 0) and hasattr(algorithm, 'change_random_matrix'):
                
    #             algorithm.change_random_matrix()#.cuda()
    #     else:
    #         if (epoch == 1 or epoch % 1 == 0) and hasattr(algorithm, 'change_random_matrix'):
                
    #             algorithm.change_random_matrix()#.cuda()
            
    #     # print((epoch) % 1)
    #     # print( (epoch == 1 or epoch % 1 == 0))
            
    #     loss_avg = LossCounter()
    #     # if cfg.FREQ:
    #     for i, imgs in enumerate(train_loader):
    #         if cfg.TRANSFORM.FREQ: 
    #             # print("Freq")
    #             image, image_freq, image_freq2, mask, label, domain, img_index=imgs
    #         else:
    #             image, mask, label, domain, img_index=imgs
                
    #         # print(image)

    #         # if epoch==1:
    #         #     sample_path1 = f'./samples/new'
    #         #     sample_path2 = f'./samples/freq'

    #         #     os.makedirs(sample_path1, exist_ok=True)
    #         #     os.makedirs(sample_path2, exist_ok=True)

    #         #     for i, image_ in enumerate(image):
    #         #         # Move the image tensor to CUDA and save it with the index as the filename
    #         #         image_ = image_.to("cuda")
    #         #         # save_image(image, f"output/visualize/{folder_name}/image_{i}.png")
    #         #         save_image(image_, f"{sample_path1}/image_{i}.png")            

    #         #     for i, image_2 in enumerate(image_freq):
    #         #         # Move the image tensor to CUDA and save it with the index as the filename
    #         #         image_2 = image_2.to("cuda")
    #         #         # save_image(image, f"output/visualize/{folder_name}/image_{i}.png")
    #         #         save_image(image_2, f"{sample_path2}/image_{i}.png")    
                    
            
            
    #         iters = epoch * len(train_loader) + i
            
    #         if args.algorithm=='GDRNet_MASK_SIAM1':
    #             get_scheduler_siam(algorithm.optimizer, cfg.LEARNING_RATE, epoch, cfg.EPOCHS)

    #         # algorithm.change_random_matrix()
    #         algorithm.train()
            
                
    #         if cfg.TRANSFORM.FREQ: 
    #             minibatch = [image.cuda(), image_freq.cuda(), image_freq2.cuda(), mask.cuda(), label.cuda().long(), domain.cuda().long()]
    #         else:
    #             minibatch = [image.cuda(), mask.cuda(), label.cuda().long(), domain.cuda().long()]
    #         loss_dict_iter = algorithm.update(minibatch)  
    #         algorithm.update_ema_model(iters)  
    #         loss_avg.update(loss_dict_iter['loss'])

    #     alpha = algorithm.update_epoch(epoch)
    #     weight_cls = algorithm.update_epoch_weight(epoch)
        
    #     if args.algorithm=='GDRNet_MASK_SIAM':
    #         if cfg.DG_MODE =='DG':
    #             update_writer_siam_v2(writer, epoch, scheduler, loss_dict_iter)
    #         else:
    #             update_writer(writer, epoch, scheduler, loss_avg)
    #         scheduler.step()
    #     else:
    #         update_writer(writer, epoch, scheduler, loss_avg)
    #         scheduler.step()
            
    #     # update_writer(writer, epoch, scheduler, loss_avg)
    #     # scheduler.step()

    #     # validation
    #     if epoch % cfg.VAL_EPOCH == 0:
    #         val_auc, test_auc = algorithm.validate(val_loader, test_loader, writer)
    #         if val_auc > best_performance and epoch > cfg.EPOCHS * 0.3:
    #             best_performance = val_auc
    #             algorithm.save_model(log_path)
    
    algorithm.load_model(log_path)
    _, test_auc = algorithm.validate_class(val_loader, test_loader, writer)
    # os.mknod(os.path.join(log_path, 'done'))
    writer.close()
