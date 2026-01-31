import torch
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import logging
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import normalize
import seaborn as sns


# validate the algorithm by AUC, accuracy and f1 score on val/test datasets

# [Merged] 使用 Original 版本，因为它包含 tqdm 进度条，体验更好
def algorithm_validate(algorithm, data_loader, writer, epoch, val_type):
    algorithm.eval()
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        softmax = torch.nn.Softmax(dim=1)
        loss = 0
        label_list = []
        output_list = []
        pred_list = []

        # 使用 tqdm 包裹 data_loader
        loader_bar = tqdm(data_loader, desc=f"Validating ({val_type})", leave=False, dynamic_ncols=True)

        for image, label, domain, _ in loader_bar:
            image = image.cuda()
            label = label.cuda().long()

            output = algorithm.predict(image)
            batch_loss = criterion(output, label).item()
            loss += batch_loss

            _, pred = torch.max(output, 1)
            output_sf = softmax(output)

            label_list.append(label.cpu().data.numpy())
            pred_list.append(pred.cpu().data.numpy())
            output_list.append(output_sf.cpu().data.numpy())

            # 在进度条尾部显示当前的 Loss
            loader_bar.set_postfix({'loss': f'{batch_loss:.4f}'})

        # 循环结束，计算指标
        label = [item for sublist in label_list for item in sublist]
        pred = [item for sublist in pred_list for item in sublist]
        output = [item for sublist in output_list for item in sublist]

        acc = accuracy_score(label, pred)
        f1 = f1_score(label, pred, average='macro')

        # 处理可能缺类导致的 AUC 计算报错
        try:
            auc_ovo = roc_auc_score(label, output, average='macro', multi_class='ovo')
        except ValueError:
            auc_ovo = 0.0

        loss = loss / len(data_loader)

        if val_type in ['val', 'test']:
            # 记录到 Tensorboard
            if writer is not None:
                writer.add_scalar('info/{}_accuracy'.format(val_type), acc, epoch)
                writer.add_scalar('info/{}_loss'.format(val_type), loss, epoch)
                writer.add_scalar('info/{}_auc_ovo'.format(val_type), auc_ovo, epoch)
                writer.add_scalar('info/{}_f1'.format(val_type), f1, epoch)

            # 记录到日志文件
            logging.info('{} - epoch: {}, loss: {:.4f}, acc: {:.4f}, auc: {:.4f}, F1: {:.4f}.'.format
                         (val_type, epoch, loss, acc, auc_ovo, f1))

    algorithm.train()
    return auc_ovo, loss


# [New from MaskSiam] 详细分类报告验证
def algorithm_validate_class(algorithm, data_loader, writer, epoch, val_type):
    algorithm.eval()
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        softmax = torch.nn.Softmax(dim=1)
        loss = 0
        label_list = []
        output_list = []
        pred_list = []

        # 使用 tqdm
        loader_bar = tqdm(data_loader, desc=f"Validating Class ({val_type})", leave=False, dynamic_ncols=True)

        for image, label, domain, _ in loader_bar:
            image = image.cuda()
            label = label.cuda().long()

            output = algorithm.predict(image)
            loss += criterion(output, label).item()

            _, pred = torch.max(output, 1)
            output_sf = softmax(output)

            label_list.append(label.cpu().data.numpy())
            pred_list.append(pred.cpu().data.numpy())
            output_list.append(output_sf.cpu().data.numpy())

        label = [item for sublist in label_list for item in sublist]
        pred = [item for sublist in pred_list for item in sublist]
        output = [item for sublist in output_list for item in sublist]

        acc = accuracy_score(label, pred)
        report = classification_report(label, pred, output_dict=False)
        f1 = f1_score(label, pred, average='macro')

        try:
            auc_ovo = roc_auc_score(label, output, average='macro', multi_class='ovo')
        except ValueError:
            auc_ovo = 0.0

        loss = loss / len(data_loader)

        if val_type in ['val', 'test']:
            if writer is not None:
                writer.add_scalar('info/{}_accuracy'.format(val_type), acc, epoch)
                writer.add_scalar('info/{}_loss'.format(val_type), loss, epoch)
                writer.add_scalar('info/{}_auc_ovo'.format(val_type), auc_ovo, epoch)
                writer.add_scalar('info/{}_f1'.format(val_type), f1, epoch)

            logging.info('{} - epoch: {}, loss: {}, acc: {}, auc: {}, F1: {}.'.format
                         (val_type, epoch, loss, acc, auc_ovo, f1))

            logging.info(report)

    algorithm.train()
    return auc_ovo, loss


# [New from MaskSiam] 蒸馏/多头输出验证 (Crucial for MASK_SIAM)
def algorithm_validate_distill(algorithm, data_loader, writer, epoch, val_type):
    algorithm.eval()
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        softmax = torch.nn.Softmax(dim=1)
        loss = 0
        label_list = []
        output_list = []
        pred_list = []

        output_list1 = []
        pred_list1 = []

        output_list2 = []
        pred_list2 = []

        output_list3 = []
        pred_list3 = []

        loader_bar = tqdm(data_loader, desc=f"Validating Distill ({val_type})", leave=False, dynamic_ncols=True)

        for image, label, domain, _ in loader_bar:
            image = image.cuda()
            label = label.cuda().long()

            # MASK_SIAM 返回的是列表 [output1, output2, output3, output_final]
            output = algorithm.predict(image)
            # output[3] 是最终融合输出
            loss += criterion(output[3], label).item()

            _, pred = torch.max(output[3], 1)
            output_sf = softmax(output[3])

            label_list.append(label.cpu().data.numpy())
            pred_list.append(pred.cpu().data.numpy())
            output_list.append(output_sf.cpu().data.numpy())

            _, pred1 = torch.max(output[0], 1)
            output_sf1 = softmax(output[0])
            pred_list1.append(pred1.cpu().data.numpy())
            output_list1.append(output_sf1.cpu().data.numpy())

            _, pred2 = torch.max(output[1], 1)
            output_sf2 = softmax(output[1])
            pred_list2.append(pred2.cpu().data.numpy())
            output_list2.append(output_sf2.cpu().data.numpy())

            _, pred3 = torch.max(output[2], 1)
            output_sf3 = softmax(output[2])
            pred_list3.append(pred3.cpu().data.numpy())
            output_list3.append(output_sf3.cpu().data.numpy())

        label = [item for sublist in label_list for item in sublist]
        pred = [item for sublist in pred_list for item in sublist]
        output = [item for sublist in output_list for item in sublist]

        pred1 = [item for sublist in pred_list1 for item in sublist]
        output1 = [item for sublist in output_list1 for item in sublist]

        pred2 = [item for sublist in pred_list2 for item in sublist]
        output2 = [item for sublist in output_list2 for item in sublist]

        pred3 = [item for sublist in pred_list3 for item in sublist]
        output3 = [item for sublist in output_list3 for item in sublist]

        acc = accuracy_score(label, pred)
        f1 = f1_score(label, pred, average='macro')
        try:
            auc_ovo = roc_auc_score(label, output, average='macro', multi_class='ovo')
        except:
            auc_ovo = 0.0

        acc1 = accuracy_score(label, pred1)
        f1_1 = f1_score(label, pred1, average='macro')
        try:
            auc_ovo1 = roc_auc_score(label, output1, average='macro', multi_class='ovo')
        except:
            auc_ovo1 = 0.0

        acc2 = accuracy_score(label, pred2)
        f1_2 = f1_score(label, pred2, average='macro')
        try:
            auc_ovo2 = roc_auc_score(label, output2, average='macro', multi_class='ovo')
        except:
            auc_ovo2 = 0.0

        acc3 = accuracy_score(label, pred3)
        f1_3 = f1_score(label, pred3, average='macro')
        try:
            auc_ovo3 = roc_auc_score(label, output3, average='macro', multi_class='ovo')
        except:
            auc_ovo3 = 0.0

        loss = loss / len(data_loader)

        if val_type in ['val', 'test']:
            if writer is not None:
                writer.add_scalar('info/{}_accuracy'.format(val_type), acc, epoch)
                writer.add_scalar('info/{}_loss'.format(val_type), loss, epoch)
                writer.add_scalar('info/{}_auc_ovo'.format(val_type), auc_ovo, epoch)
                writer.add_scalar('info/{}_f1'.format(val_type), f1, epoch)

            logging.info('{} - epoch: {}, loss: {:.4f}, acc: {:.4f}, auc: {:.4f}, F1: {:.4f}.'.format
                         (val_type, epoch, loss, acc, auc_ovo, f1))

            logging.info('{} - epoch: {}, loss: {:.4f}, acc1: {:.4f}, auc1: {:.4f}, F1_1: {:.4f}.'.format
                         (val_type, epoch, loss, acc1, auc_ovo1, f1_1))

            logging.info('{} - epoch: {}, loss: {:.4f}, acc2: {:.4f}, auc2: {:.4f}, F1_2: {:.4f}.'.format
                         (val_type, epoch, loss, acc2, auc_ovo2, f1_2))

            logging.info('{} - epoch: {}, loss: {:.4f}, acc3: {:.4f}, auc3: {:.4f}, F1_3: {:.4f}.'.format
                         (val_type, epoch, loss, acc3, auc_ovo3, f1_3))

    algorithm.train()
    return auc_ovo, loss


# [New from MaskSiam] 蒸馏验证 V2
def algorithm_validate_distill_v2(algorithm, data_loader, writer, epoch, val_type):
    algorithm.eval()
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        softmax = torch.nn.Softmax(dim=1)
        loss = 0
        label_list = []
        output_list = []
        pred_list = []

        output_list1 = []
        pred_list1 = []
        output_list2 = []
        pred_list2 = []

        loader_bar = tqdm(data_loader, desc=f"Validating Distill V2 ({val_type})", leave=False, dynamic_ncols=True)

        for image, label, domain, _ in loader_bar:
            image = image.cuda()
            label = label.cuda().long()

            output = algorithm.predict(image)
            loss += criterion(output[0], label).item()

            _, pred = torch.max(output[0], 1)
            output_sf = softmax(output[0])

            label_list.append(label.cpu().data.numpy())
            pred_list.append(pred.cpu().data.numpy())
            output_list.append(output_sf.cpu().data.numpy())

            _, pred1 = torch.max(output[1], 1)
            output_sf1 = softmax(output[1])
            pred_list1.append(pred1.cpu().data.numpy())
            output_list1.append(output_sf1.cpu().data.numpy())

            _, pred2 = torch.max(output[2], 1)
            output_sf2 = softmax(output[2])
            pred_list2.append(pred2.cpu().data.numpy())
            output_list2.append(output_sf2.cpu().data.numpy())

        label = [item for sublist in label_list for item in sublist]
        pred = [item for sublist in pred_list for item in sublist]
        output = [item for sublist in output_list for item in sublist]

        pred1 = [item for sublist in pred_list1 for item in sublist]
        output1 = [item for sublist in output_list1 for item in sublist]

        pred2 = [item for sublist in pred_list2 for item in sublist]
        output2 = [item for sublist in output_list2 for item in sublist]

        acc = accuracy_score(label, pred)
        f1 = f1_score(label, pred, average='macro')
        try:
            auc_ovo = roc_auc_score(label, output, average='macro', multi_class='ovo')
        except:
            auc_ovo = 0.0

        acc1 = accuracy_score(label, pred1)
        f1_1 = f1_score(label, pred1, average='macro')
        try:
            auc_ovo1 = roc_auc_score(label, output1, average='macro', multi_class='ovo')
        except:
            auc_ovo1 = 0.0

        acc2 = accuracy_score(label, pred2)
        f1_2 = f1_score(label, pred2, average='macro')
        try:
            auc_ovo2 = roc_auc_score(label, output2, average='macro', multi_class='ovo')
        except:
            auc_ovo2 = 0.0

        loss = loss / len(data_loader)

        if val_type in ['val', 'test']:
            if writer is not None:
                writer.add_scalar('info/{}_accuracy'.format(val_type), acc, epoch)
                writer.add_scalar('info/{}_loss'.format(val_type), loss, epoch)
                writer.add_scalar('info/{}_auc_ovo'.format(val_type), auc_ovo, epoch)
                writer.add_scalar('info/{}_f1'.format(val_type), f1, epoch)

            logging.info('{} - epoch: {}, loss: {:.4f}, acc: {:.4f}, auc: {:.4f}, F1: {:.4f}.'.format
                         (val_type, epoch, loss, acc, auc_ovo, f1))

            logging.info('{} - epoch: {}, loss: {:.4f}, acc1: {:.4f}, auc1: {:.4f}, F1_1: {:.4f}.'.format
                         (val_type, epoch, loss, acc1, auc_ovo1, f1_1))

            logging.info('{} - epoch: {}, loss: {:.4f}, acc2: {:.4f}, auc2: {:.4f}, F1_2: {:.4f}.'.format
                         (val_type, epoch, loss, acc2, auc_ovo2, f1_2))

    algorithm.train()
    return auc_ovo, loss


# [New from MaskSiam] 双流网络验证
def algorithm_validate_dual(algorithm, data_loader, writer, epoch, val_type):
    algorithm.eval()
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        softmax = torch.nn.Softmax(dim=1)
        loss1 = 0
        loss2 = 0

        label_list = []
        output_list = []
        pred_list = []

        loader_bar = tqdm(data_loader, desc=f"Validating Dual ({val_type})", leave=False, dynamic_ncols=True)

        for image, label, domain, _ in loader_bar:
            image = image.cuda()
            label = label.cuda().long()

            output1, output2 = algorithm.predict(image)
            loss1 += criterion(output1, label).item()
            loss2 += criterion(output2, label).item()

            _, pred1 = torch.max(output1, 1)
            output_sf = softmax(output1)

            label_list.append(label.cpu().data.numpy())
            pred_list.append(pred1.cpu().data.numpy())
            output_list.append(output_sf.cpu().data.numpy())

        label = [item for sublist in label_list for item in sublist]
        pred = [item for sublist in pred_list for item in sublist]
        output = [item for sublist in output_list for item in sublist]

        acc = accuracy_score(label, pred)
        f1 = f1_score(label, pred, average='macro')

        try:
            auc_ovo = roc_auc_score(label, output, average='macro', multi_class='ovo')
        except:
            auc_ovo = 0.0

        loss = (loss1 + loss2) / (2 * len(data_loader))

        if val_type in ['val', 'test']:
            if writer is not None:
                writer.add_scalar('info/{}_accuracy'.format(val_type), acc, epoch)
                writer.add_scalar('info/{}_loss'.format(val_type), loss, epoch)
                writer.add_scalar('info/{}_auc_ovo'.format(val_type), auc_ovo, epoch)
                writer.add_scalar('info/{}_f1'.format(val_type), f1, epoch)

            logging.info('{} - epoch: {}, loss: {:.4f}, acc: {:.4f}, auc: {:.4f}, F1: {:.4f}.'.format
                         (val_type, epoch, loss, acc, auc_ovo, f1))

    algorithm.train()
    return auc_ovo, loss


# [New from MaskSiam] 评估均值和方差
def algorithm_eval_mu(algorithm, data_loader, epoch, val_type):
    with torch.no_grad():
        i = 0
        for image, label, domain, _ in tqdm(data_loader, desc="Eval Mu", leave=False):
            image = image.cuda()
            x = image
            img_mean = x.mean(dim=[2, 3], keepdim=True)  # B,C,1,1 mu
            img_std = x.std(dim=[2, 3], keepdim=True) + 1e-7  # B,C,1,1 sigma
            if i == 0:
                mu = img_mean.cpu().numpy()
                sigma = img_std.cpu().numpy()
            else:
                mu = np.concatenate((mu, img_mean.detach().cpu().numpy()), axis=0)
                sigma = np.concatenate((sigma, img_std.detach().cpu().numpy()), axis=0)
            i += 1

    return mu, sigma


# [New from MaskSiam] TSNE 特征提取
def algorithm_eval_tsne(algorithm, data_loader, epoch, val_type):
    with torch.no_grad():
        i = 0
        for image, label, domain, _ in tqdm(data_loader, desc="Eval TSNE", leave=False):
            image = image.cuda()
            x = image
            img_mean = x.mean(dim=[2, 3], keepdim=True)  # B,C,1,1 mu
            img_std = x.std(dim=[2, 3], keepdim=True) + 1e-7  # B,C,1,1 sigma
            label = label.cuda().long()

            output = algorithm.predict(image)

            if i == 0:
                feats = output.cpu().numpy()
                labels = label.cpu().numpy()
            else:
                feats = np.concatenate((feats, output.detach().cpu().numpy()), axis=0)
                labels = np.concatenate((labels, label.detach().cpu().numpy()), axis=0)
            i += 1

    return feats, labels


# [New from MaskSiam] 热力图特征提取
def algorithm_eval_heat(algorithm, data_loader, epoch, val_type):
    with torch.no_grad():
        i = 0
        for image, label, domain, _ in tqdm(data_loader, desc="Eval Heatmap", leave=False):
            image = image.cuda()
            x = image
            label = label.cuda().long()

            # 注意：这里假设 algorithm.predict 返回4个值
            output, iw_loss, feat_, feat = algorithm.predict(image)
            B, N, C = feat.shape[:]
            feat = feat.view(B, int(np.sqrt(N)), -1, C)
            feat = feat.mean(dim=-1)

            if i == 0:
                feats = feat.cpu().numpy()
            else:
                feats = np.concatenate((feats, feat.detach().cpu().numpy()), axis=0)
            i += 1

    return feats