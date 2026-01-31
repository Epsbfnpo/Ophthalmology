"""
This code is partially borrowed from https://github.com/HobbitLong/SupContrast
"""
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np




def JS_Divergence(output_clean, output_aug1, output_aug2, output_aug3):
    p_clean, p_aug1 = F.softmax(output_clean, dim=1), F.softmax(output_aug1, dim=1)
    p_aug2, p_aug3 = F.softmax(output_aug2, dim=1), F.softmax(output_aug3, dim=1)
    p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2 + p_aug3) / 4., 1e-7, 1).log()
    loss_ctr = (F.kl_div(p_mixture, p_clean, reduction='batchmean') + F.kl_div(p_mixture, p_aug1,
                                                                               reduction='batchmean') + F.kl_div(
        p_mixture, p_aug2, reduction='batchmean') + F.kl_div(p_mixture, p_aug3, reduction='batchmean')) / 4.
    return loss_ctr


# Our loss function
class DahLoss(nn.Module):
    def __init__(self, max_iteration, training_domains, beta = 0.8, scaling_factor = 4, alpha = 1, temperature = 0.07) -> None:
        super(DahLoss, self).__init__()
        self.max_iteration = max_iteration
        self.training_domains = training_domains
        self.alpha = alpha
        self.beta = beta
        self.scaling_factor = scaling_factor
        self.temperature = temperature
        
        self.domain_num_dict = {'MESSIDOR': 1744,
                                'IDRID': 516,
                                'DEEPDR': 2000,
                                'FGADR': 1842,
                                'APTOS': 3662,
                                'RLDR': 1593}
        
        self.label_num_dict = {'MESSIDOR': [1016, 269, 347, 75, 35],
                                'IDRID': [175, 26, 163, 89, 60],
                                'DEEPDR': [917, 214, 402, 353, 113],
                                'FGADR': [100, 211, 595, 646, 286],
                                'APTOS': [1804, 369, 999, 192, 294],
                                'RLDR': [165, 336, 929, 98, 62]}

        self.domain_prob, self.label_prob = self.get_domain_label_prob()
        self.domain_prob, self.label_prob = self.multinomial_soomthing(self.domain_prob, self.label_prob, self.beta)

        self.UnsupLoss = SupConLoss(temperature = self.temperature, reduction='none')
        self.SupLoss = nn.CrossEntropyLoss(reduction='none')

    def get_domain_label_prob(self):
        source_domain_num_list = torch.Tensor([self.domain_num_dict[domain] for domain in self.training_domains])
        source_domain_num = torch.sum(source_domain_num_list)
        domain_prob = source_domain_num_list / source_domain_num

        label_num_list = torch.Tensor([self.label_num_dict[domain] for domain in  self.training_domains]).sum(dim=0)
        label_num = torch.sum(label_num_list)
        label_prob = label_num_list / label_num

        return domain_prob.cuda(), label_prob.cuda()

    def multinomial_soomthing(self, domain_prob, label_prob, beta = 0.8):
        domain_prob = torch.pow(domain_prob, beta)
        label_prob = torch.pow(label_prob, beta)

        domain_prob = domain_prob / torch.sum(domain_prob)
        label_prob = label_prob / torch.sum(label_prob)

        return domain_prob, label_prob

    def get_weights(self, labels, domains):
        domain_prob = torch.index_select(self.domain_prob, 0, domains).cuda()
        domain_weight = 1 / domain_prob
        class_prob = torch.index_select(self.label_prob, 0, labels).cuda()
        class_weight = 1 / class_prob

        return domain_weight, class_weight
                            
    def forward(self, output, features, labels, domains):
        
        domain_weight, class_weight = self.get_weights(labels, domains)

        loss_dict = {}

        features_ori, features_new = features

        loss_sup = 0

        for op_item in output:
            loss_sup += self.SupLoss(op_item, labels)            

        features_multi = torch.stack([features_ori, features_new], dim = 1)
        features_multi = F.normalize(features_multi, p=2, dim=2)      
        
        loss_unsup = torch.mean(self.UnsupLoss(features_multi))
        loss_sup = torch.mean(loss_sup * class_weight * domain_weight) / (torch.mean(domain_weight) * torch.mean(class_weight))

        loss = (1 - self.alpha) * loss_sup + self.alpha * loss_unsup / self.scaling_factor

        loss_dict['loss'] = loss.item()
        loss_dict['loss_sup'] = loss_sup.item()
        loss_dict['loss_unsup'] = loss_unsup.item()
        
        return loss, loss_dict

    def update_alpha(self, iteration):
        self.alpha = 1 - iteration / self.max_iteration
        return self.alpha



# Our loss function
class DahLoss_Mask(nn.Module):
    def __init__(self, max_iteration, training_domains, beta = 0.8, scaling_factor = 4, alpha = 1, temperature = 0.07) -> None:
        super(DahLoss_Mask, self).__init__()
        self.max_iteration = max_iteration
        self.training_domains = training_domains
        self.alpha = alpha
        self.beta = beta
        self.scaling_factor = scaling_factor
        self.temperature = temperature
        
        self.domain_num_dict = {'MESSIDOR': 1744,
                                'IDRID': 516,
                                'DEEPDR': 2000,
                                'FGADR': 1842,
                                'APTOS': 3662,
                                'RLDR': 1593}
        
        self.label_num_dict = {'MESSIDOR': [1016, 269, 347, 75, 35],
                                'IDRID': [175, 26, 163, 89, 60],
                                'DEEPDR': [917, 214, 402, 353, 113],
                                'FGADR': [100, 211, 595, 646, 286],
                                'APTOS': [1804, 369, 999, 192, 294],
                                'RLDR': [165, 336, 929, 98, 62]}

        self.domain_prob, self.label_prob = self.get_domain_label_prob()
        self.domain_prob, self.label_prob = self.multinomial_soomthing(self.domain_prob, self.label_prob, self.beta)

        self.UnsupLoss = SupConLoss(temperature = self.temperature, reduction='none')
        self.SupLoss = nn.CrossEntropyLoss(reduction='none')

    def get_domain_label_prob(self):
        source_domain_num_list = torch.Tensor([self.domain_num_dict[domain] for domain in self.training_domains])
        source_domain_num = torch.sum(source_domain_num_list)
        domain_prob = source_domain_num_list / source_domain_num

        label_num_list = torch.Tensor([self.label_num_dict[domain] for domain in  self.training_domains]).sum(dim=0)
        label_num = torch.sum(label_num_list)
        label_prob = label_num_list / label_num

        return domain_prob.cuda(), label_prob.cuda()

    def multinomial_soomthing(self, domain_prob, label_prob, beta = 0.8):
        domain_prob = torch.pow(domain_prob, beta)
        label_prob = torch.pow(label_prob, beta)

        domain_prob = domain_prob / torch.sum(domain_prob)
        label_prob = label_prob / torch.sum(label_prob)


        return domain_prob, label_prob

    def get_weights(self, labels, domains):
        domain_prob = torch.index_select(self.domain_prob, 0, domains).cuda()
        domain_weight = 1 / domain_prob
        class_prob = torch.index_select(self.label_prob, 0, labels).cuda()
        class_weight = 1 / class_prob

        return domain_weight, class_weight
                            
    def forward(self, output, features, labels, domains):
        
        domain_weight, class_weight = self.get_weights(labels, domains)

        loss_dict = {}

        output,  output_new_masked, output_new_masked_c = output
        
        
        features_ori, features_new = features

        loss_sup = 0


        # for op_item in output:
        loss_sup += self.SupLoss(output, labels)            
        
        # output_sofamax =  F.log_softmax(output, 1)

        # output_new_masked_sofamax =  F.log_softmax(output_new_masked, 1)

        # loss_sup += self.SupLoss(output_new_masked, output_sofamax.detach())            
        # loss_sup += self.SupLoss(output_new_masked_c, output.detach())            
        # loss_sup += self.SupLoss(output_new_masked_c, output_new_masked.detach())            

        temperature=1.0

        loss_sup_masked = - 1  * (F.softmax(output / temperature, 1).detach() * \
                                                F.log_softmax(output_new_masked / temperature, 1)).sum() / \
                                 output.size()[0]

        loss_sup_masked_c = - 1  * (F.softmax(output / temperature, 1).detach() * \
                                                F.log_softmax(output_new_masked_c / temperature, 1)).sum() / \
                                 output.size()[0]
                                 
        # loss_sup_kd = - 1 * (F.softmax(output_new_masked / temperature, 1).detach() * \
        #                                             F.log_softmax(output_new_masked_c / temperature, 1)).sum() / \
        #                              output.size()[0]
                                     
                    
        loss_sup +=  (loss_sup_masked+ loss_sup_masked_c ) / 2    
        features_multi = torch.stack([features_ori, features_new], dim = 1)
        features_multi = F.normalize(features_multi, p=2, dim=2)      
        
        loss_unsup = torch.mean(self.UnsupLoss(features_multi))
        loss_sup = torch.mean(loss_sup * class_weight * domain_weight) / (torch.mean(domain_weight) * torch.mean(class_weight))

        loss = (1 - self.alpha) * loss_sup / 2 + self.alpha * loss_unsup / self.scaling_factor

        loss_dict['loss'] = loss.item()
        loss_dict['loss_sup'] = loss_sup.item()
        loss_dict['loss_unsup'] = loss_unsup.item()
        
        return loss, loss_dict

    def update_alpha(self, iteration):
        self.alpha = 1 - iteration / self.max_iteration
        return self.alpha
    
    
# Our loss function
class DahLoss_Dual(nn.Module):
    def __init__(self, max_iteration, training_domains, beta = 0.8, scaling_factor = 4, alpha = 1.0, temperature = 0.07) -> None:
        super(DahLoss_Dual, self).__init__()
        self.max_iteration = max_iteration
        self.training_domains = training_domains
        self.alpha = alpha
        self.beta = beta
        self.scaling_factor = scaling_factor
        self.temperature = temperature
        
        self.domain_num_dict = {'MESSIDOR': 1744,
                                'IDRID': 516,
                                'DEEPDR': 2000,
                                'FGADR': 1842,
                                'APTOS': 3662,
                                'RLDR': 1593}
        
        self.label_num_dict = {'MESSIDOR': [1016, 269, 347, 75, 35],
                                'IDRID': [175, 26, 163, 89, 60],
                                'DEEPDR': [917, 214, 402, 353, 113],
                                'FGADR': [100, 211, 595, 646, 286],
                                'APTOS': [1804, 369, 999, 192, 294],
                                'RLDR': [165, 336, 929, 98, 62]}

        self.domain_prob, self.label_prob = self.get_domain_label_prob()
        self.domain_prob, self.label_prob = self.multinomial_soomthing(self.domain_prob, self.label_prob, self.beta)

        self.UnsupLoss = SupConLoss(temperature = self.temperature, reduction='none')
        self.SupLoss = nn.CrossEntropyLoss(reduction='none')

    def get_domain_label_prob(self):
        source_domain_num_list = torch.Tensor([self.domain_num_dict[domain] for domain in self.training_domains])
        source_domain_num = torch.sum(source_domain_num_list)
        domain_prob = source_domain_num_list / source_domain_num

        label_num_list = torch.Tensor([self.label_num_dict[domain] for domain in  self.training_domains]).sum(dim=0)
        label_num = torch.sum(label_num_list)
        label_prob = label_num_list / label_num

        return domain_prob.cuda(), label_prob.cuda()

    def multinomial_soomthing(self, domain_prob, label_prob, beta = 0.8):
        domain_prob = torch.pow(domain_prob, beta)
        label_prob = torch.pow(label_prob, beta)

        domain_prob = domain_prob / torch.sum(domain_prob)
        label_prob = label_prob / torch.sum(label_prob)

        return domain_prob, label_prob

    def get_weights(self, labels, domains):
        domain_prob = torch.index_select(self.domain_prob, 0, domains).cuda()
        domain_weight = 1 / domain_prob
        class_prob = torch.index_select(self.label_prob, 0, labels).cuda()
        class_weight = 1 / class_prob

        return domain_weight, class_weight
                            
    def forward(self, output, features, labels, domains):  
        
        domain_weight, class_weight = self.get_weights(labels, domains)
        self.SupLoss3 = LogitAdjust(self.label_num_dict[self.training_domains[0]])    #### 使用逻辑补偿loss

        loss_dict = {}

        output1, output2, output_new_masked1, output_new_masked2 = output
        features_ori1, features_new1, features_ori2, features_new2 = features
        
        # print(output1.shape, output2.shape, labels.shape)


        loss_sup1 = 0
        loss_sup2 = 0


        # for op_item in output1:
        loss_sup1 += self.SupLoss3(output1, labels)            

        # for op_item in output2:
        loss_sup2 += self.SupLoss3(output2, labels)    
            
        temperature=1.0
        # loss_sup1 += - 1  * (F.softmax(output2 / temperature, 1).detach() * \
        #                                         F.log_softmax(output1 / temperature, 1)).sum() / \
        #                          output1.size()[0]


        # loss_sup2 += - 1 * (F.softmax(output1 / temperature, 1).detach() * \
        #                                             F.log_softmax(output2 / temperature, 1)).sum() / \
        #                              output2.size()[0]

        
        loss_sup1_mask = - 1  * (F.softmax(output1 / temperature, 1).detach() * \
                                                F.log_softmax(output_new_masked1 / temperature, 1)).sum() / \
                                 output1.size()[0]


        loss_sup2_mask = - 1 * (F.softmax(output2 / temperature, 1).detach() * \
                                                    F.log_softmax(output_new_masked2 / temperature, 1)).sum() / \
                                     output2.size()[0]
        
        # features_multi1_intra = torch.stack([features_ori1, features_new1], dim = 1)
        # features_multi1_intra = F.normalize(features_multi1_intra, p=2, dim=2)      
        
        # loss_unsup1_intra = torch.mean(self.UnsupLoss(features_multi1_intra))
        
        # features_multi2_intra = torch.stack([features_ori2, features_new2], dim = 1)
        # features_multi2_intra = F.normalize(features_multi2_intra, p=2, dim=2)      
        
        # loss_unsup2_intra = torch.mean(self.UnsupLoss(features_multi2_intra))
        
        
        features_multi1 = torch.stack([features_ori1, features_new2], dim = 1)
        features_multi1 = F.normalize(features_multi1, p=2, dim=2)      
        
        loss_unsup1 = torch.mean(self.UnsupLoss(features_multi1))
        
        features_multi2 = torch.stack([features_ori2, features_new1], dim = 1)
        features_multi2 = F.normalize(features_multi2, p=2, dim=2)      
        
        loss_unsup2 = torch.mean(self.UnsupLoss(features_multi2))  ### 没有使用labels

        # features_multi = torch.stack([features_ori1, features_new1, features_ori2, features_new2], dim = 1)
        # features_multi = F.normalize(features_multi, p=2, dim=2)      

        # loss_unsup = torch.mean(self.UnsupLoss(features_multi))  ### 没有使用labels

        
        # loss_sup1 = torch.mean(loss_sup1 * class_weight * domain_weight) / (torch.mean(domain_weight) * torch.mean(class_weight)) + loss_sup1_mask 

        # loss_sup2 = torch.mean(loss_sup2 * class_weight * domain_weight) / (torch.mean(domain_weight) * torch.mean(class_weight)) + loss_sup2_mask

        # loss1 = (1 - self.alpha) * loss_sup1 + self.alpha * (loss_unsup1 / self.scaling_factor) # + loss_unsup1_intra / self.scaling_factor)
        # loss2 = (1 - self.alpha) * loss_sup2 + self.alpha * (loss_unsup2 / self.scaling_factor) # + loss_unsup2_intra / self.scaling_factor)
        

        loss1 = loss_sup1  + loss_sup1_mask  + self.alpha * (loss_unsup1 / self.scaling_factor) # + loss_unsup1_intra / self.scaling_factor)
        loss2 = loss_sup2 + loss_sup2_mask  + self.alpha * (loss_unsup2 / self.scaling_factor) # + loss_unsup2_intra / self.scaling_factor)
        
        # print(loss_sup1,loss_sup2, loss_unsup1 / self.scaling_factor, loss_unsup2 / self.scaling_factor)
        loss=loss1+loss2 #+  self.alpha * (loss_unsup / self.scaling_factor)

        # print(loss)

        loss_dict['loss'] = loss.item()
        loss_dict['loss_sup1'] = loss_sup1.item()
        loss_dict['loss_unsup1'] = loss_unsup1.item()
        loss_dict['loss_sup2'] = loss_sup2.item()
        loss_dict['loss_unsup2'] = loss_unsup2.item()
         
        return loss, loss_dict

    def update_alpha(self, iteration):
        self.alpha = 1 - iteration / self.max_iteration #* 0.5 
        return self.alpha
    
    
def SimSiamLoss(p, z, version='simplified'):  # negative cosine similarity
    z = z.detach()  # stop gradient

    if version == 'original':
        p = F.normalize(p, dim=1)  # l2-normalize
        z = F.normalize(z, dim=1)  # l2-normalize
        return -(p * z).sum(dim=1).mean()

    elif version == 'simplified':  # same thing, much faster. Scroll down, speed test in __main__
        return - F.cosine_similarity(p, z, dim=-1).mean()
    else:
        raise Exception
    

# Our loss function
class DahLoss_Siam(nn.Module):
    def __init__(self, max_iteration, training_domains, beta = 0.8, scaling_factor = 4, alpha = 1.0, temperature = 0.07) -> None:
        super(DahLoss_Siam, self).__init__()
        self.max_iteration = max_iteration
        self.training_domains = training_domains
        self.alpha = alpha
        self.beta = beta
        self.scaling_factor = scaling_factor
        self.temperature = temperature
        
        self.domain_num_dict = {'MESSIDOR': 1744,
                                'IDRID': 516,
                                'DEEPDR': 2000,
                                'FGADR': 1842,
                                'APTOS': 3662,
                                'RLDR': 1593}
        
        self.label_num_dict = {'MESSIDOR': [1016, 269, 347, 75, 35],
                                'IDRID': [175, 26, 163, 89, 60],
                                'DEEPDR': [917, 214, 402, 353, 113],
                                'FGADR': [100, 211, 595, 646, 286],
                                'APTOS': [1804, 369, 999, 192, 294],
                                'RLDR': [165, 336, 929, 98, 62]}

        self.domain_prob, self.label_prob = self.get_domain_label_prob()
        self.domain_prob, self.label_prob = self.multinomial_soomthing(self.domain_prob, self.label_prob, self.beta)

        self.UnsupLoss = SupConLoss(temperature = self.temperature, reduction='none')
        self.SupLoss = nn.CrossEntropyLoss(reduction='none')

    def get_domain_label_prob(self):
        source_domain_num_list = torch.Tensor([self.domain_num_dict[domain] for domain in self.training_domains])
        source_domain_num = torch.sum(source_domain_num_list)
        domain_prob = source_domain_num_list / source_domain_num

        label_num_list = torch.Tensor([self.label_num_dict[domain] for domain in  self.training_domains]).sum(dim=0)
        label_num = torch.sum(label_num_list)
        label_prob = label_num_list / label_num

        return domain_prob.cuda(), label_prob.cuda()

    def multinomial_soomthing(self, domain_prob, label_prob, beta = 0.8):
        domain_prob = torch.pow(domain_prob, beta)
        label_prob = torch.pow(label_prob, beta)

        domain_prob = domain_prob / torch.sum(domain_prob)
        label_prob = label_prob / torch.sum(label_prob)

        return domain_prob, label_prob

    def get_weights(self, labels, domains):
        domain_prob = torch.index_select(self.domain_prob, 0, domains).cuda()
        domain_weight = 1 / domain_prob
        class_prob = torch.index_select(self.label_prob, 0, labels).cuda()
        class_weight = 1 / class_prob

        return domain_weight, class_weight
                            
    def forward(self, output, features, labels, domains):  
        
        domain_weight, class_weight = self.get_weights(labels, domains)

        loss_dict = {}

        output, output_new_masked = output
        # features_ori1, features_new1, features_ori2, features_new2 = features
        z1,z2, z1_sup, z2_sup, p1, p2 = features

        # print(output1.shape, output2.shape, labels.shape)


        loss_sup1 = 0
        loss_sup2 = 0


        # for op_item in output1:
        loss_sup1 += self.SupLoss(output, labels)            

        temperature=1.0
        # loss_sup1 += - 1  * (F.softmax(output2 / temperature, 1).detach() * \
        #                                         F.log_softmax(output1 / temperature, 1)).sum() / \
        #                          output1.size()[0]


        # loss_sup2 += - 1 * (F.softmax(output1 / temperature, 1).detach() * \
        #                                             F.log_softmax(output2 / temperature, 1)).sum() / \
        #                              output2.size()[0]
        
        loss_sup_mask = - 1  * (F.softmax(output / temperature, 1).detach() * \
                                                F.log_softmax(output_new_masked / temperature, 1)).sum() / output.size()[0]

        # calculate loss
        SSL_loss = -0.5 * (torch.sum(p1 * z2.detach(), dim=-1) + torch.sum(p2 * z1.detach(), dim=-1))

        Sup_loss = -0.5 * (torch.sum(p1 * z2_sup, dim=-1) + torch.sum(p2 * z1_sup, dim=-1))

        loss_siam = 0.5 * SSL_loss.mean()  + 0.5 * Sup_loss.mean()
        

        # features_multi1 = torch.stack([features_ori1, features_new2], dim = 1)
        # features_multi1 = F.normalize(features_multi1, p=2, dim=2)      
        
        # loss_unsup1 = torch.mean(self.UnsupLoss(features_multi1))
        
        # features_multi2 = torch.stack([features_ori2, features_new1], dim = 1)
        # features_multi2 = F.normalize(features_multi2, p=2, dim=2)      
        
        # loss_unsup2 = torch.mean(self.UnsupLoss(features_multi2))  ### 没有使用labels

        # features_multi = torch.stack([features_ori1, features_new1, features_ori2, features_new2], dim = 1)
        # features_multi = F.normalize(features_multi, p=2, dim=2)      

        # loss_unsup = torch.mean(self.UnsupLoss(features_multi))  ### 没有使用labels

        
        loss_sup = torch.mean(loss_sup1 * class_weight * domain_weight) / (torch.mean(domain_weight) * torch.mean(class_weight)) + loss_sup_mask

        # loss1 = (1 - self.alpha) * loss_sup1 + self.alpha * (loss_unsup1 / self.scaling_factor) # + loss_unsup1_intra / self.scaling_factor)
        # loss2 = (1 - self.alpha) * loss_sup2 + self.alpha * (loss_unsup2 / self.scaling_factor) # + loss_unsup2_intra / self.scaling_factor)
        

        # loss1 = (1 - self.alpha) * loss_sup1 * 0.5  + self.alpha * loss_siam1 / 200  # + loss_unsup1_intra / self.scaling_factor)
        # loss2 = (1 - self.alpha) * loss_sup2 * 0.5  + self.alpha * loss_siam2 / 200   # + loss_unsup2_intra / self.scaling_factor)
        
        # loss = loss_sup * 0.5  + ( 1- self.alpha) * loss_siam  # + loss_unsup1_intra / self.scaling_factor)
        # loss =  ( 1- self.alpha) * loss_sup  + self.alpha * loss_siam * 0.5  # + loss_unsup1_intra / self.scaling_factor)
        # loss =  ( 1- self.alpha) * loss_sup * 0.5  + self.alpha * loss_siam * 0.5  # + loss_unsup1_intra / self.scaling_factor)

        loss =   loss_sup  +  loss_siam * 3.0  # + loss_unsup1_intra / self.scaling_factor)

        # loss =   (1-self.alpha)* loss_sup  +  loss_siam *  self.alpha * 4.0
        # loss1 = loss_sup1 * 0.5  + self.alpha * loss_siam1 / 1000 # + loss_unsup1_intra / self.scaling_factor)
        # loss2 = loss_sup2 * 0.5  + self.alpha * loss_siam2 / 1000 # + loss_unsup2_intra / self.scaling_factor)
        
        # loss = loss_sup  + loss_siam *  self.alpha * 3.0# + loss_unsup2_intra / self.scaling_factor)


        # print(loss)

        loss_dict['loss'] = loss.item()
        loss_dict['loss_sup'] = loss_sup.item()
        loss_dict['loss_siam'] = loss_siam.item()
        # loss_dict['loss_siam2'] = loss_siam2.item()

        # loss_dict['loss_sup2'] = loss_sup2.item()
        # loss_dict['loss_unsup2'] = loss_unsup2.item()
         
        return loss, loss_dict

    def update_alpha(self, iteration):
        self.alpha = 1 - iteration / self.max_iteration * 0.5
        return self.alpha
    
    
def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
    

class LogitAdjust(nn.Module):

    def __init__(self, cls_num_list, tau=1, weight=None):
        super(LogitAdjust, self).__init__()
        cls_num_list = torch.cuda.FloatTensor(cls_num_list)
        cls_p_list = cls_num_list / cls_num_list.sum()
        m_list = tau * torch.log(cls_p_list)
        self.m_list = m_list.view(1, -1)
        self.weight = weight

    def forward(self, x, target):
        x_m = x + self.m_list
        return F.cross_entropy(x_m, target, weight=self.weight)


# Our loss function
class DahLoss_Siam_Fastmoco(nn.Module):
    def __init__(self, cfg, max_iteration, training_domains, beta = 0.8, scaling_factor = 4, alpha = 1.0, temperature = 0.07,  fastmoco=1.0) -> None:
        super(DahLoss_Siam_Fastmoco, self).__init__()
        self.max_iteration = max_iteration
        self.training_domains = training_domains
        self.alpha = alpha
        self.beta = beta
        self.scaling_factor = scaling_factor
        self.temperature = temperature
        self.gamma = fastmoco    
        self.cfg=cfg    
        self.domain_num_dict = {'MESSIDOR': 1744,
                                'IDRID': 516,
                                'DEEPDR': 2000,
                                'FGADR': 1842,
                                'APTOS': 3662,
                                'RLDR': 1593}
        
        self.label_num_dict = {'MESSIDOR': [1016, 269, 347, 75, 35],
                                'IDRID': [175, 26, 163, 89, 60],
                                'DEEPDR': [917, 214, 402, 353, 113],
                                'FGADR': [100, 211, 595, 646, 286],
                                'APTOS': [1804, 369, 999, 192, 294],
                                'RLDR': [165, 336, 929, 98, 62]}

        self.domain_prob, self.label_prob = self.get_domain_label_prob()
        self.domain_prob, self.label_prob = self.multinomial_soomthing(self.domain_prob, self.label_prob, self.beta)

        self.UnsupLoss = SupConLoss(temperature = self.temperature, reduction='none')
        self.SupLoss = nn.CrossEntropyLoss(reduction='none')
        self.SupLoss2 = nn.CrossEntropyLoss()
        
        
    def get_domain_label_prob(self):
        source_domain_num_list = torch.Tensor([self.domain_num_dict[domain] for domain in self.training_domains])
        source_domain_num = torch.sum(source_domain_num_list)
        domain_prob = source_domain_num_list / source_domain_num

        label_num_list = torch.Tensor([self.label_num_dict[domain] for domain in  self.training_domains]).sum(dim=0)
        label_num = torch.sum(label_num_list)
        label_prob = label_num_list / label_num

        return domain_prob.cuda(), label_prob.cuda()

    def multinomial_soomthing(self, domain_prob, label_prob, beta = 0.8):
        domain_prob = torch.pow(domain_prob, beta)
        label_prob = torch.pow(label_prob, beta)

        domain_prob = domain_prob / torch.sum(domain_prob)
        label_prob = label_prob / torch.sum(label_prob)

        return domain_prob, label_prob

    def get_weights(self, labels, domains):
        domain_prob = torch.index_select(self.domain_prob, 0, domains).cuda()
        domain_weight = 1 / domain_prob
        class_prob = torch.index_select(self.label_prob, 0, labels).cuda()
        class_weight = 1 / class_prob

        return domain_weight, class_weight
                            
    def forward(self, output, features, labels, domains):  
        
        # labels=labels 
        # domains=domains
        domain_weight, class_weight = self.get_weights(labels, domains)
        # domain_weight_a, class_weight_a = self.get_weights(labels_a, domains_a)

        loss_dict = {}

        output, output_new_a, output_new_masked = output
        # features_ori1, features_new1, features_ori2, features_new2 = features
        if self.gamma> 0:
            features_ori, features_new, z1,z2, z1_sup, z2_sup, p1, p2, z1_orthmix, z2_orthmix, p1_orthmix, p2_orthmix, features_x_mixed, mix_z_, mix_result, targets_b, lam = features
            # z_new_masked,z_ori_masked,p_new_masked, p_ori_masked=features_all
        else:
             z1,z2, z1_sup, z2_sup, p1, p2 = features

# features_ori_z1, features_new_z2, z1_sup, z2_sup, p1, p2
        loss_sup = 0
        # print(domains)
        # print(self.label_num_dict)
        # print(self.training_domains)
        self.SupLoss3 = LogitAdjust(self.label_num_dict[self.training_domains[0]],tau=0.6)    #### 使用逻辑补偿loss

        # for op_item in output1:
        loss_sup += self.SupLoss3(output, labels)            
        # loss_sup2 = 1.0 * self.SupLoss(output_new_a, labels_a)         

        temperature=1.0

        # loss_sup_mask = self.SupLoss3(output_new_masked, F.softmax( output / temperature, 1).detach())   # 
        loss_sup_mask = - 1  * (F.softmax( output / temperature, 1).detach() * F.log_softmax(output_new_masked / temperature, 1)).sum() / output.size()[0]

        # loss_sup_mask += - 1  * (F.softmax(output / temperature, 1).detach() * F.log_softmax(output_new_masked_freq / temperature, 1)).sum() / output.size()[0]
        
        
        # loss_sup = torch.mean(loss_sup * class_weight * domain_weight) / (torch.mean(domain_weight) * torch.mean(class_weight)) 
        
        # print(class_weight)
        # print(domain_weight)
        # loss_sup2 = torch.mean(loss_sup2 * class_weight_a * domain_weight_a) / (torch.mean(domain_weight_a) * torch.mean(class_weight_a)) 


        # features_multi = torch.stack([features_ori, features_new], dim = 1)
        # features_multi = F.normalize(features_multi, p=2, dim=2)      
        
        # loss_unsup = torch.mean(self.UnsupLoss(features_multi))
        

        # calculate loss
        SSL_loss = -0.5 * (torch.sum(p1 * z2.detach(), dim=-1) + torch.sum(p2 * z1.detach(), dim=-1))

        Sup_loss = -0.5 * (torch.sum(p1 * z2_sup, dim=-1) + torch.sum(p2 * z1_sup, dim=-1))

        loss_siam = 0.5 * SSL_loss.mean()  + 0.5 * Sup_loss.mean()
        
        
        # calculate loss
        # SSL_loss = -0.5 * (torch.sum(p_ori_masked * z_new_masked.detach(), dim=-1) + torch.sum(p_new_masked * z_ori_masked.detach(), dim=-1))

        # Sup_loss = -0.5 * (torch.sum(p_ori_masked * z2_sup, dim=-1) + torch.sum(p_new_masked * z1_sup, dim=-1))

        # loss_siam = 0.5 * SSL_loss.mean()  + 0.5 * Sup_loss.mean()
        
        
        
        if self.gamma> 0:
            # calculate loss for fastcoco
            SSL_loss_fastmoco=0.0
            Sup_loss_fastmoco=0.0
            
            # print(len(p1_orthmix)) # 6

            for i in range(len(p1_orthmix)):
                # z1_orthmix, z2_orthmix = nn.functional.normalize(z1_orthmix, dim=-1), nn.functional.normalize(z2_orthmix, dim=-1)
                # print(p1_orthmix, p1_orthmix[i].shape)
                p1_orthmix_, p2_orthmix_ = nn.functional.normalize(p1_orthmix[i], dim=-1), nn.functional.normalize(p2_orthmix[i], dim=-1)
                SSL_loss_fastmoco += -0.5 * (torch.sum(p1_orthmix_ * z2.detach(), dim=-1).mean() + torch.sum(p2_orthmix_ * z1.detach(), dim=-1).mean())

                Sup_loss_fastmoco += -0.5 * (torch.sum(p1_orthmix_ * z2_sup, dim=-1).mean() + torch.sum(p2_orthmix_ * z1_sup, dim=-1).mean())

            loss_siam_fastmoco = 0.5 * SSL_loss_fastmoco / len(p1_orthmix) + 0.5 * Sup_loss_fastmoco/ len(p1_orthmix)
            
            # print(loss_sup ,loss_sup_mask, loss_siam_fastmoco )
            loss =  loss_siam_fastmoco * self.gamma #*  self.alpha # + self.alpha * (loss_unsup / self.scaling_factor) #self.cfg.MASKED * loss_sup_mask + 
        else:
            print(" Not FastMOCO")
            loss =   loss_sup  +  loss_siam *  self.alpha #+ self.alpha * (loss_unsup / self.scaling_factor)

        # criterion = nn.CrossEntropyLoss()  #nn.CrossEntropyLoss(reduction='none') # nn.CrossEntropyLoss()
        # loss_mix = F.mse_loss(features_x_mixed, mix_z_.detach())
        # mix_cate_loss = mixup_criterion(criterion,mix_result,labels,targets_b,lam)

        # loss =   loss_sup  +   loss_siam_fastmoco * 4.0  # + loss_unsup1_intra / self.scaling_factor)

        
        # loss =   loss_sup  +  (loss_siam + loss_siam_fastmoco) * 2.0 # + 1.0 * (loss_mix + 0.1 * mix_cate_loss) # + loss_unsup1_intra / self.scaling_factor)

        loss_dict['loss'] = loss.item()
        loss_dict['loss_sup'] = loss_sup.item()
        loss_dict['loss_siam'] = loss_siam.item()
        loss_dict['loss_siam_fastmoco'] = loss_siam_fastmoco.item()

        # loss_dict['loss_siam2'] = loss_siam2.item()

        # loss_dict['loss_sup2'] = loss_sup2.item()
        # loss_dict['loss_unsup2'] = loss_unsup2.item()
         
        return loss, loss_dict

    def update_alpha(self, iteration):
        self.alpha = 1 - iteration / self.max_iteration * 0.5
        return self.alpha
    
    
def D(p, z, random_matrix=None, version='simplified'): 
    if version == 'original':
        z = z.detach()            # stop gradient
        p = F.normalize(p, dim=1) # l2-normalize 
        z = F.normalize(z, dim=1) # l2-normalize 
        return -(p*z).sum(dim=1).mean()

    elif version == 'simplified':
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()

    elif version == 'random':
        # z = z.detach()            # stop gradient
        # p = F.normalize(p, dim=1) # l2-normalize 
        # z = F.normalize(z, dim=1) # l2-normalize 
        p = torch.matmul(p, random_matrix)
        z = torch.matmul(z, random_matrix)
        # p = F.normalize(p, dim=1) # l2-normalize 
        # z = F.normalize(z, dim=1) # l2-normalize 
        return - F.cosine_similarity(p, z.detach() , dim=-1).mean()

    else:
        raise Exception


def Cal_Loss_Matrix(z1, z2, random_matrix, margin=1.0):
    Triplet_loss = torch.tensor(0.).cuda()
    Triplet_loss.requires_grad = True
    N, Z = z1.shape 
    device = z1.device 

    representations = torch.cat([z1, z2], dim=0)               # 未归一化
    representations_norm = F.normalize(representations, dim=1) # 归一化
    representations_temp = torch.matmul(representations, random_matrix)                             # 2N x Z
    representations_temp = F.normalize(representations_temp, dim=1)

    similarity_matrix = torch.matmul(representations_temp, torch.transpose(representations_norm, 0, 1))  # 2N x 2N
    l_pos = torch.diag(similarity_matrix, N)                                                        # 得到 1~N+1 ... 共N个正样本对
    r_pos = torch.diag(similarity_matrix, -N)
    positives = torch.cat([l_pos, r_pos]).view(2 * N, 1)                                            # 2N x 1
    diag = torch.eye(2*N, dtype=torch.bool, device=device)
    diag[N:,:N] = diag[:N,N:] = diag[:N,:N]
    negatives = similarity_matrix[~diag].view(2*N, -1)                                              # 2N x 2(N-1)
    temp_loss = margin + negatives - positives
    temp_loss = torch.clamp(temp_loss, min=0.)
    Triplet_loss = torch.mean(temp_loss)
    return Triplet_loss


class LogitAdjust_KD_V1(nn.Module):

    def __init__(self, cls_num_list, tau=1,  temperature=1.0, weight=None):
        super(LogitAdjust_KD_V1, self).__init__()
        cls_num_list = torch.cuda.FloatTensor(cls_num_list)
        cls_p_list = cls_num_list / cls_num_list.sum()
        m_list1 = tau * torch.log(cls_p_list)
        # m_list2 = tau2 * torch.log(cls_p_list)

        self.m_list1 = m_list1.view(1, -1)
        self.m_list2 = m_list1.view(1, -1)

        self.weight = weight
        self.temperature =  temperature

    def forward(self, x, target, label):
        x_m = x + self.m_list1
        target = target + self.m_list2    ##### 对于输出的logit进行矫正
        target = F.softmax( target / self.temperature , 1).detach()
        x_m = F.softmax( x / self.temperature , 1) #.detach()

        output_target_max, output_target_index = torch.max(F.softmax((target), dim=1).detach(), dim=1)
        
        return  -(target[(output_target_index == label)] * torch.log(x_m[(output_target_index == label)])).sum() / target.size()[0]




def softmax_focal_loss(x, target, gamma=2.0, alpha=0.25, class_weight=None, domain_weight=None):
    n = x.shape[0]
    
    device = target.device
    range_n = torch.arange(0, n, dtype=torch.int64, device=device)

    pos_num =  float(x.shape[1])   ### classes of number
    p = torch.softmax(x, dim=1)
    p = p[range_n, target]
    # gamma = gamma + 2 * (class_weight * domain_weight / (torch.mean(domain_weight) * torch.mean(class_weight))) / pos_num
    # print(gamma)
    # print()
    # print(p.shape,class_weight.shape, domain_weight.shape )
    # loss = -(1-p)**gamma*alpha*torch.log(p)  ## 
    # print(loss)
    # print(loss.shape, pos_num, x.shape) ## torch.Size([32]) 5.0 torch.Size([32, 5])
    # print(class_weight / torch.mean(class_weight ) / pos_num )

    if class_weight is not None:
        # print(torch.mean((loss * class_weight * domain_weight) / (torch.mean(domain_weight) * torch.mean(class_weight))))
        gamma = gamma + 3.5 * (class_weight * domain_weight / (torch.mean(domain_weight) * torch.mean(class_weight))) / pos_num
        # print(gamma)
        # print(gamma)
        # print()
        # print(p.shape,class_weight.shape, domain_weight.shape )
        loss = -(1-p)**gamma*alpha*torch.log(p)  ## 
        # loss = torch.sum((loss * class_weight * domain_weight)).mean() / (torch.mean(domain_weight) * torch.mean(class_weight)) 
        loss = loss # (loss * class_weight * domain_weight) / (torch.mean(domain_weight) * torch.mean(class_weight))  / pos_num
            # loss_sup_mask += - 1 * (torch.sum(soft_output.detach() * F.log_softmax(output_new_masked / temperature, 1), dim=-1) * class_weight * domain_weight).mean() 
# = torch.mean(loss_sup * class_weight * domain_weight) / (torch.mean(domain_weight) * torch.mean(class_weight)) 

        return torch.sum(loss) / pos_num  #torch.sum(loss) / pos_num # loss #.mean()  # torch.sum(loss) / pos_num
    else:
        loss = -(1-p)**gamma*alpha*torch.log(p)  ## 

        return torch.sum(loss) / pos_num



def softmax_focal_kd_loss(x, target, gamma=2., alpha=0.25):
    n = x.shape[0]
    device = target.device
    range_n = torch.arange(0, n, dtype=torch.int64, device=device)

    pos_num =  float(x.shape[1])
    # p = torch.softmax(x, dim=1)
    # print()
    p = x[range_n, target]
    loss = -(1-p)**gamma*alpha*torch.log(p)
    return torch.sum(loss) / pos_num


class FocalLossWithSmoothing(nn.Module):
    def __init__(
            self,
            num_classes: int,
            gamma: int = 1,
            lb_smooth: float = 0.1,
            size_average: bool = True,
            ignore_index: int = None,
            alpha: float = None):
        """
        :param gamma:
        :param lb_smooth:
        :param ignore_index:
        :param size_average:
        :param alpha:
        """
        super(FocalLossWithSmoothing, self).__init__()
        self._num_classes = num_classes
        self._gamma = gamma
        self._lb_smooth = lb_smooth
        self._size_average = size_average
        self._ignore_index = ignore_index
        self._log_softmax = nn.LogSoftmax(dim=1)
        self._alpha = alpha

        if self._num_classes <= 1:
            raise ValueError('The number of classes must be 2 or higher')
        if self._gamma < 0:
            raise ValueError('Gamma must be 0 or higher')
        if self._alpha is not None:
            if self._alpha <= 0 or self._alpha >= 1:
                raise ValueError('Alpha must be 0 <= alpha <= 1')

    def forward(self, logits, logits_target, label):
        """
        :param logits: (batch_size, class, height, width)
        :param label:
        :return:
        """
        logits = logits.float()
        difficulty_level = self._estimate_difficulty_level(logits, label)   ###

        # with torch.no_grad():
        #     label = label.clone().detach()
        #     if self._ignore_index is not None:
        #         ignore = label.eq(self._ignore_index)
        #         label[ignore] = 0
        #     lb_pos, lb_neg = 1. - self._lb_smooth, self._lb_smooth / (self._num_classes - 1)
        #     lb_one_hot = torch.empty_like(logits).fill_(
        #         lb_neg).scatter_(1, label.unsqueeze(1), lb_pos).detach()
        logs = self._log_softmax(logits)
        loss = -torch.sum(difficulty_level * logs, dim=1)
        # if self._ignore_index is not None:
        #     loss[ignore] = 0
        return loss.mean()

    def _estimate_difficulty_level(self, logits, label):
        """
        :param logits:
        :param label:
        :return:
        """
        one_hot_key = torch.nn.functional.one_hot(label, num_classes=self._num_classes)
        if len(one_hot_key.shape) == 4:
            one_hot_key = one_hot_key.permute(0, 3, 1, 2)
        if one_hot_key.device != logits.device:
            one_hot_key = one_hot_key.to(logits.device)
        pt = one_hot_key * F.softmax(logits)
        difficulty_level = torch.pow(1 - pt, self._gamma)
        return difficulty_level

# from torch.nn.modules.loss import _Loss
# class SoftmaxEQLV2Loss(_Loss):
#     def __init__(self, num_classes, indicator='pos', loss_weight=1.0, tau=1.0, eps=1e-4):
#         super(SoftmaxEQLV2Loss, self).__init__()
#         self.loss_weight = loss_weight
#         self.num_classes = num_classes
#         self.tau = tau
#         self.eps = eps

#         assert indicator in ['pos', 'neg', 'pos_and_neg'], 'Wrong indicator type!'
#         self.indicator = indicator

#         # initial variables
#         self.register_buffer('pos_grad', torch.zeros(num_classes))
#         self.register_buffer('neg_grad', torch.zeros(num_classes))
#         self.register_buffer('pos_neg', torch.ones(num_classes))

#     def forward(self, input, label):
#         if self.indicator == 'pos':
#             indicator = self.pos_grad.detach()
#         elif self.indicator == 'neg':
#             indicator = self.neg_grad.detach()
#         elif self.indicator == 'pos_and_neg':
#             indicator = self.pos_neg.detach() + self.neg_grad.detach()
#         else:
#             raise NotImplementedError

#         if label.dim() == 1:
#             one_hot = F.one_hot(label, self.num_classes)
#         else:
#             one_hot = label.clone()
#         self.targets = one_hot.detach()

#         indicator = indicator / (indicator.sum() + self.eps)
#         indicator = (indicator ** self.tau + 1e-9).log()
#         cls_score = input + indicator[None, :]
#         loss = F.cross_entropy(cls_score, label)
#         return loss * self.loss_weight

    def collect_grad(self, grad):
        grad = torch.abs(grad)
        pos_grad = torch.sum(grad * self.targets, dim=0)
        neg_grad = torch.sum(grad * (1 - self.targets), dim=0)

        # allreduce(pos_grad)
        # allreduce(neg_grad)

        self.pos_grad += pos_grad
        self.neg_grad += neg_grad
        self.pos_neg = self.pos_grad / (self.neg_grad + 1e-10)


# Our loss function
class DahLoss_Siam_Fastmoco_v0(nn.Module):
    def __init__(self, cfg, max_iteration, training_domains, beta = 0.8, scaling_factor = 4, alpha = 1.0, temperature = 0.07,  fastmoco=1.0) -> None:
        super(DahLoss_Siam_Fastmoco_v0, self).__init__()
        self.max_iteration = max_iteration
        self.training_domains = training_domains
        self.alpha = alpha
        self.beta = beta
        self.scaling_factor = scaling_factor
        self.temperature = temperature
        self.gamma = fastmoco    
        self.cfg=cfg    
        self.domain_num_dict = {'MESSIDOR': 1744,
                                'IDRID': 516,
                                'DEEPDR': 2000,
                                'FGADR': 1842,
                                'APTOS': 3662,
                                'RLDR': 1593}
        
        self.label_num_dict = {'MESSIDOR': [1016, 269, 347, 75, 35],
                                'IDRID': [175, 26, 163, 89, 60],
                                'DEEPDR': [917, 214, 402, 353, 113],
                                'FGADR': [100, 211, 595, 646, 286],
                                'APTOS': [1804, 369, 999, 192, 294],
                                'RLDR': [165, 336, 929, 98, 62]}

        self.domain_prob, self.label_prob = self.get_domain_label_prob()
        self.domain_prob, self.label_prob = self.multinomial_soomthing(self.domain_prob, self.label_prob, self.beta)

        self.UnsupLoss = SupConLoss(temperature = self.temperature, reduction='none')
        self.SupLoss = nn.CrossEntropyLoss(reduction='none')
        self.SupLoss2 = nn.CrossEntropyLoss()
        self.per_cls_weights = torch.ones(5).cuda()
        self.KdLoss = LogitAdjust_KD_V1(self.label_num_dict[self.cfg.DATASET.SOURCE_DOMAINS[0]],tau=0.0)   
        self.KdLoss_Focal = FocalLossWithSmoothing(cfg.DATASET.NUM_CLASSES,2.0)


        # self.random_matrix = None
        # self.random_in_dim = 2048
        # self.random_out_dim = 2048
        
    def get_domain_label_prob(self):
        source_domain_num_list = torch.Tensor([self.domain_num_dict[domain] for domain in self.training_domains])
        source_domain_num = torch.sum(source_domain_num_list)
        domain_prob = source_domain_num_list / source_domain_num

        label_num_list = torch.Tensor([self.label_num_dict[domain] for domain in  self.training_domains]).sum(dim=0)
        label_num = torch.sum(label_num_list)
        label_prob = label_num_list / label_num

        return domain_prob.cuda(), label_prob.cuda()

    def multinomial_soomthing(self, domain_prob, label_prob, beta = 0.8):
        domain_prob = torch.pow(domain_prob, beta)
        label_prob = torch.pow(label_prob, beta)

        domain_prob = domain_prob / torch.sum(domain_prob)
        label_prob = label_prob / torch.sum(label_prob)

        return domain_prob, label_prob

    def get_weights_v2(self, epoch):
        # domain_prob = torch.index_select(self.domain_prob, 0, domains).cuda()
        # domain_weight = 1 / domain_prob
        # class_prob = torch.index_select(self.label_prob, 0, labels).cuda()
        # class_weight = 1 / class_prob
        cls_num_list = self.label_num_dict[self.training_domains[0]]
        idx = epoch // 80
        betas = [0, 0.9999]
        effective_num = 1.0 - np.power(betas[idx], cls_num_list)
        per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
        self.per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()
            
            
        return self.per_cls_weights
                         

    def get_weights(self, labels, domains):
        domain_prob = torch.index_select(self.domain_prob, 0, domains).cuda()
        domain_weight = 1 / domain_prob
        class_prob = torch.index_select(self.label_prob, 0, labels).cuda()
        class_weight = 1 / class_prob

        return domain_weight, class_weight
    

    def _estimate_difficulty_level(self, logits, label,  gamma=2., alpha=0.25):
        """
        :param logits:
        :param label:
        :return:
        """
        one_hot_key = torch.nn.functional.one_hot(label, num_classes=self.cfg.DATASET.NUM_CLASSES)
        if len(one_hot_key.shape) == 4:
            one_hot_key = one_hot_key.permute(0, 3, 1, 2)
        if one_hot_key.device != logits.device:
            one_hot_key = one_hot_key.to(logits.device)
        pt = one_hot_key * F.softmax(logits)
        difficulty_level = alpha * torch.pow(1 - pt, gamma)
        return difficulty_level
    

    # def change_random_matrix(self):
    #     random_matrix = torch.randn(self.random_in_dim, self.random_out_dim).cuda()
    #     self.random_matrix = random_matrix
    #     # if dist.is_initialized():
        #     dist.broadcast(random_matrix, src=0)
        # self.random_matrix = random_matrix
        # if dist.is_initialized() and dist.get_rank() == 0:
        #     print('change random matrix')
            
            
       
    def forward(self, output, features, features_masked, labels, domains, random_matrix):  
        
        # labels=labels 
        # domains=domains
        domain_weight, class_weight = self.get_weights(labels, domains)
        # domain_weight_a, class_weight_a = self.get_weights(labels_a, domains_a)

        loss_dict = {}

        output, output_new_a, output_new_masked , output_ori_masked_c = output
        # features_ori1, features_new1, features_ori2, features_new2 = features
        if self.gamma> -1.0:
            features_ori, features_new, z1,z2, z1_sup, z2_sup, p1, p2, z1_orthmix, z2_orthmix, p1_orthmix, p2_orthmix = features
            z_new_masked, z_ori_masked_c, p_new_masked, p_ori_masked_c = features_masked
            # z_new_masked,z_ori_masked,p_new_masked, p_ori_masked=features_all
        else:
             z1,z2, z1_sup, z2_sup, p1, p2 = features

# features_ori, features_new, features_ori_z1, features_new_z2, z1_sup, z2_sup, p1, p2, z1_orthmix, z2_orthmix, p1_orthmix, p2_orthmix, z_all_orthmix, p_all_orthmix, features_all_z, z_all_sup, p_all

        loss_sup = 0
        self.SupLoss3 = LogitAdjust(self.label_num_dict[self.training_domains[0]],tau=0.6)    #### 使用逻辑补偿loss
        self.SupLoss4 = VSLoss(self.label_num_dict[self.training_domains[0]], gamma=0.1, tau=0.3)    #### 使用逻辑补偿loss

        # for op_item in output1:
        # loss_sup += 1.0 * self.SupLoss(output, labels)      
        if self.cfg.TRANSFORM.FREQ:
            loss_sup += 0.5 * self.SupLoss(output, labels)   
            loss_sup += 0.5 * self.SupLoss(output_new_a, labels)  
        else:       
            loss_sup += 1.0 * self.SupLoss(output, labels)      
#             loss_sup += 1.0 * self.SupLoss(output_new_a, labels)      
# # 
            loss_sup = torch.mean(loss_sup * class_weight * domain_weight) / (torch.mean(domain_weight) * torch.mean(class_weight)) 

            # loss_sup += 1.0 * softmax_focal_loss(output, labels,  class_weight=class_weight, domain_weight=domain_weight) 
            # loss_sup += 1.0 * softmax_focal_loss(output, labels) 

        

        # print(class_weight.shape, class_weight)

        temperature=1.0

        # loss_sup_mask = self.SupLoss3(output_new_masked, F.softmax( output / temperature, 1).detach())   # ss
        loss_sup_mask = 0
        if self.cfg.TRANSFORM.FREQ:
            # loss_sup_mask += - 0.5 * (F.softmax( output_new_a / temperature, 1).detach() * F.log_softmax(output_new_masked / temperature, 1)).sum() / output.size()[0]
            # loss_sup_mask += - 0.5  * (F.softmax( output / temperature, 1).detach() * F.log_softmax(output_ori_masked_c / temperature, 1)).sum() / output.size()[0]
            
            loss_sup_mask += 0.5 * self.KdLoss(output_new_masked, output.detach(), labels)
            loss_sup_mask += 0.5 * self.KdLoss(output_ori_masked_c, output.detach(), labels)

        else:


# 
            # loss_sup_mask += - 1.0 * (F.softmax( output / temperature, 1).detach() * F.log_softmax(output_new_masked / temperature, 1)).sum() / output.size()[0]


# ((torch.sum(p1_orthmix_ * z2.detach(), dim=-1)* class_weight * domain_weight).mean() 

            # loss_sup_mask += 1.0 * self.KdLoss(output_new_masked, output.detach(), labels)
            # loss_sup_mask += 0.5 * self.KdLoss(output_ori_masked_c, output.detach(), labels)
            
            # alpha_t = self.alpha_T * ((self.step_count + 1) / self.n_steps)
            alpha_t  = self.alpha * self.cfg.SMOOTH #0.4 # 0.4 #0.2 #1 - self.alpha
            alpha_t = max(0, alpha_t)

            # output, output_rb = self.predict_Train(labels)

            targets_numpy = labels.cpu().detach().numpy()
            identity_matrix = torch.eye(self.cfg.DATASET.NUM_CLASSES) 
            targets_one_hot = identity_matrix[targets_numpy]   ### 转换为one-hot biaoqian

            # soft_output = ((1 - alpha_t) * targets_one_hot).to('cuda') + (alpha_t * F.softmax(output, dim=1))
            # soft_output_rb = ((1 - alpha_t) * targets_one_hot).to('cuda') + (alpha_t * F.softmax(output_new_masked, dim=1))
            
            soft_output = ( alpha_t * targets_one_hot).to('cuda') + ( (1-alpha_t ) * F.softmax(output.detach(), dim=1))
            soft_output_rb = (alpha_t * targets_one_hot).to('cuda') + ((1-alpha_t ) * F.softmax(output_new_masked, dim=1))

            # loss_sup_mask += - 1  * (torch.sum((F.softmax( output/ temperature, 1).detach() * F.log_softmax(output_new_masked / temperature, 1)), dim=-1) * class_weight * domain_weight).mean() 

            # loss_sup_mask += - 1  * (torch.sum( soft_output/ temperature * F.log_softmax(output_new_masked / temperature, 1), dim=-1) * class_weight * domain_weight).mean() / (torch.mean(domain_weight) * torch.mean(class_weight)) 

            # loss_sup_mask += - 1  * (torch.sum( soft_output/ temperature * F.log_softmax(output_new_masked / temperature, 1), dim=-1) * class_weight * domain_weight).mean() / (torch.mean(domain_weight) * torch.mean(class_weight)) 

            # loss_sup_mask += - 1  * torch.sum((F.softmax( soft_output/ temperature, 1).detach() * F.log_softmax(output_new_masked / temperature, 1)), dim=-1) 

            loss_sup_mask += - 1.0 * ( soft_output / temperature * F.log_softmax(output_new_masked / temperature, 1)).sum() / output.size()[0]
# 
            # loss_sup_mask += - 1.0 * ( soft_output / temperature * torch.log(soft_output_rb / temperature)).sum() / output.size()[0]

            # focal_class_weight = self._estimate_difficulty_level(output_new_masked, labels)
            # # print(focal_class_weight.shape)  ## torch.Size([32, 5])
            # # print()
            # # print(domain_weight.shape)  ## torch.Size([32])
            # loss_sup_mask += - 1 * (torch.sum(soft_output.detach() * F.log_softmax(output_new_masked / temperature, 1) * focal_class_weight, dim=-1)).mean() 

            # loss_sup_mask += - 1 * (torch.sum(soft_output.detach() * F.log_softmax(output_new_masked / temperature, 1), dim=-1) * class_weight * domain_weight).mean() 



            # loss_sup_mask += self.KdLoss_Focal(output_new_masked, output.detach(), labels) #- 1  * (torch.sum(soft_output.detach() * F.log_softmax(output_new_masked / temperature, 1), dim=-1) * class_weight * domain_weight).mean() 



            # loss_sup_mask += F.kl_div(
            #     torch.log(soft_output_rb),
            #     torch.log(soft_output),
            #     reduction='sum',
            #     log_target=True
            # ) * (temperature * temperature) / output.numel()
        
            # loss_sup_mask += self.KdLoss_Focal(soft_output_rb,soft_output)
            # loss_sup_mask += - 1  * (soft_output.detach() * torch.log(soft_output_rb)).sum() / output.size()[0]
            # loss_sup_mask += softmax_focal_kd_loss(soft_output_rb,soft_output.detach() )

        
        # loss_sup_mask += - 1  * (F.softmax(output / temperature, 1).detach() * F.log_softmax(output_new_masked_freq / temperature, 1)).sum() / output.size()[0]
        
        

        # calculate loss
        SSL_loss = -0.5 * (torch.sum(p1 * z2.detach(), dim=-1) + torch.sum(p2 * z1.detach(), dim=-1))

        Sup_loss = -0.5 * (torch.sum(p1 * z2_sup, dim=-1) + torch.sum(p2 * z1_sup, dim=-1))

        # loss_siam = 0.5 * SSL_loss.mean()  + 0.5 * Sup_loss.mean()
        # loss_siam = 1.0 * SSL_loss.mean()  + 0.0 * Sup_loss.mean()
        loss_siam = 0.5 * SSL_loss.mean()  + 0.5 * Sup_loss.mean()
        
        
        
        features_multi = torch.stack([features_ori[3], features_new[3]], dim = 1)
        features_multi = F.normalize(features_multi, p=2, dim=2)      
        
        loss_unsup = torch.mean(self.UnsupLoss(features_multi))
        
        
        
        # calculate loss
        # SSL_loss = -0.5 * (torch.sum(p_ori_masked * z_new_masked.detach(), dim=-1) + torch.sum(p_new_masked * z_ori_masked.detach(), dim=-1))

        # Sup_loss = -0.5 * (torch.sum(p_ori_masked * z2_sup, dim=-1) + torch.sum(p_new_masked * z1_sup, dim=-1))

        # loss_siam = 0.5 * SSL_loss.mean()  + 0.5 * Sup_loss.mean()
        
        
        
        if self.gamma> -1.0:
            # calculate loss for fastcoco
            SSL_loss_fastmoco=0.0
            Sup_loss_fastmoco=0.0
            
            # print(len(p1_orthmix)) # 6

            for i in range(len(p1_orthmix)):
                # z1_orthmix, z2_orthmix = nn.functional.normalize(z1_orthmix, dim=-1), nn.functional.normalize(z2_orthmix, dim=-1)
                # print(p1_orthmix, p1_orthmix[i].shape)
                p1_orthmix_, p2_orthmix_ = p1_orthmix[i], p2_orthmix[i]

                # p1_orthmix_, p2_orthmix_ = nn.functional.normalize(p1_orthmix[i], dim=-1), nn.functional.normalize(p2_orthmix[i], dim=-1)
                # p_all_orthmix_ = nn.functional.normalize(p_all_orthmix[i], dim=-1)
                
                # SSL_loss_fastmoco += -0.5 * (torch.sum(p1_orthmix_ * z_all.detach(), dim=-1).mean() + torch.sum(p2_orthmix_ * z_all.detach(), dim=-1).mean()+ torch.sum(p_all_orthmix_ * z1.detach(), dim=-1).mean() + torch.sum(p_all_orthmix_ * z2.detach(), dim=-1).mean() )

                # Sup_loss_fastmoco += -0.5 * (torch.sum(p1_orthmix_ * z_all_sup, dim=-1).mean() + torch.sum(p2_orthmix_ * z_all_sup, dim=-1).mean() + torch.sum(p_all_orthmix_ * z1_sup, dim=-1).mean() + torch.sum(p_all_orthmix_ * z2_sup, dim=-1).mean())
                
                
                # SSL_loss_fastmoco += -0.5 * ((torch.sum(p1_orthmix_ * z2.detach(), dim=-1)).mean() + (torch.sum(p2_orthmix_ * z1.detach(), dim=-1)).mean() ) #+ torch.sum(p_all_orthmix_ * z1.detach(), dim=-1).mean() + torch.sum(p_all_orthmix_ * z2.detach(), dim=-1).mean() )

                # Sup_loss_fastmoco += -0.5 * ((torch.sum(p1_orthmix_ * z2_sup, dim=-1)).mean() + (torch.sum(p2_orthmix_ * z1_sup, dim=-1)).mean()) # + torch.sum(p_all_orthmix_ * z1_sup, dim=-1).mean() + torch.sum(p_all_orthmix_ * z2_sup, dim=-1).mean())
                
                # print(random_matrix)
                if self.cfg.DG_MODE =='DG': # DG_MODE
                    # print("DG")
                    SSL_loss_fastmoco += D(p1_orthmix_, z2, random_matrix, version='random') / 2 + D(p2_orthmix_, z1, random_matrix, version='random') / 2
                    
                    Sup_loss_fastmoco += D(p1_orthmix_ , z2_sup, random_matrix, version='random') / + D(p2_orthmix_ , z1_sup, random_matrix, version='random') /2
                else:

                    # SSL_loss_fastmoco += D(p1_orthmix_, z2, random_matrix, version='random') / 2 + D(p2_orthmix_, z1, random_matrix, version='random') / 2
                    
                    # Sup_loss_fastmoco += D(p1_orthmix_ , z2_sup, random_matrix, version='random') / + D(p2_orthmix_ , z1_sup, random_matrix, version='random') /2

                    SSL_loss_fastmoco += D(p1_orthmix_, z2, random_matrix, version='random') / 2 + D(p2_orthmix_, z1, random_matrix, version='random') / 2
                    
                    Sup_loss_fastmoco += D(p1_orthmix_ , z2_sup, random_matrix, version='random') / 2  + D(p2_orthmix_ , z1_sup, random_matrix, version='random') /2 # + torch.sum(p_all_orthmix_ * z1_sup, dim=-1).mean() + torch.sum(p_all_orthmix_ * z2_sup, dim=-1).mean())

                # SSL_loss_fastmoco += -0.5 * ((torch.sum(p1_orthmix_ * z2.detach(), dim=-1)* class_weight * domain_weight).mean() + (torch.sum(p2_orthmix_ * z1.detach(), dim=-1)* class_weight * domain_weight).mean() ) #+ torch.sum(p_all_orthmix_ * z1.detach(), dim=-1).mean() + torch.sum(p_all_orthmix_ * z2.detach(), dim=-1).mean() )
                

                # Sup_loss_fastmoco += -0.5 * ((torch.sum(p1_orthmix_ * z2_sup, dim=-1)* class_weight * domain_weight).mean() + (torch.sum(p2_orthmix_ * z1_sup, dim=-1)* class_weight * domain_weight).mean()) # + torch.sum(p_all_orthmix_ * z1_sup, dim=-1).mean() + torch.sum(p_all_orthmix_ * z2_sup, dim=-1).mean())
                
                
                # SSL_loss_fastmoco += -0.5 * (( torch.sum(p_all_orthmix_ * z1.detach(), dim=-1)).mean() + (torch.sum(p_all_orthmix_ * z2.detach(), dim=-1)).mean() ) #-0.5 * ((torch.sum(p1_orthmix_ * z_all.detach(), dim=-1)).mean() + (torch.sum(p2_orthmix_ * z_all.detach(), dim=-1)).mean())

                # Sup_loss_fastmoco += -0.5 * (( torch.sum(p_all_orthmix_ * z1_sup, dim=-1)).mean() + (torch.sum(p_all_orthmix_ * z2_sup, dim=-1)).mean()) #-0.5 * ((torch.sum(p1_orthmix_ * z_all_sup, dim=-1)).mean() + (torch.sum(p2_orthmix_ * z_all_sup, dim=-1)*).mean())

            
            # SSL_loss_fastmoco1 = 0.5 * ((torch.sum(p1 * z_all.detach(), dim=-1)).mean() + (torch.sum(p2 * z_all.detach(), dim=-1)).mean())
            # Sup_loss_fastmoco1 = 0.5 * ((torch.sum(p1 * z_all_sup, dim=-1)).mean() + (torch.sum(p2 * z_all_sup, dim=-1)).mean())

            # Sup_loss_fastmoco += D(p1 , z2_sup, random_matrix, version='original') / 2 + D(p2 , z1_sup, random_matrix, version='original') /2

            
            loss_siam_fastmoco1 = 0.5 * SSL_loss_fastmoco / len(p1_orthmix) 
            loss_siam_fastmoco2 = 0.5 * Sup_loss_fastmoco / len(p1_orthmix) #+  0.5 * (SSL_loss_fastmoco1 + Sup_loss_fastmoco1)
            # print(0.5 * Sup_loss_fastmoco / len(p1_orthmix))
            # loss_siam_fastmoco = loss_siam_fastmoco1 + loss_siam_fastmoco2
            loss_siam_fastmoco = 0.5 * SSL_loss_fastmoco / len(p1_orthmix) + 0.5 * Sup_loss_fastmoco/ len(p1_orthmix) #+  0.5 * (SSL_loss_fastmoco1 + Sup_loss_fastmoco1)

            # loss_siam_fastmoco += 0.5 * (D(p_new_masked, z_ori_masked_c, random_matrix, version='random') / 2 + D(p_ori_masked_c, z_new_masked, random_matrix, version='random') / 2 + D(p_ori_masked_c , z2_sup, random_matrix, version='random') / + D(p_new_masked , z1_sup, random_matrix, version='random') /2 )

            # print(loss_sup ,loss_sup_mask, loss_siam_fastmoco )
            # loss =   loss_sup * self.cfg.SUP  + self.cfg.MASKED * loss_sup_mask * self.alpha  +  loss_siam_fastmoco * self.gamma  #+ self.alpha * (loss_unsup / self.scaling_factor)
            loss =     loss_sup * self.cfg.SUP  + self.cfg.MASKED * loss_sup_mask   +  loss_siam_fastmoco * self.gamma #*  self.alpha #+ self.alpha * (loss_unsup / self.scaling_factor)
            # print(loss)

        else:
            print(" Not FastMOCO")
            
            loss =   loss_sup  +  loss_siam *  self.alpha #+ self.alpha * (loss_unsup / self.scaling_factor)

        # criterion = nn.CrossEntropyLoss()  #nn.CrossEntropyLoss(reduction='none') # nn.CrossEntropyLoss()
        # loss_mix = F.mse_loss(features_x_mixed, mix_z_.detach())
        # mix_cate_loss = mixup_criterion(criterion,mix_result,labels,targets_b,lam)

        # loss =   loss_sup  +   loss_siam_fastmoco * 4.0  # + loss_unsup1_intra / self.scaling_factor)

        
        # loss =   loss_sup  +  (loss_siam + loss_siam_fastmoco) * 2.0 # + 1.0 * (loss_mix + 0.1 * mix_cate_loss) # + loss_unsup1_intra / self.scaling_factor)
        if self.cfg.DG_MODE =='DG':
            loss_dict['loss'] = loss.item()
            loss_dict['loss_sup'] = loss_sup.item()
            loss_dict['loss_sup_mask'] = loss_sup_mask.item()
            loss_dict['loss_siam_fastmoco1'] = loss_siam_fastmoco1.item()
            loss_dict['loss_siam_fastmoco2'] = loss_siam_fastmoco2.item()

        else:
            loss_dict['loss'] = loss.item()
            loss_dict['loss_sup'] = loss_sup.item()
            loss_dict['loss_siam'] = loss_siam.item()
            loss_dict['loss_siam_fastmoco'] = loss_siam_fastmoco.item()


        # loss_dict['loss_siam2'] = loss_siam2.item()

        # loss_dict['loss_sup2'] = loss_sup2.item()
        # loss_dict['loss_unsup2'] = loss_unsup2.item()
         
        return loss, loss_dict

    def update_alpha(self, iteration):
        # self.alpha = (1+iteration )/ self.max_iteration #* 0.5
        self.alpha = 1 - iteration / self.max_iteration # * 0.5

        return self.alpha
    



# Our loss function
class DahLoss_Siam_Fastmoco_v0_bsda(nn.Module):
    def __init__(self, cfg, max_iteration, training_domains, beta = 0.8, scaling_factor = 4, alpha = 1.0, temperature = 0.07,  fastmoco=1.0) -> None:
        super(DahLoss_Siam_Fastmoco_v0_bsda, self).__init__()
        self.max_iteration = max_iteration
        self.training_domains = training_domains
        self.alpha = alpha
        self.beta = beta
        self.scaling_factor = scaling_factor
        self.temperature = temperature
        self.gamma = fastmoco    
        self.cfg=cfg    
        self.domain_num_dict = {'MESSIDOR': 1744,
                                'IDRID': 516,
                                'DEEPDR': 2000,
                                'FGADR': 1842,
                                'APTOS': 3662,
                                'RLDR': 1593}
        
        self.label_num_dict = {'MESSIDOR': [1016, 269, 347, 75, 35],
                                'IDRID': [175, 26, 163, 89, 60],
                                'DEEPDR': [917, 214, 402, 353, 113],
                                'FGADR': [100, 211, 595, 646, 286],
                                'APTOS': [1804, 369, 999, 192, 294],
                                'RLDR': [165, 336, 929, 98, 62]}

        self.domain_prob, self.label_prob = self.get_domain_label_prob()
        self.domain_prob, self.label_prob = self.multinomial_soomthing(self.domain_prob, self.label_prob, self.beta)

        self.UnsupLoss = SupConLoss(temperature = self.temperature, reduction='none')
        self.SupLoss = nn.CrossEntropyLoss(reduction='none')
        self.SupLoss2 = nn.CrossEntropyLoss()
        self.per_cls_weights = torch.ones(5).cuda()
        
        # self.random_matrix = None
        # self.random_in_dim = 2048
        # self.random_out_dim = 2048
        
    def get_domain_label_prob(self):
        source_domain_num_list = torch.Tensor([self.domain_num_dict[domain] for domain in self.training_domains])
        source_domain_num = torch.sum(source_domain_num_list)
        domain_prob = source_domain_num_list / source_domain_num

        label_num_list = torch.Tensor([self.label_num_dict[domain] for domain in  self.training_domains]).sum(dim=0)
        label_num = torch.sum(label_num_list)
        label_prob = label_num_list / label_num

        return domain_prob.cuda(), label_prob.cuda()

    def multinomial_soomthing(self, domain_prob, label_prob, beta = 0.8):
        domain_prob = torch.pow(domain_prob, beta)
        label_prob = torch.pow(label_prob, beta)

        domain_prob = domain_prob / torch.sum(domain_prob)
        label_prob = label_prob / torch.sum(label_prob)

        return domain_prob, label_prob

    def get_weights_v2(self, epoch):
        # domain_prob = torch.index_select(self.domain_prob, 0, domains).cuda()
        # domain_weight = 1 / domain_prob
        # class_prob = torch.index_select(self.label_prob, 0, labels).cuda()
        # class_weight = 1 / class_prob
        cls_num_list = self.label_num_dict[self.training_domains[0]]
        idx = epoch // 80
        betas = [0, 0.9999]
        effective_num = 1.0 - np.power(betas[idx], cls_num_list)
        per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
        self.per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()
            
            
        return self.per_cls_weights
                         

    def get_weights(self, labels, domains):
        domain_prob = torch.index_select(self.domain_prob, 0, domains).cuda()
        domain_weight = 1 / domain_prob
        class_prob = torch.index_select(self.label_prob, 0, labels).cuda()
        class_weight = 1 / class_prob

        return domain_weight, class_weight

            
       
    def forward(self, output, features, features_masked, labels, domains, random_matrix):  
        
        # labels=labels 
        # domains=domains
        domain_weight, class_weight = self.get_weights(labels, domains)
        # domain_weight_a, class_weight_a = self.get_weights(labels_a, domains_a)

        loss_dict = {}

        output, output_new_a, output_new_masked = output
        # features_ori1, features_new1, features_ori2, features_new2 = features
        if self.gamma> 0:
            features_ori, features_new, z1,z2, z1_sup, z2_sup, p1, p2, z1_orthmix, z2_orthmix, p1_orthmix, p2_orthmix, z_all_orthmix, p_all_orthmix, z_all, z_all_sup, p_all = features
            z_new_masked, z_ori_masked_c, p_new_masked, p_ori_masked_c = features_masked
            # z_new_masked,z_ori_masked,p_new_masked, p_ori_masked=features_all
        else:
             z1,z2, z1_sup, z2_sup, p1, p2 = features

# features_ori, features_new, features_ori_z1, features_new_z2, z1_sup, z2_sup, p1, p2, z1_orthmix, z2_orthmix, p1_orthmix, p2_orthmix, z_all_orthmix, p_all_orthmix, features_all_z, z_all_sup, p_all

        loss_sup = 0
        self.SupLoss3 = LogitAdjust(self.label_num_dict[self.training_domains[0]],tau=0.5)    #### 使用逻辑补偿loss
        self.SupLoss4 = VSLoss(self.label_num_dict[self.training_domains[0]], gamma=0.1, tau=0.3)    #### 使用逻辑补偿loss

        # for op_item in output1:
        # loss_sup += self.SupLoss(output, labels)          
 
        # print(output_new_a.shape,  labels.repeat(10, ).shape)
        # # loss_sup +=  self.SupLoss3(output_new_a,  labels.repeat(1, ))       ###### 特征语义增强部分 
        # print("output_new_a", output_new_a.shape, "labels", labels.shape)   
        # print("output", output.shape, "labels", labels.shape)     
  
        loss_sup +=  self.SupLoss(output_new_a,  labels)       ###### 特征语义增强部分      


        loss_sup = torch.mean(loss_sup * class_weight * domain_weight) / (torch.mean(domain_weight) * torch.mean(class_weight)) 



        # loss_sup2 = 1.0 * self.SupLoss(output_new_a, labels_a)         
        
        # print(class_weight.shape, class_weight)

        temperature=1.0

        # loss_sup_mask = self.SupLoss3(output_new_masked, F.softmax( output / temperature, 1).detach())   # 
        loss_sup_mask = - 1  * (F.softmax( output / temperature, 1).detach() * F.log_softmax(output_new_masked / temperature, 1)).sum() / output.size()[0]

        # loss_sup_mask += - 1  * (F.softmax(output / temperature, 1).detach() * F.log_softmax(output_new_masked_freq / temperature, 1)).sum() / output.size()[0]
        
        

        # calculate loss
        SSL_loss = -0.5 * (torch.sum(p1 * z2.detach(), dim=-1) + torch.sum(p2 * z1.detach(), dim=-1))

        Sup_loss = -0.5 * (torch.sum(p1 * z2_sup, dim=-1) + torch.sum(p2 * z1_sup, dim=-1))

        loss_siam = 0.5 * SSL_loss.mean()  + 0.5 * Sup_loss.mean()
        
        
        
        features_multi = torch.stack([features_ori, features_new], dim = 1)
        features_multi = F.normalize(features_multi, p=2, dim=2)      
        
        loss_unsup = torch.mean(self.UnsupLoss(features_multi))
        
        
        
        # calculate loss
        # SSL_loss = -0.5 * (torch.sum(p_ori_masked * z_new_masked.detach(), dim=-1) + torch.sum(p_new_masked * z_ori_masked.detach(), dim=-1))

        # Sup_loss = -0.5 * (torch.sum(p_ori_masked * z2_sup, dim=-1) + torch.sum(p_new_masked * z1_sup, dim=-1))

        # loss_siam = 0.5 * SSL_loss.mean()  + 0.5 * Sup_loss.mean()
        
        
        
        if self.gamma> 0:
            # calculate loss for fastcoco
            SSL_loss_fastmoco=0.0
            Sup_loss_fastmoco=0.0
            
            # print(len(p1_orthmix)) # 6

            for i in range(len(p1_orthmix)):
                # z1_orthmix, z2_orthmix = nn.functional.normalize(z1_orthmix, dim=-1), nn.functional.normalize(z2_orthmix, dim=-1)
                # print(p1_orthmix, p1_orthmix[i].shape)
                # p1_orthmix_, p2_orthmix_ = p1_orthmix[i], p2_orthmix[i]

                
                p1_orthmix_, p2_orthmix_ = nn.functional.normalize(p1_orthmix[i], dim=-1), nn.functional.normalize(p2_orthmix[i], dim=-1)
                p_all_orthmix_ = nn.functional.normalize(p_all_orthmix[i], dim=-1)
                
                # SSL_loss_fastmoco += -0.5 * (torch.sum(p1_orthmix_ * z_all.detach(), dim=-1).mean() + torch.sum(p2_orthmix_ * z_all.detach(), dim=-1).mean()+ torch.sum(p_all_orthmix_ * z1.detach(), dim=-1).mean() + torch.sum(p_all_orthmix_ * z2.detach(), dim=-1).mean() )

                # Sup_loss_fastmoco += -0.5 * (torch.sum(p1_orthmix_ * z_all_sup, dim=-1).mean() + torch.sum(p2_orthmix_ * z_all_sup, dim=-1).mean() + torch.sum(p_all_orthmix_ * z1_sup, dim=-1).mean() + torch.sum(p_all_orthmix_ * z2_sup, dim=-1).mean())
                
                
                # SSL_loss_fastmoco += -0.5 * ((torch.sum(p1_orthmix_ * z2.detach(), dim=-1)).mean() + (torch.sum(p2_orthmix_ * z1.detach(), dim=-1)).mean() ) #+ torch.sum(p_all_orthmix_ * z1.detach(), dim=-1).mean() + torch.sum(p_all_orthmix_ * z2.detach(), dim=-1).mean() )

                # Sup_loss_fastmoco += -0.5 * ((torch.sum(p1_orthmix_ * z2_sup, dim=-1)).mean() + (torch.sum(p2_orthmix_ * z1_sup, dim=-1)).mean()) # + torch.sum(p_all_orthmix_ * z1_sup, dim=-1).mean() + torch.sum(p_all_orthmix_ * z2_sup, dim=-1).mean())
                

                # print(random_matrix)
                if self.cfg.DG_MODE =='DG':
                    SSL_loss_fastmoco += D(p1_orthmix_, z2, random_matrix, version='original') / 2 + D(p2_orthmix_, z1, random_matrix, version='original') / 2
                    
                    Sup_loss_fastmoco += D(p1_orthmix_ , z2_sup, random_matrix, version='original') / + D(p2_orthmix_ , z1_sup, random_matrix, version='original') /2
                else:
                    SSL_loss_fastmoco += D(p1_orthmix_, z2, random_matrix, version='random') / 2 + D(p2_orthmix_, z1, random_matrix, version='random') / 2
                    
                    Sup_loss_fastmoco += D(p1_orthmix_ , z2_sup, random_matrix, version='random') / + D(p2_orthmix_ , z1_sup, random_matrix, version='random') /2
                
            loss_siam_fastmoco = 0.5 * SSL_loss_fastmoco / len(p1_orthmix) + 0.5 * Sup_loss_fastmoco/ len(p1_orthmix)  

                # SSL_loss_fastmoco += -0.5 * ((torch.sum(p1_orthmix_ * z2.detach(), dim=-1)* class_weight * domain_weight).mean() + (torch.sum(p2_orthmix_ * z1.detach(), dim=-1)* class_weight * domain_weight).mean() ) #+ torch.sum(p_all_orthmix_ * z1.detach(), dim=-1).mean() + torch.sum(p_all_orthmix_ * z2.detach(), dim=-1).mean() )
                

                # Sup_loss_fastmoco += -0.5 * ((torch.sum(p1_orthmix_ * z2_sup, dim=-1)* class_weight * domain_weight).mean() + (torch.sum(p2_orthmix_ * z1_sup, dim=-1)* class_weight * domain_weight).mean()) # + torch.sum(p_all_orthmix_ * z1_sup, dim=-1).mean() + torch.sum(p_all_orthmix_ * z2_sup, dim=-1).mean())
                
                
                # SSL_loss_fastmoco += -0.5 * (( torch.sum(p_all_orthmix_ * z1.detach(), dim=-1)).mean() + (torch.sum(p_all_orthmix_ * z2.detach(), dim=-1)).mean() ) #-0.5 * ((torch.sum(p1_orthmix_ * z_all.detach(), dim=-1)).mean() + (torch.sum(p2_orthmix_ * z_all.detach(), dim=-1)).mean())

                # Sup_loss_fastmoco += -0.5 * (( torch.sum(p_all_orthmix_ * z1_sup, dim=-1)).mean() + (torch.sum(p_all_orthmix_ * z2_sup, dim=-1)).mean()) #-0.5 * ((torch.sum(p1_orthmix_ * z_all_sup, dim=-1)).mean() + (torch.sum(p2_orthmix_ * z_all_sup, dim=-1)*).mean())

            
            # SSL_loss_fastmoco1 = 0.5 * ((torch.sum(p1 * z_all.detach(), dim=-1)).mean() + (torch.sum(p2 * z_all.detach(), dim=-1)).mean())
            # Sup_loss_fastmoco1 = 0.5 * ((torch.sum(p1 * z_all_sup, dim=-1)).mean() + (torch.sum(p2 * z_all_sup, dim=-1)).mean())


            
            loss_siam_fastmoco = 0.5 * SSL_loss_fastmoco / len(p1_orthmix) + 0.5 * Sup_loss_fastmoco/ len(p1_orthmix) #+  0.5 * (SSL_loss_fastmoco1 + Sup_loss_fastmoco1)
    
            # loss_siam_fastmoco += 0.5 * (D(p_new_masked, z_ori_masked_c, random_matrix, version='random') / 2 + D(p_ori_masked_c, z_new_masked, random_matrix, version='random') / 2 + D(p_ori_masked_c , z2_sup, random_matrix, version='random') / + D(p_new_masked , z1_sup, random_matrix, version='random') /2 )

            # print(loss_sup ,loss_sup_mask, loss_siam_fastmoco )
            loss = 0.5 * loss_sup * self.cfg.MASKED + self.cfg.MASKED * loss_sup_mask +  loss_siam_fastmoco * self.gamma #* self.alpha #+ 1.0 * (loss_unsup / self.scaling_factor)
        else:
            print(" Not FastMOCO")
            
            loss =   loss_sup  +  loss_siam *  self.alpha #+ self.alpha * (loss_unsup / self.scaling_factor)

        # criterion = nn.CrossEntropyLoss()  #nn.CrossEntropyLoss(reduction='none') # nn.CrossEntropyLoss()
        # loss_mix = F.mse_loss(features_x_mixed, mix_z_.detach())
        # mix_cate_loss = mixup_criterion(criterion,mix_result,labels,targets_b,lam)

        # loss =   loss_sup  +   loss_siam_fastmoco * 4.0  # + loss_unsup1_intra / self.scaling_factor)

        
        # loss =   loss_sup  +  (loss_siam + loss_siam_fastmoco) * 2.0 # + 1.0 * (loss_mix + 0.1 * mix_cate_loss) # + loss_unsup1_intra / self.scaling_factor)

        loss_dict['loss'] = loss.item()
        loss_dict['loss_sup'] = loss_sup.item()
        loss_dict['loss_siam'] = loss_siam.item()
        loss_dict['loss_siam_fastmoco'] = loss_siam_fastmoco.item()

        # loss_dict['loss_siam2'] = loss_siam2.item()

        # loss_dict['loss_sup2'] = loss_sup2.item()
        # loss_dict['loss_unsup2'] = loss_unsup2.item()
         
        return loss, loss_dict

    def update_alpha(self, iteration):
        self.alpha = 1 - iteration / self.max_iteration * 0.5
        return self.alpha
    
    
    
class VSLoss(nn.Module):

    def __init__(self, cls_num_list, gamma=0.3, tau=0.5, weight=None):
        super(VSLoss, self).__init__()

        cls_probs = [cls_num / sum(cls_num_list) for cls_num in cls_num_list]
        temp = (1.0 / np.array(cls_num_list)) ** gamma
        temp = temp / np.min(temp)

        iota_list = tau * np.log(cls_probs)
        Delta_list = temp

        self.iota_list = torch.cuda.FloatTensor(iota_list)
        self.Delta_list = torch.cuda.FloatTensor(Delta_list)
        self.weight = weight

    def forward(self, x, target):
        output = x / self.Delta_list + self.iota_list

        return F.cross_entropy(output, target, weight=self.weight)
    
    
# Our loss function
class DahLoss_Siam_Fastmoco_Dual(nn.Module):
    def __init__(self, cfg, max_iteration, training_domains, beta = 0.8, scaling_factor = 4, alpha = 1.0, temperature = 0.07,  fastmoco=1.0) -> None:
        super(DahLoss_Siam_Fastmoco, self).__init__()
        self.max_iteration = max_iteration
        self.training_domains = training_domains
        self.alpha = alpha
        self.beta = beta
        self.scaling_factor = scaling_factor
        self.temperature = temperature
        self.gamma = fastmoco    
        self.cfg=cfg    
        self.domain_num_dict = {'MESSIDOR': 1744,
                                'IDRID': 516,
                                'DEEPDR': 2000,
                                'FGADR': 1842,
                                'APTOS': 3662,
                                'RLDR': 1593}
        
        self.label_num_dict = {'MESSIDOR': [1016, 269, 347, 75, 35],
                                'IDRID': [175, 26, 163, 89, 60],
                                'DEEPDR': [917, 214, 402, 353, 113],
                                'FGADR': [100, 211, 595, 646, 286],
                                'APTOS': [1804, 369, 999, 192, 294],
                                'RLDR': [165, 336, 929, 98, 62]}

        self.domain_prob, self.label_prob = self.get_domain_label_prob()
        self.domain_prob, self.label_prob = self.multinomial_soomthing(self.domain_prob, self.label_prob, self.beta)

        self.UnsupLoss = SupConLoss(temperature = self.temperature, reduction='none')
        self.SupLoss = nn.CrossEntropyLoss(reduction='none')
        self.SupLoss2 = nn.CrossEntropyLoss()


    def get_domain_label_prob(self):
        source_domain_num_list = torch.Tensor([self.domain_num_dict[domain] for domain in self.training_domains])
        source_domain_num = torch.sum(source_domain_num_list)
        domain_prob = source_domain_num_list / source_domain_num

        label_num_list = torch.Tensor([self.label_num_dict[domain] for domain in  self.training_domains]).sum(dim=0)
        label_num = torch.sum(label_num_list)
        label_prob = label_num_list / label_num

        return domain_prob.cuda(), label_prob.cuda()

    def multinomial_soomthing(self, domain_prob, label_prob, beta = 0.8):
        domain_prob = torch.pow(domain_prob, beta)
        label_prob = torch.pow(label_prob, beta)

        domain_prob = domain_prob / torch.sum(domain_prob)
        label_prob = label_prob / torch.sum(label_prob)

        return domain_prob, label_prob

    def get_weights(self, labels, domains):
        domain_prob = torch.index_select(self.domain_prob, 0, domains).cuda()
        domain_weight = 1 / domain_prob
        class_prob = torch.index_select(self.label_prob, 0, labels).cuda()
        class_weight = 1 / class_prob

        return domain_weight, class_weight
                            
    def forward(self, output, features, labels, domains):  
        
        labels,labels_a=labels 
        domains,domains_a=domains
        domain_weight, class_weight = self.get_weights(labels, domains)
        domain_weight_a, class_weight_a = self.get_weights(labels_a, domains_a)

        loss_dict = {}

        output, output_new_a, output_new_masked = output
        # features_ori1, features_new1, features_ori2, features_new2 = features
        if self.gamma> 0:
            features_ori, features_new, z1,z2, z1_sup, z2_sup, p1, p2, z1_orthmix, z2_orthmix, p1_orthmix, p2_orthmix, features_x_mixed, mix_z_, mix_result, targets_b, lam = features
            # z_new_masked,z_ori_masked,p_new_masked, p_ori_masked=features_all
        else:
             z1,z2, z1_sup, z2_sup, p1, p2 = features

# features_ori_z1, features_new_z2, z1_sup, z2_sup, p1, p2
        loss_sup = 0


        # for op_item in output1:
        loss_sup += self.SupLoss2(output, labels)            
        loss_sup2 = 1.0 * self.SupLoss(output_new_a, labels_a)         

        temperature=1.0

        loss_sup_mask = - 1  * (F.softmax( output_new_a / temperature, 1).detach() * F.log_softmax(output_new_masked / temperature, 1)).sum() / output.size()[0]

        # loss_sup_mask += - 1  * (F.softmax(output / temperature, 1).detach() * F.log_softmax(output_new_masked_freq / temperature, 1)).sum() / output.size()[0]
        
        
        # loss_sup = torch.mean(loss_sup * class_weight * domain_weight) / (torch.mean(domain_weight) * torch.mean(class_weight)) 
        
        # print(class_weight)
        loss_sup2 = torch.mean(loss_sup2 * class_weight_a * domain_weight_a) / (torch.mean(domain_weight_a) * torch.mean(class_weight_a)) 


        # features_multi = torch.stack([features_ori, features_new], dim = 1)
        # features_multi = F.normalize(features_multi, p=2, dim=2)      
        
        # loss_unsup = torch.mean(self.UnsupLoss(features_multi))
        

        # calculate loss
        SSL_loss = -0.5 * (torch.sum(p1 * z2.detach(), dim=-1) + torch.sum(p2 * z1.detach(), dim=-1))

        Sup_loss = -0.5 * (torch.sum(p1 * z2_sup, dim=-1) + torch.sum(p2 * z1_sup, dim=-1))

        loss_siam = 0.5 * SSL_loss.mean()  + 0.5 * Sup_loss.mean()
        
        
        # calculate loss
        # SSL_loss = -0.5 * (torch.sum(p_ori_masked * z_new_masked.detach(), dim=-1) + torch.sum(p_new_masked * z_ori_masked.detach(), dim=-1))

        # Sup_loss = -0.5 * (torch.sum(p_ori_masked * z2_sup, dim=-1) + torch.sum(p_new_masked * z1_sup, dim=-1))

        # loss_siam = 0.5 * SSL_loss.mean()  + 0.5 * Sup_loss.mean()
        
        
        
        if self.gamma> 0:
            # calculate loss for fastcoco
            SSL_loss_fastmoco=0.0
            Sup_loss_fastmoco=0.0
            
            # print(len(p1_orthmix)) # 6

            for i in range(len(p1_orthmix)):
                # z1_orthmix, z2_orthmix = nn.functional.normalize(z1_orthmix, dim=-1), nn.functional.normalize(z2_orthmix, dim=-1)
                # print(p1_orthmix, p1_orthmix[i].shape)
                p1_orthmix_, p2_orthmix_ = nn.functional.normalize(p1_orthmix[i], dim=-1), nn.functional.normalize(p2_orthmix[i], dim=-1)
                SSL_loss_fastmoco += -0.5 * (torch.sum(p1_orthmix_ * z2.detach(), dim=-1).mean() + torch.sum(p2_orthmix_ * z1.detach(), dim=-1).mean())

                Sup_loss_fastmoco += -0.5 * (torch.sum(p1_orthmix_ * z2_sup, dim=-1).mean() + torch.sum(p2_orthmix_ * z1_sup, dim=-1).mean())

            loss_siam_fastmoco = 0.5 * SSL_loss_fastmoco / len(p1_orthmix) + 0.5 * Sup_loss_fastmoco/ len(p1_orthmix)
            
            # print(loss_sup ,loss_sup_mask, loss_siam_fastmoco )
            loss =   (loss_sup+loss_sup2) * 0.5 + self.cfg.MASKED * loss_sup_mask +  loss_siam_fastmoco * self.gamma #*  self.alpha # + self.alpha * (loss_unsup / self.scaling_factor)
        else:
            print(" Not FastMOCO")
            loss =   loss_sup  +  loss_siam *  self.alpha #+ self.alpha * (loss_unsup / self.scaling_factor)

        # criterion = nn.CrossEntropyLoss()  #nn.CrossEntropyLoss(reduction='none') # nn.CrossEntropyLoss()
        # loss_mix = F.mse_loss(features_x_mixed, mix_z_.detach())
        # mix_cate_loss = mixup_criterion(criterion,mix_result,labels,targets_b,lam)

        # loss =   loss_sup  +   loss_siam_fastmoco * 4.0  # + loss_unsup1_intra / self.scaling_factor)

        
        # loss =   loss_sup  +  (loss_siam + loss_siam_fastmoco) * 2.0 # + 1.0 * (loss_mix + 0.1 * mix_cate_loss) # + loss_unsup1_intra / self.scaling_factor)

        loss_dict['loss'] = loss.item()
        loss_dict['loss_sup'] = loss_sup.item()
        loss_dict['loss_siam'] = loss_siam.item()
        loss_dict['loss_siam_fastmoco'] = loss_siam_fastmoco.item()

        # loss_dict['loss_siam2'] = loss_siam2.item()

        # loss_dict['loss_sup2'] = loss_sup2.item()
        # loss_dict['loss_unsup2'] = loss_unsup2.item()
         
        return loss, loss_dict

    def update_alpha(self, iteration):
        self.alpha = 1 - iteration / self.max_iteration * 0.5
        return self.alpha
    
# Our loss function
class DahLoss_Dual_Siam(nn.Module):
    def __init__(self, max_iteration, training_domains, beta = 0.8, scaling_factor = 4, alpha = 1.0, temperature = 0.07) -> None:
        super(DahLoss_Dual_Siam, self).__init__()
        self.max_iteration = max_iteration
        self.training_domains = training_domains
        self.alpha = alpha
        self.beta = beta
        self.scaling_factor = scaling_factor
        self.temperature = temperature
        
        self.domain_num_dict = {'MESSIDOR': 1744,
                                'IDRID': 516,
                                'DEEPDR': 2000,
                                'FGADR': 1842,
                                'APTOS': 3662,
                                'RLDR': 1593}
        
        self.label_num_dict = {'MESSIDOR': [1016, 269, 347, 75, 35],
                                'IDRID': [175, 26, 163, 89, 60],
                                'DEEPDR': [917, 214, 402, 353, 113],
                                'FGADR': [100, 211, 595, 646, 286],
                                'APTOS': [1804, 369, 999, 192, 294],
                                'RLDR': [165, 336, 929, 98, 62]}

        self.domain_prob, self.label_prob = self.get_domain_label_prob()
        self.domain_prob, self.label_prob = self.multinomial_soomthing(self.domain_prob, self.label_prob, self.beta)

        self.UnsupLoss = SupConLoss(temperature = self.temperature, reduction='none')
        self.SupLoss = nn.CrossEntropyLoss(reduction='none')

    def get_domain_label_prob(self):
        source_domain_num_list = torch.Tensor([self.domain_num_dict[domain] for domain in self.training_domains])
        source_domain_num = torch.sum(source_domain_num_list)
        domain_prob = source_domain_num_list / source_domain_num

        label_num_list = torch.Tensor([self.label_num_dict[domain] for domain in  self.training_domains]).sum(dim=0)
        label_num = torch.sum(label_num_list)
        label_prob = label_num_list / label_num

        return domain_prob.cuda(), label_prob.cuda()

    def multinomial_soomthing(self, domain_prob, label_prob, beta = 0.8):
        domain_prob = torch.pow(domain_prob, beta)
        label_prob = torch.pow(label_prob, beta)

        domain_prob = domain_prob / torch.sum(domain_prob)
        label_prob = label_prob / torch.sum(label_prob)

        return domain_prob, label_prob

    def get_weights(self, labels, domains):
        domain_prob = torch.index_select(self.domain_prob, 0, domains).cuda()
        domain_weight = 1 / domain_prob
        class_prob = torch.index_select(self.label_prob, 0, labels).cuda()
        class_weight = 1 / class_prob

        return domain_weight, class_weight
                            
    def forward(self, output, features, labels, domains):  
        
        domain_weight, class_weight = self.get_weights(labels, domains)

        loss_dict = {}

        output1, output2, output_new_masked1, output_new_masked2 = output
        # features_ori1, features_new1, features_ori2, features_new2 = features
        z11,z12, z11_sup, z12_sup, p11, p12, z21,z22, z21_sup, z22_sup, p21, p22 = features

        # print(output1.shape, output2.shape, labels.shape)


        loss_sup1 = 0
        loss_sup2 = 0


        # for op_item in output1:
        loss_sup1 += self.SupLoss(output1, labels)            

        # for op_item in output2:
        loss_sup2 += self.SupLoss(output2, labels)    
            
        temperature=1.0
        # loss_sup1 += - 1  * (F.softmax(output2 / temperature, 1).detach() * \
        #                                         F.log_softmax(output1 / temperature, 1)).sum() / \
        #                          output1.size()[0]


        # loss_sup2 += - 1 * (F.softmax(output1 / temperature, 1).detach() * \
        #                                             F.log_softmax(output2 / temperature, 1)).sum() / \
        #                              output2.size()[0]

        
        loss_sup1_mask = - 1  * (F.softmax(output2 / temperature, 1).detach() * \
                                                F.log_softmax(output_new_masked1 / temperature, 1)).sum() #/ output1.size()[0]


        loss_sup2_mask = - 1 * (F.softmax(output1 / temperature, 1).detach() * \
                                                    F.log_softmax(output_new_masked2 / temperature, 1)).sum() #/ output2.size()[0]
        

        # calculate loss
        SSL_loss1 = -0.5 * (torch.sum(p11 * z22.detach(), dim=-1) + torch.sum(p12 * z21.detach(), dim=-1))
        SSL_loss2 = -0.5 * (torch.sum(p21 * z12.detach(), dim=-1) + torch.sum(p22 * z11.detach(), dim=-1))

        Sup_loss1 = -0.5 * (torch.sum(p11 * z22_sup, dim=-1) + torch.sum(p12 * z21_sup, dim=-1))
        Sup_loss2 = -0.5 * (torch.sum(p21 * z12_sup, dim=-1) + torch.sum(p22 * z11_sup, dim=-1))

        loss_siam1 = 0.5 * SSL_loss1.mean()  + 0.5 * Sup_loss1.mean()
        
        loss_siam2 = 0.5 * SSL_loss2.mean()  + 0.5 * Sup_loss2.mean()

        
        
        # features_multi1 = torch.stack([features_ori1, features_new2], dim = 1)
        # features_multi1 = F.normalize(features_multi1, p=2, dim=2)      
        
        # loss_unsup1 = torch.mean(self.UnsupLoss(features_multi1))
        
        # features_multi2 = torch.stack([features_ori2, features_new1], dim = 1)
        # features_multi2 = F.normalize(features_multi2, p=2, dim=2)      
        
        # loss_unsup2 = torch.mean(self.UnsupLoss(features_multi2))  ### 没有使用labels

        # features_multi = torch.stack([features_ori1, features_new1, features_ori2, features_new2], dim = 1)
        # features_multi = F.normalize(features_multi, p=2, dim=2)      

        # loss_unsup = torch.mean(self.UnsupLoss(features_multi))  ### 没有使用labels

        
        loss_sup1 = torch.mean(loss_sup1 * class_weight * domain_weight) / (torch.mean(domain_weight) * torch.mean(class_weight)) + loss_sup1_mask

        loss_sup2 = torch.mean(loss_sup2 * class_weight * domain_weight) / (torch.mean(domain_weight) * torch.mean(class_weight)) + loss_sup2_mask

        # loss1 = (1 - self.alpha) * loss_sup1 + self.alpha * (loss_unsup1 / self.scaling_factor) # + loss_unsup1_intra / self.scaling_factor)
        # loss2 = (1 - self.alpha) * loss_sup2 + self.alpha * (loss_unsup2 / self.scaling_factor) # + loss_unsup2_intra / self.scaling_factor)
        

        # loss1 = (1 - self.alpha) * loss_sup1 * 0.5  + self.alpha * loss_siam1 / 200  # + loss_unsup1_intra / self.scaling_factor)
        # loss2 = (1 - self.alpha) * loss_sup2 * 0.5  + self.alpha * loss_siam2 / 200   # + loss_unsup2_intra / self.scaling_factor)
        
        # loss1 = loss_sup1 * 0.5  + ( 1- self.alpha) * loss_siam1 / 500  # + loss_unsup1_intra / self.scaling_factor)
        # loss2 = loss_sup2 * 0.5  + ( 1- self.alpha) * loss_siam2 / 500   # + loss_unsup2_intra / self.scaling_factor)
        
        # loss1 =  ( 1- self.alpha)  * loss_sup1  + self.alpha * loss_siam1   # + loss_unsup1_intra / self.scaling_factor)
        # loss2 =( 1- self.alpha) * loss_sup2   +  self.alpha * loss_siam2   # + loss_unsup2_intra / self.scaling_factor)
        
        loss1 =  loss_sup1  +  4 * loss_siam1   # + loss_unsup1_intra / self.scaling_factor)
        loss2 = loss_sup2   +  4 * loss_siam2   # + loss_unsup2_intra / self.scaling_factor)
        # loss1 = loss_sup1 * 0.5  + self.alpha * loss_siam1 / 1000 # + loss_unsup1_intra / self.scaling_factor)
        # loss2 = loss_sup2 * 0.5  + self.alpha * loss_siam2 / 1000 # + loss_unsup2_intra / self.scaling_factor)
        
        loss=loss1+loss2 #+  self.alpha * (loss_unsup / self.scaling_factor)

        # print(loss)

        loss_dict['loss'] = loss.item()
        loss_dict['loss_sup1'] = loss_sup1.item()
        loss_dict['loss_siam1'] = loss_siam1.item()
        loss_dict['loss_siam2'] = loss_siam2.item()

        loss_dict['loss_sup2'] = loss_sup2.item()
        # loss_dict['loss_unsup2'] = loss_unsup2.item()
         
        return loss, loss_dict

    def update_alpha(self, iteration):
        self.alpha = 1 - iteration / self.max_iteration
        return self.alpha
    
    

def similarity_cross_entropy(similarities, num_pos):
    # modified from vince/loss_util.py

    # assert mask.shape == similarities.shape
    # log similarity over (self + all other entries as denom)
    row_maxes = torch.max(similarities, dim=-1, keepdim=True)[0]
    scaled_similarities = similarities - row_maxes

    pos_similarities = scaled_similarities[:, :num_pos]
    neg_similarities = scaled_similarities[:, num_pos:]

    neg_similarities_exp = torch.exp(neg_similarities).sum(-1, keepdim=True)

    pos_similarities_exp = torch.exp(pos_similarities)
    similarity_log_softmax = pos_similarities - torch.log(pos_similarities_exp + neg_similarities_exp)
    dists = -similarity_log_softmax

    loss = dists.mean()
    return loss

    
    
class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, reduction = 'mean'):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.reduction = reduction

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]  ###  4
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  ### 
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # print(anchor_count,features.shape, anchor_feature.shape, contrast_feature.shape) # 4 torch.Size([16, 4, 512]) torch.Size([64, 512]) torch.Size([64, 512])
        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count) ## 
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        # print(anchor_dot_contrast.shape,logits_mask.shape) ## torch.Size([64, 64]) torch.Size([64, 64])
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
                
        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        
        if self.reduction == 'mean':
            loss = loss.view(anchor_count, batch_size).mean()
        else:
            loss = loss.view(anchor_count, batch_size)

        return loss


# Our loss function
class DahLoss_Dual_BalSCL(nn.Module):
    def __init__(self, max_iteration, training_domains, beta = 0.8, scaling_factor = 4, alpha = 1.0, temperature = 0.07) -> None:
        super(DahLoss_Dual_BalSCL, self).__init__()
        self.max_iteration = max_iteration
        self.training_domains = training_domains
        self.alpha = alpha
        self.beta = beta
        self.scaling_factor = scaling_factor
        self.temperature = temperature
        
        self.domain_num_dict = {'MESSIDOR': 1744,
                                'IDRID': 516,
                                'DEEPDR': 2000,
                                'FGADR': 1842,
                                'APTOS': 3662,
                                'RLDR': 1593}
        
        self.label_num_dict = {'MESSIDOR': [1016, 269, 347, 75, 35],
                                'IDRID': [175, 26, 163, 89, 60],
                                'DEEPDR': [917, 214, 402, 353, 113],
                                'FGADR': [100, 211, 595, 646, 286],
                                'APTOS': [1804, 369, 999, 192, 294],
                                'RLDR': [165, 336, 929, 98, 62]}

        self.domain_prob, self.label_prob = self.get_domain_label_prob()
        self.domain_prob, self.label_prob = self.multinomial_soomthing(self.domain_prob, self.label_prob, self.beta)

        self.UnsupLoss = BalSCL(5, temperature = self.temperature)
        self.SupLoss = nn.CrossEntropyLoss(reduction='none')

    def get_domain_label_prob(self):
        source_domain_num_list = torch.Tensor([self.domain_num_dict[domain] for domain in self.training_domains])
        source_domain_num = torch.sum(source_domain_num_list)
        domain_prob = source_domain_num_list / source_domain_num

        label_num_list = torch.Tensor([self.label_num_dict[domain] for domain in  self.training_domains]).sum(dim=0)
        label_num = torch.sum(label_num_list)
        label_prob = label_num_list / label_num

        return domain_prob.cuda(), label_prob.cuda()

    def multinomial_soomthing(self, domain_prob, label_prob, beta = 0.8):
        domain_prob = torch.pow(domain_prob, beta)
        label_prob = torch.pow(label_prob, beta)

        domain_prob = domain_prob / torch.sum(domain_prob)
        label_prob = label_prob / torch.sum(label_prob)

        return domain_prob, label_prob

    def get_weights(self, labels, domains):
        domain_prob = torch.index_select(self.domain_prob, 0, domains).cuda()
        domain_weight = 1 / domain_prob
        class_prob = torch.index_select(self.label_prob, 0, labels).cuda()
        class_weight = 1 / class_prob

        return domain_weight, class_weight
                            
    def forward(self, output, features, labels, domains):  
        
        domain_weight, class_weight = self.get_weights(labels, domains)
        
        self.SupLoss3 = LogitAdjust(self.label_num_dict[self.training_domains[0]])    #### 使用逻辑补偿loss

        loss_dict = {}

        output1, output2 = output
        features1_mlp, centers_logits1, features2_mlp, centers_logits2 = features
        
        # print(output1.shape, output2.shape, labels.shape)


        loss_sup1 = 0
        loss_sup2 = 0


        # for op_item in output1:
        loss_sup1 += self.SupLoss3(output1, labels)            

        # for op_item in output2:
        loss_sup2 += self.SupLoss3(output2, labels)    
            
        temperature=1.0
        loss_sup1 += - 1  * (F.softmax(output2 / temperature, 1).detach() * \
                                                F.log_softmax(output1 / temperature, 1)).sum() / \
                                 output1.size()[0]


        loss_sup2 += - 1 * (F.softmax(output1 / temperature, 1).detach() * \
                                                    F.log_softmax(output2 / temperature, 1)).sum() / \
                                     output2.size()[0]

        
        
        # features_multi1_intra = torch.stack([features_ori1, features_new1], dim = 1)
        # features_multi1_intra = F.normalize(features_multi1_intra, p=2, dim=2)      
        
        # loss_unsup1_intra = torch.mean(self.UnsupLoss(features_multi1_intra))
        
        # features_multi2_intra = torch.stack([features_ori2, features_new2], dim = 1)
        # features_multi2_intra = F.normalize(features_multi2_intra, p=2, dim=2)      
        
        # loss_unsup2_intra = torch.mean(self.UnsupLoss(features_multi2_intra))
        
        
        loss_unsup1 = self.UnsupLoss(centers_logits1, features1_mlp, labels)
        
        loss_unsup2 = self.UnsupLoss(centers_logits2, features2_mlp, labels)


        # features_multi1 = F.normalize(features_multi1, p=2, dim=2)      
        
        # loss_unsup1 = torch.mean(self.UnsupLoss(features_multi1))
        
        # features_multi2 = BalSCL([features_ori2, features_new1], dim = 1)
        # features_multi2 = F.normalize(features_multi2, p=2, dim=2)      
        
        # loss_unsup2 = torch.mean(self.UnsupLoss(features_multi2))

        
        # loss_sup1 = torch.mean(loss_sup1 * class_weight * domain_weight) / (torch.mean(domain_weight) * torch.mean(class_weight))

        # loss_sup2 = torch.mean(loss_sup2 * class_weight * domain_weight) / (torch.mean(domain_weight) * torch.mean(class_weight))

        # loss1 = (1 - self.alpha) * loss_sup1 + self.alpha * (loss_unsup1 / self.scaling_factor) #+ 0.5 * loss_sup1_dis # + loss_unsup1_intra / self.scaling_factor)
        # loss2 = (1 - self.alpha) * loss_sup2 + self.alpha * (loss_unsup2 / self.scaling_factor) #+  0.5 *  loss_sup2_dis # + loss_unsup2_intra / self.scaling_factor)
        
        # loss1 = self.alpha * loss_sup1 + 1.0 * (loss_unsup1 / self.scaling_factor) #+ 0.5 * loss_sup1_dis # + loss_unsup1_intra / self.scaling_factor)
        # loss2 = self.alpha * loss_sup2 + 1.0 * (loss_unsup2 / self.scaling_factor) #+  0.5 *  loss_sup2_dis # + loss_unsup2_intra / self.scaling_factor)
        
        loss1 = loss_sup1 + 1.0 * (loss_unsup1 / self.scaling_factor) #+ 0.5 * loss_sup1_dis # + loss_unsup1_intra / self.scaling_factor)
        loss2 = loss_sup2 + 1.0 * (loss_unsup2 / self.scaling_factor) #+  0.5 *  loss_sup2_dis # + loss_unsup2_intra / self.scaling_factor)
        
        
        loss=loss1+loss2


        loss_dict['loss'] = loss.item()
        loss_dict['loss_sup1'] = loss_sup1.item()
        loss_dict['loss_unsup1'] = loss_unsup1.item()
        loss_dict['loss_sup2'] = loss_sup2.item()
        loss_dict['loss_unsup2'] = loss_unsup2.item()
         
        return loss, loss_dict

    def update_alpha(self, iteration):
        self.alpha = 1 - iteration / self.max_iteration
        return self.alpha

class BalSCL(nn.Module):
    def __init__(self, cls_num_list=None, temperature=0.1):
        super(BalSCL, self).__init__()
        self.temperature = temperature
        self.cls_num_list = 5

    def forward(self, centers1, features, targets, ):
        
        # print(centers1.shape, features.shape, targets.shape)  # torch.Size([2048, 1024]) torch.Size([16, 2, 1024]) torch.Size([16])

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        batch_size = features.shape[0]
        targets = targets.contiguous().view(-1, 1)
        targets_centers = torch.arange(self.cls_num_list).view(-1, 1).to(device)
        targets = torch.cat([targets.repeat(2, 1), targets_centers], dim=0)
        batch_cls_count = torch.eye(self.cls_num_list).to(device)[targets].sum(dim=0).squeeze()

        mask = torch.eq(targets[:2 * batch_size], targets.T).float().to(device)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * 2).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        
        # class-complement
        features = torch.cat(torch.unbind(features, dim=1), dim=0)
        features = torch.cat([features, centers1], dim=0)
        logits = features[:2 * batch_size].mm(features.T)
        logits = torch.div(logits, self.temperature)

        # For numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # class-averaging
        exp_logits = torch.exp(logits) * logits_mask
        per_ins_weight = torch.tensor([batch_cls_count[i] for i in targets], device=device).view(1, -1).expand(
            2 * batch_size, 2 * batch_size + self.cls_num_list) - mask
        exp_logits_sum = exp_logits.div(per_ins_weight).sum(dim=1, keepdim=True)
        
        log_prob = logits - torch.log(exp_logits_sum)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - mean_log_prob_pos
        loss = loss.view(2, batch_size).mean()
        return loss
