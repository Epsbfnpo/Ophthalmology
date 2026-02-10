# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

def flatten(t):
    return t.reshape(t.shape[0], -1)

class MoCo(nn.Module):
    """
    Build a MoCo model with a base encoder, a momentum encoder, and two MLPs
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=256, mlp_dim=4096,K=65536, T=1.0,feat_dim=2048,normalize=False,num_classes=7):
        """
        dim: feature dimension (default: 256)
        mlp_dim: hidden dimension in MLPs (default: 4096)
        T: softmax temperature (default: 1.0)
        """
        super(MoCo, self).__init__()

        self.T = T
        self.K=K
        # build encoders

        self.base_encoder = base_encoder(num_classes=mlp_dim)
        hidden_dim = self.base_encoder.head.weight.shape[1]
        self.base_projector = self._build_mlp(3, hidden_dim, mlp_dim, dim)
        self.linear = nn.Linear(feat_dim, num_classes)

        self._build_projector_and_predictor_mlps(dim, mlp_dim)

        # create the queue
        self.register_buffer("queue", torch.randn(K, dim))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_l", torch.randint(0, num_classes, (K,)))

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.normalize = normalize


    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)

    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        pass


    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels,accelerator):
        # gather keys before updating queue
        keys = accelerator.gather(keys)
        labels =  accelerator.gather(labels)
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[ptr:ptr + batch_size,:] = keys
        self.queue_l[ptr:ptr + batch_size] = labels

        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr


    def _train(self,im_q, im_k,im_q_2,im_k_2,labels,ratio,accelerator):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        if im_q_2 ==None:
            q = self.base_encoder.forward_features(im_q)[:,0]  # queries: NxC
        else:
            q = self.base_encoder.forward_features(im_q,im_q_2,ratio[0])[:,0]  # queries: NxC
        feat_avg_q=q
        q=self.predictor(self.base_projector(q))
        q = nn.functional.normalize(q, dim=1)
        logits_q = self.linear(feat_avg_q)

        if im_k_2 ==None:
            k= self.base_encoder.forward_features(im_k)[:,0]
        else:
            k = self.base_encoder.forward_features(im_k,im_k_2,ratio[1])[:,0]  # keys: NxC
        k = self.base_projector(k)  # keys: NxC
        k = nn.functional.normalize(k, dim=1)
        logits_k = self.linear(feat_avg_q)


        features = torch.cat((q, k, self.queue.clone().detach()), dim=0)  # 2NxC, N samples, C=feature dimension (default: 128)
        target = torch.cat((labels, labels, self.queue_l.clone().detach()), dim=0)
        self._dequeue_and_enqueue(k, labels,accelerator)
        logits = torch.cat((logits_q, logits_k), dim=0)

        return features, target, logits

    def _inference(self, image):
        q = self.base_encoder.forward_features(image)[:,0]
        feat_avg_q=q
        encoder_q_logits = self.linear(feat_avg_q)
        return encoder_q_logits

    def forward(self, im_q, im_k=None,im_q_2=None,im_k_2=None, labels=None,ratio=None,accelerator=None):
        if self.training:
            return self._train(im_q, im_k,im_q_2,im_k_2,labels,ratio,accelerator)
        else:
            return self._inference(im_q)



class MoCo_ResNet(MoCo):
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = self.base_encoder.fc.weight.shape[1]
        del self.base_encoder.fc # remove original fc layer

        # projectors
        self.base_encoder.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)

        # predictor
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim, False)


class MoCo_ViT(MoCo):
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = self.base_encoder.head.weight.shape[1]
        del self.base_encoder.head # remove original fc layer

        # projectors
        # self.base_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)
        # self.momentum_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)
        # predictor
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim)


