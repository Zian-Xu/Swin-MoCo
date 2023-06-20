import torch
import torch.nn as nn
import torch.nn.functional


class MoCo(nn.Module):
    def __init__(self, base_encoder, dim=256, mlp_dim=4096, T=1.0):
        super(MoCo, self).__init__()

        self.T = T
        self.base_encoder = base_encoder(num_classes=mlp_dim)
        self.momentum_encoder = base_encoder(num_classes=mlp_dim)
        self._build_projector_and_predictor_mlps(dim, mlp_dim)

        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data.copy_(param_b.data)
            param_m.requires_grad = False

    @staticmethod
    def _build_mlp(num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for index in range(num_layers):
            dim1 = input_dim if index == 0 else mlp_dim
            dim2 = output_dim if index == num_layers - 1 else mlp_dim
            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if index < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)

    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        pass

    def contrastive_loss(self, q, k):
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        # negative cosine similarity
        # return -(q * k).sum()
        logits = torch.einsum('nc,mc->nm', q, k) / self.T
        N = logits.shape[0]
        labels = (torch.arange(N, dtype=torch.long)).cuda()
        return nn.CrossEntropyLoss()(logits, labels) * (2 * self.T)

    @torch.no_grad()
    def _update_momentum_encoder(self, m):
        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)

    def forward(self, x1, x2, m):
        q1 = self.predictor(self.base_encoder(x1))
        q2 = self.predictor(self.base_encoder(x2))

        with torch.no_grad():
            self._update_momentum_encoder(m)
            k1 = self.momentum_encoder(x1)
            k2 = self.momentum_encoder(x2)

        return self.contrastive_loss(q1, k2) + self.contrastive_loss(q2, k1)


class SwinMoCo(MoCo):
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = self.base_encoder.head.weight.shape[1]
        del self.base_encoder.head, self.momentum_encoder.head
        self.base_encoder.head = self._build_mlp(num_layers=3, input_dim=hidden_dim, mlp_dim=mlp_dim, output_dim=dim)
        self.momentum_encoder.head = self._build_mlp(
            num_layers=3, input_dim=hidden_dim, mlp_dim=mlp_dim, output_dim=dim)
        self.predictor = self._build_mlp(num_layers=2, input_dim=dim, mlp_dim=mlp_dim, output_dim=dim)
