import os
import torch


def load_weight(model, device, path):
    if not path:
        print('No pre-train weights are used.')
        return model
    # 读入当前模型需要的所有参数
    model_dict = model.state_dict()
    # 分情况读入预训练模型
    assert os.path.exists(path)
    full_dict = torch.load(path, map_location=device)
    if 'model' in full_dict.keys():  # 有监督迁移
        # 去掉所有带layer_up的参数
        full_dict = full_dict['model']
        for key in list(full_dict):
            if key.startswith('layers_up'):
                del full_dict[key]
        # 调整layer_up部分的参数名称
        for key in list(full_dict):
            if key.startswith('layers'):
                current_layer_index = 2 - int(key[7:8])
                if current_layer_index >= 0:
                    current_key = 'layers_up.' + str(current_layer_index) + key[8:]
                    full_dict[current_key] = full_dict[key]
    else:  # 无监督迁移
        # 只保留基编码器权重
        for key in list(full_dict):
            if key.startswith('base_encoder'):
                full_dict[key[13:]] = full_dict.pop(key)
            else:
                del full_dict[key]
        # 去除 head 权重
        for key in list(full_dict):
            if key.startswith('head'):
                del full_dict[key]
        # 调整layer_up部分的参数名称
        for key in list(full_dict):
            if key.startswith('layers') and 'downsample' not in key:
                current_layer_index = 2 - int(key[7:8])
                if current_layer_index >= 0:
                    current_key = 'layers_up.' + str(current_layer_index) + key[8:]
                    full_dict[current_key] = full_dict[key]
    # 检查是否有名称一样但是大小不匹配的冲突权重，如果有则删除
    for k in list(full_dict.keys()):
        if k in model_dict:
            if full_dict[k].shape != model_dict[k].shape:
                print(f"Delete: '{k}'; "
                      f"Weight shape: {full_dict[k].shape}; Model shape: {model_dict[k].shape}")
                del full_dict[k]
    # 将权重赋值给模型，并输出赋值结果
    result = model.load_state_dict(full_dict, strict=False)  # strict=False允许dict不必包含所有model中的参数
    print(f'missing_keys: {result.missing_keys}')
    print(f'unexpected_keys: {result.unexpected_keys}')
    return model


class DiceLoss(torch.nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    @staticmethod
    def _dice_loss(score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(
            inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes
