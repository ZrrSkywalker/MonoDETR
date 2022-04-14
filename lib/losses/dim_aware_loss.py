import torch
import torch.nn.functional as F

def dim_aware_l1_loss(input, target, dimension):
    dimension = dimension.clone().detach()

    loss = torch.abs(input - target)
    loss /= dimension

    with torch.no_grad():
        compensation_weight = F.l1_loss(input, target) / loss.mean()
    loss *= compensation_weight

    return loss.mean()


if __name__ == '__main__':
    input = torch.zeros(3, 3, 3)
    target = torch.Tensor(range(27)).reshape(3, 3, 3)


    print(dim_aware_l1_loss(input, target, target+1))