import torch
from torch.nn import functional


def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (sequence_length.unsqueeze(1).expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand

def soft_cross_entropy_loss(predict_score,label_score):
    log_softmax = torch.nn.LogSoftmax(dim = 1)
    softmax = torch.nn.Softmax(dim = 1)

    predict_prob_log = log_softmax(predict_score).float()
    label_prob = softmax(label_score).float()


    loss_elem = -label_prob * predict_prob_log
    loss = loss_elem.sum(dim = 1)
    return loss

def soft_target_loss(logits, soft_target,length):
    if torch.cuda.is_available():
        length = torch.LongTensor(length).cuda()
    else:
        length = torch.LongTensor(length)
    loss_total = []
    for predict,label in zip(logits.split(1,dim=1),soft_target.split(1,dim=1)):
        predict = predict.squeeze()
        label = label.squeeze()
        loss_t = soft_cross_entropy_loss(predict,label)
        loss_total.append(loss_t)
    loss_total = torch.stack(loss_total,dim=0).transpose(1,0)
    #loss_total = loss_total.sum(dim=1)
    loss_total = loss_total.sum() / length.float().sum()
    return loss_total

def cosine_sim(logits, logits_1):
    return torch.ones(logits.size(0)).cuda() + torch.cosine_similarity(logits, logits_1, dim=1).cuda()

def cosine_loss(logits, logits_1,length):
    if torch.cuda.is_available():
        length = torch.LongTensor(length).cuda()
    else:
        length = torch.LongTensor(length)
    loss_total = []
    for predict,label in zip(logits.split(1,dim=1),logits_1.split(1,dim=1)):
        predict = predict.squeeze()
        label = label.squeeze()
        loss_t = cosine_sim(predict,label)
        loss_total.append(loss_t)
    loss_total = torch.stack(loss_total,dim=0).transpose(1,0)
    #loss_total = loss_total.sum(dim=1)
    loss_total = loss_total.sum() / length.float().sum()
    return loss_total

def masked_cross_entropy(logits, target, length):
    if torch.cuda.is_available():
        length = torch.LongTensor(length).cuda()
    else:
        length = torch.LongTensor(length)
    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.
    Returns:
        loss: An average loss value masked by the length.
    """

    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = functional.log_softmax(logits_flat, dim=1)
    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)
    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)

    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size())
    # mask: (batch, max_len)
    mask = sequence_mask(sequence_length=length, max_len=target.size(1))
    losses = losses * mask.float()
    loss = losses.sum() / length.float().sum()
    # if loss.item() > 10:
    #     print(losses, target)
    return loss


def masked_cross_entropy_without_logit(logits, target, length):
    if torch.cuda.is_available():
        length = torch.LongTensor(length).cuda()
    else:
        length = torch.LongTensor(length)
    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.
    Returns:
        loss: An average loss value masked by the length.
    """

    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))

    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = torch.log(logits_flat + 1e-12)

    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)
    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)

    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size())

    # mask: (batch, max_len)
    mask = sequence_mask(sequence_length=length, max_len=target.size(1))
    losses = losses * mask.float()
    loss = losses.sum() / length.float().sum()
    # if loss.item() > 10:
    #     print(losses, target)
    return loss

