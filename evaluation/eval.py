import tqdm
import torch
import logging
import os
from sklearn import metrics
logger = logging.getLogger('main.eval')

def __load_model__(ckpt):
    '''
    ckpt: Path of the checkpoint
    return: Checkpoint dict
    '''
    if os.path.isfile(ckpt):
        checkpoint = torch.load(ckpt)
        print("Successfully loaded checkpoint '%s'" % ckpt)
        return checkpoint
    else:
        raise Exception("No checkpoint found at '%s'" % ckpt)


def train_epoch(model, trainLoader, optimizer, device):
    model.to(device)
    for batch in tqdm.tqdm(trainLoader, desc='Training:    ', mininterval=2):
        batch = batch.to(device)
        datas = torch.chunk(batch, 3, 2)
        optimizer.zero_grad()
        loss, prediction, ground_truth = model(datas[0].squeeze(2), datas[1].squeeze(2), datas[2])
        loss.backward()
        optimizer.step()
    return model, optimizer


def test_epoch(model, testLoader, device, ckpt=None):
    model.to(device)
    if ckpt is not None:
        checkpoint = __load_model__(ckpt)
        model.load_state_dict(checkpoint['state_dict'])
    ground_truth = torch.tensor([], device=device)
    prediction = torch.tensor([], device=device)
    for batch in tqdm.tqdm(testLoader, desc='Testing:     ', mininterval=2):
        batch = batch.to(device)
        datas = torch.chunk(batch, 3, 2)
        loss, p, label = model(datas[0].squeeze(2), datas[1].squeeze(2), datas[2])
        prediction = torch.cat([prediction, p])
        ground_truth = torch.cat([ground_truth, label])
    acc = metrics.accuracy_score(torch.round(ground_truth).detach().cpu().numpy(), torch.round(prediction).detach().cpu().numpy())
    auc = metrics.roc_auc_score(ground_truth.detach().cpu().numpy(), prediction.detach().cpu().numpy())
    logger.info('auc: ' + str(auc) + ' acc: ' + str(acc))
    print('auc: ' + str(auc) + ' acc: ' + str(acc))
    return auc
