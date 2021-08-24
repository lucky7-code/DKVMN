import torch
import torch.utils.data as Data
from .readdata import DataReader
#assist2015/assist2015_train.txt assist2015/assist2015_test.txt
#assist2017/assist2017_train.txt assist2017/assist2017_test.txt
#assist2009/builder_train.csv assist2009/builder_test.csv


def getDataLoader(batch_size, num_of_questions, max_step):
    handle = DataReader('../dataset/assist2015/assist2015_train.txt',
                        '../dataset/assist2015/assist2015_test.txt', max_step,
                        num_of_questions)
    train, vali = handle.getTrainData()
    dtrain = torch.tensor(train.astype(int).tolist(), dtype=torch.long)
    dvali = torch.tensor(vali.astype(int).tolist(), dtype=torch.long)
    dtest = torch.tensor(handle.getTestData().astype(int).tolist(),
                         dtype=torch.long)
    trainLoader = Data.DataLoader(dtrain, batch_size=batch_size, shuffle=True)
    valiLoader = Data.DataLoader(dvali, batch_size=batch_size, shuffle=True)
    testLoader = Data.DataLoader(dtest, batch_size=batch_size, shuffle=False)
    return trainLoader, valiLoader, testLoader