import numpy as np
import itertools
from sklearn.model_selection import KFold


class DataReader():
    def __init__(self, train_path, test_path, maxstep, num_ques):
        self.train_path = train_path
        self.test_path = test_path
        self.maxstep = maxstep
        self.num_ques = num_ques

    def getData(self, file_path):
        datas = []
        with open(file_path, 'r') as file:
            for len, ques, ans in itertools.zip_longest(*[file] * 3):
                len = int(len.strip().strip(','))
                ques = [int(q) for q in ques.strip().strip(',').split(',')]
                ans = [int(a) for a in ans.strip().strip(',').split(',')]
                slices = len//self.maxstep + (1 if len % self.maxstep > 0 else 0)
                for i in range(slices):
                    data = np.zeros(shape=[self.maxstep, 3])  # 0 ->question and answer(1->)
                    if len > 0:                               # 1->question (1->)
                        if len >= self.maxstep:               # 2->label (0->1, 1->2)
                            steps = self.maxstep
                        else:
                            steps = len
                        for j in range(steps):
                            data[j][0] = ques[i * self.maxstep + j] + 1
                            data[j][2] = ans[i * self.maxstep + j] + 1
                            if ans[i * self.maxstep + j] == 1:
                                data[j][1] = ques[i * self.maxstep + j] + 1
                            else:
                                data[j][1] = ques[i * self.maxstep + j] + self.num_ques + 1
                        len = len - self.maxstep
                    datas.append(data.tolist())
            print('done: ' + str(np.array(datas).shape))
        return datas

    def getTrainData(self):
        print('loading train data...')
        kf = KFold(n_splits=5, shuffle=True, random_state=3)
        Data = np.array(self.getData(self.train_path))
        for train_indexes, vali_indexes in kf.split(Data):
            valiData = Data[vali_indexes].tolist()
            trainData = Data[train_indexes].tolist()
        return np.array(trainData), np.array(valiData)

    def getTestData(self):
        print('loading test data...')
        testData = self.getData(self.test_path)
        return np.array(testData)
