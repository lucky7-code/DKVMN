"""
Usage:
    run.py  [options]

Options:
    --length=<int>                      max length of question sequence [default: 50]
    --questions=<int>                   num of question [default: 100]
    --lr=<float>                        learning rate [default: 0.001]
    --bs=<int>                          batch size [default: 64]
    --seed=<int>                        random seed [default: 59]
    --epochs=<int>                      number of epochs [default: 30]
    --cuda=<int>                        use GPU id [default: 0]
    --final_fc_dim=<int>                dimension of final dim [default: 10]
    --question_dim=<int>                dimension of question dim[default: 50]
    --question_and_answer_dim=<int>     dimension of question and answer dim [default: 100]
    --memory_size=<int>               memory size [default: 20]
    --model=<string>                    model type [default: DKVMN]
"""

import os
import random
import logging
import torch
import torch.optim as optim
import numpy as np
from datetime import datetime
from docopt import docopt
from data.dataloader import getDataLoader
from evaluation import eval


def setup_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():

    args = docopt(__doc__)
    length = int(args['--length'])
    questions = int(args['--questions'])
    lr = float(args['--lr'])
    bs = int(args['--bs'])
    seed = int(args['--seed'])
    epochs = int(args['--epochs'])
    cuda = args['--cuda']
    final_fc_dim = int(args['--final_fc_dim'])
    question_dim = int(args['--question_dim'])
    question_and_answer_dim = int(args['--question_and_answer_dim'])
    memory_size = int(args['--memory_size'])
    model_type = args['--model']

    logger = logging.getLogger('main')
    logger.setLevel(level=logging.DEBUG)
    date = datetime.now()
    handler = logging.FileHandler(
        f'log/{date.year}_{date.month}_{date.day}_{model_type}_result.log')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.info('DKVMN')
    logger.info(list(args.items()))

    setup_seed(seed)

    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    trainLoader, validationLoader, testLoader = getDataLoader(bs, questions, length)
    from model.model import MODEL
    model = MODEL(n_question=questions, batch_size=bs, q_embed_dim=question_dim, qa_embed_dim=question_and_answer_dim,
                  memory_size=memory_size, final_fc_dim=final_fc_dim)
    model.init_params()
    model.init_embeddings()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_auc = 0
    for epoch in range(epochs):
        print('epoch: ' + str(epoch+1))
        model, optimizer = eval.train_epoch(model, trainLoader, optimizer, device)
        logger.info(f'epoch {epoch+1}')
        auc = eval.test_epoch(model, validationLoader, device)
        if auc > best_auc:
            print('best checkpoint')
            torch.save({'state_dict': model.state_dict()}, 'checkpoint/'+model_type+'.pth.tar')
            best_auc = auc
    eval.test_epoch(model, testLoader, device, ckpt='checkpoint/'+model_type+'.pth.tar')


if __name__ == '__main__':
    main()
