import argparse, os
import torch
import random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from vdsr import Net
from dataset import DatasetFromHdf5

# Training settings
parser = argparse.ArgumentParser(description="PyTorch VDSR")
parser.add_argument("--batchSize", type=int, default=1, help="Training batch size")
parser.add_argument("--nEpochs", type=int, default=500, help="Number of epochs to train for")
parser.add_argument("--lr", type=float, default=1, help="Learning Rate. Default=0.1")
parser.add_argument("--step", type=int, default=3, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10")
parser.add_argument("--cuda", action="store_true", help="Use cuda?")
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--clip", type=float, default=0.4, help="Clipping Gradients. Default=0.4")
parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, help="Weight decay, Default: 1e-4")
parser.add_argument('--pretrained', default='', type=str, help='path to pretrained model (default: none)')
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")

def main():
    print("in loop")
    global opt, model
    opt = parser.parse_args()
    print(opt)

    cuda = opt.cuda
    if cuda:
        print("=> use gpu id: '{}'".format(opt.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
        if not torch.cuda.is_available():
                raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True

    print("===> Loading datasets")
    #train_set = DatasetFromHdf5("data/train.h5")   #改下面一行自己包的H5檔案
    #train_set = DatasetFromHdf5("D:/mytestfile_41x41_all_small_x2.h5") #自己包的
    train_set = DatasetFromHdf5("D:/train.h5") #作者的
    
    training_data_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=opt.batchSize, shuffle=True) #num_workers=opt.threads改成0

    print("===> Building model")
    model = Net() #重新訓練用 若繼續續練改下一行加載pth預訓練檔
    
    #model = torch.load("checkpoint/model_epoch_lr01_1.pth", map_location=lambda storage, loc: storage)["model"] #預訓練檔 會報錯改下一行寫法
    
    criterion = nn.MSELoss(size_average=False)

    print("===> Setting GPU")
    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint["epoch"] + 1
            model.load_state_dict(checkpoint["model"].state_dict())
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    # optionally copy weights from a checkpoint
    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            print("=> loading model '{}'".format(opt.pretrained))
            weights = torch.load(opt.pretrained)
            model.load_state_dict(weights['model'].state_dict())
        else:
            print("=> no model found at '{}'".format(opt.pretrained))  

    print("===> Setting Optimizer")
    optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)

    print("===> Training")
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        #print("============train()前=========================")
        train(training_data_loader, optimizer, model, criterion, epoch)
        #print("============train()後=========================")
        save_checkpoint(model, epoch)

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = opt.lr * (0.1 ** (epoch // opt.step))
    print("lr=",lr)
    return lr

def train(training_data_loader, optimizer, model, criterion, epoch):
    lr = adjust_learning_rate(optimizer, epoch-1)
    # lr=1
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    print("Epoch = {}, lr = {}".format(epoch, optimizer.param_groups[0]["lr"]))

    model.train()
    print("============model.train()後=========================")
    for iteration, batch in enumerate(training_data_loader, 1):
        input, target = Variable(batch[0]), Variable(batch[1], requires_grad=False)
       # print("============input, target 後========================= lr=="+str(lr))
        if opt.cuda:
            input = input.cuda()
            target = target.cuda()

        loss = criterion(model(input), target)
        optimizer.zero_grad()
        loss.backward() 
        nn.utils.clip_grad_norm(model.parameters(),opt.clip) 
        optimizer.step()
        # print("{}".format(epoch))
        # print("{}".format( iteration))
        # print("{}".format( len(training_data_loader)))
        # print("{}".format( loss.data.item()))  #替換loss.data[0]為loss.data.item()
        if iteration%50 == 0:
            print("===> Epoch[{}]({}/{}): Loss: {:.30f}".format(epoch, iteration, len(training_data_loader), loss.data.item()))  #替換loss.data[0]為loss.data.item()
            print("lr="+str(lr))
        # if iteration%30000 == 0:
        #     lr=lr/10
        #     save_checkpoint(model, epoch)
def save_checkpoint(model, epoch):
    model_out_path = "checkpoint/" + "model_epoch_{}.pth".format(epoch)
    state = {"epoch": epoch ,"model": model}
    if not os.path.exists("checkpoint/"):
        os.makedirs("checkpoint/")

    torch.save(state, model_out_path)

    print("Checkpoint saved to {}".format(model_out_path))

if __name__ == "__main__":
    main()