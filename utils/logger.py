import os
import csv

class Logger:
    def __init__(self, args):#初始化
        #保存结果
        if args.save_dir is None:
            result_dir = './save/'
        else:
            result_dir = './save/{}/'.format(args.save_dir)

        result_f = 'RFL_{}_EP{}_C{}_LBS[{}]_LE[{}]_IID[{}]_LR[{}]_MMT[{}]_NT[{}]_NR[{}]_TPL[{}]'.format(
            args.dataset,
            args.epochs,
            args.frac,
            args.local_bs,
            args.local_ep,
            args.iid,
            args.lr,
            args.momentum,
            args.noise_type,
            args.noise_rate,
            args.T_pl,#设置了一个文件名，包括数据集名称、训练周期数、分数、本地批大小、本地周期数、是否IID、学习率、动量、噪声类型、噪声率以及某个参数（T_pl）
        )
         
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)#创建目录
        self.result_path = os.path.join(result_dir, result_f + ".csv")
        self.wr = None

    def write(self, epoch, train_acc, train_loss, test_acc, test_loss):
        if self.wr is None:#检查self.wr是否为None
            self.f = open(self.result_path, 'w', newline='')
            self.wr = csv.writer(self.f)
            self.wr.writerow(['epoch', 'train_acc', 'train_loss', 'test_acc', 'test_loss'])
        
        data = [epoch, train_acc, train_loss, test_acc, test_loss]
        print("Writing data:", data)
        self.wr.writerow(data)
        self.f.flush()# 立即将数据写入文件，而不等到调用 close() 方法

    def close(self):
        if self.wr is not None:
            self.f.close()

