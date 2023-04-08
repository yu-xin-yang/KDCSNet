# -*-coding:utf-8-*-
# ! /usr/bin/env python
import csv
from math import log10
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from model.StudentNet import StudentNet
from model.AMCSNet import RFAN
from utils.utility import progress_bar


class KDCSNet_Trainer(object):
    def __init__(self, config, training_loader, testing_loader1=None, testing_loader2=None, testing_loader3=None):
        super(KDCSNet_Trainer, self).__init__()
        self.criterionL1 = torch.nn.L1Loss()
        self.criterionMSE = torch.nn.MSELoss()
        self.CUDA = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.CUDA else 'cpu')
        self.TeacherNet = None
        self.lr = config.lr
        self.output_path = './saved_model/'
        self.nEpochs = config.nEpochs
        self.sampling_rate = config.samplingRate
        self.sampling_point = config.samplingPoint
        self.Teacher_optimizer = None
        self.Teacher_scheduler = None
        self.seed = config.seed
        self.training_loader = training_loader
        self.testing_loaderset5 = testing_loader1
        self.testing_loaderset11 = testing_loader2
        self.testing_loaderset14 = testing_loader3
        self.num_blocks = config.resBlock
        self.step_size = config.step_size
        self.gamma = config.gamma
        self.pretrain = config.use_pretrained_model
        self.pretrain_model_path = config.use_pretrained_model_path
        self.saved_excel_path = config.save_result_path

    def pretrained_model(self):
        self.TeacherNet = torch.load(self.pretrain_model_path)
        self.StudentNet = StudentNet(num_features=self.sampling_point, num_blocks=self.num_blocks).to(self.device)
        self.StudentNet.weight_init(mean=0.0, std=0.02)
        torch.manual_seed(self.seed)
        if self.CUDA:
            torch.cuda.manual_seed(self.seed)
            cudnn.benchmark = True
            self.criterionL1.cuda()
            self.criterionMSE.cuda()

        self.Student_optimizer = torch.optim.Adam(self.StudentNet.parameters(), lr=self.lr)
        self.Student_scheduler = torch.optim.lr_scheduler.StepLR(self.Student_optimizer, step_size=self.step_size,
                                                                 gamma=self.gamma)

    def Student_train(self):
        self.StudentNet.train()
        train_loss = 0
        for batch_num, (data, target) in enumerate(self.training_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.Student_optimizer.zero_grad()
            # loss = self.criterionMSE(self.StudentNet(data), data)
            loss = self.distillation(data=data)
            train_loss += loss.item()
            loss.backward()

            self.Student_optimizer.step()
            progress_bar(batch_num, len(self.training_loader), 'Loss: %.6f' % (train_loss / (batch_num + 1)))
        return format(train_loss / len(self.training_loader))

    def getAvgTensor(self, tensor):
        out = torch.zeros(len(tensor[0][0]), len(tensor[0][0])).cuda()
        for i in range(len(tensor[0])):
            out += tensor[0][i]
        return out / len(tensor[0])

    def distillation(self, data):
        # self.TeacherNet.eval()
        self.StudentNet.eval()
        student_output = self.StudentNet(data)
        # teacher_output = teacher_output.detach()
        loss0 = self.criterionMSE(student_output, data)
        loss1 = self.criterionMSE(
            self.getAvgTensor(self.TeacherNet.get_first_out(data)),
            self.getAvgTensor(self.StudentNet.get_first_out(data)))
        loss2 = self.criterionMSE(
            self.getAvgTensor(self.TeacherNet.get_second_out(data)),
            self.getAvgTensor(self.StudentNet.get_second_out(data)))
        loss3 = self.criterionMSE(
            self.getAvgTensor(self.TeacherNet.get_third_out(data)),
            self.getAvgTensor(self.StudentNet.get_third_out(data)))

        loss = loss0 + loss1 + loss2 + loss3
        print("loos0:{}; loss1:{}; loss2:{}; loss3:{}".format(loss0, loss1, loss2, loss3))
        return loss

    def testset11(self):
        if self.testing_loaderset11 is None:
            return 0
        self.StudentNet.eval()
        avg_psnr = 0
        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.testing_loaderset11):
                data, target = data.to(self.device), target.to(self.device)
                prediction = self.StudentNet(data)
                mse = self.criterionMSE(prediction, data)
                psnr = 10 * log10(1 / mse.item())
                avg_psnr += psnr
                progress_bar(batch_num, len(self.testing_loaderset11), 'PSNR: %.4f' % (avg_psnr / (batch_num + 1)))
        return format(avg_psnr / len(self.testing_loaderset11))

    def run(self):
        self.pretrained_model()
        f = open(self.saved_excel_path, "w", encoding='utf-8', newline="")
        csv_writer = csv.writer(f)
        csv_writer.writerow(["Epoch", "loss", "Set11"])
        max_psnr = 0.0
        for epoch in range(1, self.nEpochs + 1):
            result = []
            print("\n===> Epoch {} starts:".format(epoch))
            loss = self.Student_train()
            avg_psnr_set11 = self.testset11()
            result.append(epoch)
            result.append(loss)
            result.append(avg_psnr_set11)
            csv_writer.writerow(result)
            self.Student_scheduler.step()
            if float(avg_psnr_set11) > max_psnr:
                model_out_path = self.output_path + "Student_" + str(epoch) + ".pth"
                torch.save(self.StudentNet, model_out_path)
                max_psnr = float(avg_psnr_set11)
        f.close()
