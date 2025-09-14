import torch
class evaluation_metrics():
    def __init__(self,MODE=('BCHW','BHW')):
        self.MODE=MODE
        self.lists=[]

    # 多次评价
    def evaluate_batch(self, prediction, ground_truth):
        self.lists=self.lists+self.evaluate_single_image(prediction, ground_truth, self.MODE)

    def get_results(self):
        tp=[0,0,0,0,0]
        fn=[0,0,0,0,0]
        fp=[0,0,0,0,0]
        f1=[0,0,0,0,0]
        for list in self.lists:
            for j,pingjiazhibiaomen in enumerate(list):
                tp[j]=tp[j]+pingjiazhibiaomen[0]
                fn[j]=fn[j]+pingjiazhibiaomen[1]
                fp[j]=fp[j]+pingjiazhibiaomen[2]
                f1[j]=f1[j]+pingjiazhibiaomen[3]
        precision = [tp[j] / (tp[j] + fp[j]) if (tp[j] + fp[j]) > 0 else torch.tensor(0).to(tp[j]) for j in range(5)]
        recall = [tp[j] / (tp[j] + fn[j]) if (tp[j] + fn[j]) > 0 else torch.tensor(0).to(tp[j]) for j in range(5)]
        f1_avg = [f1[j]/len(self.lists) if len(self.lists)>0 else torch.tensor(0).to(f1[j]) for j in range(5)]
        f1 = [2 * precision[j] * recall[j] / (precision[j] + recall[j]) if (precision[j] + recall[j]) > 0 else torch.tensor(0).to(precision[j]) for j in range(5)]
        self.lists=[]
        return f1,f1_avg

    @staticmethod
    def evaluate_single_image(prediction,ground_truth,MODE=('BCHW','BHW'),softmax=False):
        # 输入参数：prediction
        # ground_truth
        # 一种最简单的办法就是把维度都对上！然后for循环遍历
        if MODE[0]=='BCHW':
            # (B,C,H,W)->(B,H,W)
            assert MODE[1]=='BHW' or MODE[1]=='BCHW'
            prediction = torch.argmax(torch.softmax(prediction, dim=1), dim=1) if softmax else torch.argmax(prediction, dim=1)
            if MODE[1]=='BCHW':
                ground_truth = torch.argmax(ground_truth, dim=1)
        elif MODE[0]=='CHW':
            # (C,H,W)->(H,W)
            assert MODE[1]=='HW' or MODE[1]=='CHW'
            prediction = torch.argmax(torch.softmax(prediction)) if softmax else torch.argmax(prediction)
            if MODE[1]=='CHW':
                ground_truth = torch.argmax(ground_truth)
        # 到这里，目的就是为了把C消掉
        prediction = prediction.to(ground_truth)
        lists=[]
        assert prediction.shape == ground_truth.shape
        if 'B' in MODE[0]:
            # B,H,W
            assert (len(prediction.shape) == 3)
            for i in range(prediction.shape[0]):
                # for B
                # 图片维度
                image_classes = []
                for j in range(5):
                    tp = ((prediction[i] == j) & (ground_truth[i] == j)).sum()
                    fn = ((prediction[i] != j) & (ground_truth[i] == j)).sum()
                    fp = ((prediction[i] == j) & (ground_truth[i] != j)).sum()
                    precision = tp / (tp + fp) if (tp + fp) > 0 else torch.tensor(0).to(tp)
                    recall = tp / (tp + fn) if (tp + fn) > 0 else torch.tensor(0).to(tp)
                    f1 = 2 * precision * recall / (precision+recall) if (precision+recall) > 0 else torch.tensor(0).to(tp)
                    image_classes.append([tp,fn,fp,f1])
                lists.append(image_classes)
        else:
            # H,W
            assert (len(prediction.shape) == 2)
            image_classes = []
            for j in range(5):
                tp = ((prediction == j) & (ground_truth == j)).sum()
                fn = ((prediction != j) & (ground_truth == j)).sum()
                fp = ((prediction == j) & (ground_truth != j)).sum()
                precision = tp / (tp + fp) if (tp + fp) > 0 else torch.tensor(0.).to(tp)
                recall = tp / (tp + fn) if (tp + fn) > 0 else torch.tensor(0.).to(tp)
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else torch.tensor(0.).to(tp)
                image_classes.append([tp,fn,fp,f1])
            lists.append(image_classes)
        return lists

    def evaluate_loader(self, model, dataloader):
        model.eval()
        with torch.no_grad():
            for batch in dataloader:
                image_batch, label_batch = batch['image'], batch['label']
                image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
                outputs = model(image_batch)
                self.evaluate_batch(outputs.detach(), label_batch.detach())
        model.train()
        return self.get_results()
