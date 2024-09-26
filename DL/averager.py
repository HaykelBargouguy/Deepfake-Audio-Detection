import torch
class Averager:
    """ statistics for classification """
    def __init__(self):
        self.num_correct = 0.0
        self.num_total = 0.0
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update_acc(self, logits, truth):
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()  # [B, ]
        labels = truth.detach().cpu().numpy()  # [B, ]
        self.num_correct += sum(preds == labels)
        self.num_total += len(truth)

    def update_loss(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (self.count + 1e-10)

    def measure(self):
        accuracy = (self.num_correct / self.num_total)
        loss = self.avg
        return accuracy, loss

    def report(self, intro):
        accuracy, loss = self.measure()
        text = "{}: Accuracy = {:.4f} loss = {:.3f}\n".format(intro, accuracy, loss)
        print(text, end='')