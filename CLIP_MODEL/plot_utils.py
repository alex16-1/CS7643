import matplotlib.pyplot as plt
import numpy as np
import os

class TrainingMonitor:
    def __init__(self, output_dir="./plots"):
        self.output_dir = output_dir
        self.epochs = []
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []
        self.bleu1_scores = []
        self.bleu2_scores = []
        self.bleu3_scores = []
        self.bleu4_scores = []
        self.meteor_scores = []
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def update(self, epoch, train_loss, train_acc, val_loss=None, val_acc=None, 
               bleu1=None, bleu2=None, bleu3=None, bleu4=None, meteor=None):
        self.epochs.append(epoch)
        self.train_loss.append(train_loss)
        self.train_acc.append(train_acc)
        if val_loss is not None:
            self.val_loss.append(val_loss)
        if val_acc is not None:
            self.val_acc.append(val_acc)
        if bleu1 is not None:
            self.bleu1_scores.append(bleu1)
        if bleu2 is not None:
            self.bleu2_scores.append(bleu2)
        if bleu3 is not None:
            self.bleu3_scores.append(bleu3)
        if bleu4 is not None:
            self.bleu4_scores.append(bleu4)
        if meteor is not None:
            self.meteor_scores.append(meteor)
    
    def plot_loss(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.epochs, self.train_loss, 'b-', label='Training Loss')
        if len(self.val_loss) > 0:
            plt.plot(self.epochs[:len(self.val_loss)], self.val_loss, 'r-', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, 'loss_plot.png'))
        plt.close()
    
    def plot_accuracy(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.epochs, self.train_acc, 'b-', label='Training Accuracy')
        if len(self.val_acc) > 0:
            plt.plot(self.epochs[:len(self.val_acc)], self.val_acc, 'r-', label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, 'accuracy_plot.png'))
        plt.close()
    
    def plot_bleu(self):
        if len(self.bleu1_scores) > 0:
            plt.figure(figsize=(10, 5))
            if len(self.bleu1_scores) > 0:
                plt.plot(self.epochs[:len(self.bleu1_scores)], self.bleu1_scores, 'b-', label='BLEU-1')
            if len(self.bleu2_scores) > 0:
                plt.plot(self.epochs[:len(self.bleu2_scores)], self.bleu2_scores, 'g-', label='BLEU-2')
            if len(self.bleu3_scores) > 0:
                plt.plot(self.epochs[:len(self.bleu3_scores)], self.bleu3_scores, 'r-', label='BLEU-3')
            if len(self.bleu4_scores) > 0:
                plt.plot(self.epochs[:len(self.bleu4_scores)], self.bleu4_scores, 'c-', label='BLEU-4')
            plt.title('BLEU Scores Progression')
            plt.xlabel('Epoch')
            plt.ylabel('BLEU Score')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(self.output_dir, 'bleu_plot.png'))
            plt.close()
    
    def plot_meteor(self):
        if len(self.meteor_scores) > 0:
            plt.figure(figsize=(10, 5))
            plt.plot(self.epochs[:len(self.meteor_scores)], self.meteor_scores, 'm-', label='METEOR')
            plt.title('METEOR Score Progression')
            plt.xlabel('Epoch')
            plt.ylabel('METEOR Score')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(self.output_dir, 'meteor_plot.png'))
            plt.close()
    
    def plot_combined_metrics(self):
        if len(self.bleu4_scores) > 0 or len(self.meteor_scores) > 0:
            plt.figure(figsize=(12, 6))
            if len(self.bleu4_scores) > 0:
                plt.plot(self.epochs[:len(self.bleu4_scores)], self.bleu4_scores, 'c-', label='BLEU-4')
            if len(self.meteor_scores) > 0:
                plt.plot(self.epochs[:len(self.meteor_scores)], self.meteor_scores, 'm-', label='METEOR')
            plt.title('Caption Quality Metrics')
            plt.xlabel('Epoch')
            plt.ylabel('Score')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(self.output_dir, 'metrics_plot.png'))
            plt.close()
    
    def plot_all(self):
        self.plot_loss()
        self.plot_accuracy()
        self.plot_bleu()
        self.plot_meteor()
        self.plot_combined_metrics()
        self.save_metrics_to_csv()
    
    def save_metrics_to_csv(self):
        import csv
        header = ['Epoch', 'Train Loss', 'Train Acc', 'Val Loss', 'Val Acc', 
                  'BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 'METEOR']
        data = []
        for i in range(len(self.epochs)):
            row = [self.epochs[i], 
                   self.train_loss[i] if i < len(self.train_loss) else "",
                   self.train_acc[i] if i < len(self.train_acc) else ""]
            row.append(self.val_loss[i] if i < len(self.val_loss) else "")
            row.append(self.val_acc[i] if i < len(self.val_acc) else "")
            row.append(self.bleu1_scores[i] if i < len(self.bleu1_scores) else "")
            row.append(self.bleu2_scores[i] if i < len(self.bleu2_scores) else "")
            row.append(self.bleu3_scores[i] if i < len(self.bleu3_scores) else "")
            row.append(self.bleu4_scores[i] if i < len(self.bleu4_scores) else "")
            row.append(self.meteor_scores[i] if i < len(self.meteor_scores) else "")
            data.append(row)
        with open(os.path.join(self.output_dir, 'training_metrics.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(data)

