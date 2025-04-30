import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from models import Encoder, DecoderWithAttention
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
from plot_utils import TrainingMonitor
from metrics import calculate_metrics

data_folder = 'data'
data_name = 'coco_5_cap_per_img_5_min_word_freq'
use_subset = True
subset_percentage = 0.5

emb_dim = 512
attention_dim = 512
decoder_dim = 512
dropout = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

start_epoch = 0
epochs = 50
epochs_since_improvement = 0
batch_size = 32
workers = 1
encoder_lr = 1e-4
decoder_lr = 4e-4
grad_clip = 5.
alpha_c = 1.
best_bleu4 = 0.
print_freq = 100
fine_tune_encoder = False
checkpoint = None
plots_dir = "./training_plots"

def main():
    global best_bleu4, epochs_since_improvement, checkpoint, start_epoch, fine_tune_encoder, data_name, word_map

    monitor = TrainingMonitor(output_dir=plots_dir)

    word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

    if checkpoint is None:
        decoder = DecoderWithAttention(attention_dim=attention_dim,
                                       embed_dim=emb_dim,
                                       decoder_dim=decoder_dim,
                                       vocab_size=len(word_map),
                                       encoder_dim=2048,
                                       dropout=dropout)
        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                             lr=decoder_lr)
        encoder = Encoder()
        encoder.fine_tune(fine_tune_encoder)
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             lr=encoder_lr) if fine_tune_encoder else None
    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_bleu4 = checkpoint['bleu-4']
        decoder = checkpoint['decoder']
        decoder_optimizer = checkpoint['decoder_optimizer']
        encoder = checkpoint['encoder']
        encoder_optimizer = checkpoint['encoder_optimizer']
        if fine_tune_encoder is True and encoder_optimizer is None:
            encoder.fine_tune(fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                                 lr=encoder_lr)

    decoder = decoder.to(device)
    encoder = encoder.to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = CaptionDataset(
        data_folder,
        data_name,
        'TRAIN',
        transform=transforms.Compose([normalize]),
        use_subset=use_subset,
        subset_percentage=subset_percentage
    )

    val_dataset = CaptionDataset(
        data_folder,
        data_name,
        'VAL',
        transform=transforms.Compose([normalize]),
        use_subset=use_subset,
        subset_percentage=subset_percentage
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True
    )

    print(f"Training on {len(train_dataset)} captions.")
    print(f"Validating on {len(val_dataset)} captions.")

    for epoch in range(start_epoch, epochs):
        if epochs_since_improvement == 20:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(decoder_optimizer, 0.8)
            if fine_tune_encoder:
                adjust_learning_rate(encoder_optimizer, 0.8)

        train_loss, train_acc = train(train_loader=train_loader,
                                     encoder=encoder,
                                     decoder=decoder,
                                     criterion=criterion,
                                     encoder_optimizer=encoder_optimizer,
                                     decoder_optimizer=decoder_optimizer,
                                     epoch=epoch)

        val_loss, val_acc, bleu4, meteor = validate(val_loader=val_loader,
                                                   encoder=encoder,
                                                   decoder=decoder,
                                                   criterion=criterion)

        monitor.update(
            epoch=epoch,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
            bleu4=bleu4,
            meteor=meteor
        )
        monitor.plot_all()

        is_best = bleu4 > best_bleu4
        best_bleu4 = max(bleu4, best_bleu4)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer,
                        decoder_optimizer, bleu4, is_best)

def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch):
    decoder.train()
    encoder.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    for i, (imgs, caps, caplens) in enumerate(train_loader):
        data_time.update(time.time() - start)

        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        imgs = encoder(imgs)
        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

        targets = caps_sorted[:, 1:]

        scores_packed = pack_padded_sequence(scores, decode_lengths, batch_first=True)
        scores = scores_packed[0]
        targets_packed = pack_padded_sequence(targets, decode_lengths, batch_first=True)
        targets = targets_packed[0]

        loss = criterion(scores, targets)
        loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()

        if grad_clip is not None:
            clip_gradient(decoder_optimizer, grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, grad_clip)

        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        top5 = accuracy(scores, targets, 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses,
                                                                          top5=top5accs))
    
    return losses.avg, top5accs.avg

def validate(val_loader, encoder, decoder, criterion):
    decoder.eval()
    if encoder is not None:
        encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    references = list()
    hypotheses = list()

    with torch.no_grad():
        for i, (imgs, caps, caplens, allcaps) in enumerate(val_loader):
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)

            if encoder is not None:
                imgs = encoder(imgs)
            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

            targets = caps_sorted[:, 1:]

            scores_copy = scores.clone()
            scores_packed = pack_padded_sequence(scores, decode_lengths, batch_first=True)
            scores = scores_packed[0]
            targets_packed = pack_padded_sequence(targets, decode_lengths, batch_first=True)
            targets = targets_packed[0]

            loss = criterion(scores, targets)
            loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

            losses.update(loss.item(), sum(decode_lengths))
            top5 = accuracy(scores, targets, 5)
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            start = time.time()

            if i % print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader), batch_time=batch_time,
                                                                                loss=losses, top5=top5accs))

            sort_ind = sort_ind.cpu()
            allcaps = allcaps[sort_ind]
            
            for j in range(allcaps.shape[0]):
                img_caps = allcaps[j].tolist()
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                        img_caps))
                references.append(img_captions)

            if device != torch.device("cpu"):
                scores_copy = scores_copy.cpu()
            
            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths[j]])
            preds = temp_preds
            hypotheses.extend(preds)

            assert len(references) == len(hypotheses)

        metrics = calculate_metrics(references, hypotheses, word_map)
        bleu1 = metrics['bleu1']
        bleu2 = metrics['bleu2']
        bleu3 = metrics['bleu3']
        bleu4 = metrics['bleu4']
        meteor = metrics['meteor']

        print(
            '\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}\n'
            ' * BLEU-1 - {bleu1:.3f}, BLEU-2 - {bleu2:.3f}, BLEU-3 - {bleu3:.3f}, BLEU-4 - {bleu4:.3f}\n'
            ' * METEOR - {meteor:.3f}\n'.format(
                loss=losses,
                top5=top5accs,
                bleu1=bleu1,
                bleu2=bleu2,
                bleu3=bleu3,
                bleu4=bleu4,
                meteor=meteor))

    return losses.avg, top5accs.avg, bleu4, meteor

if __name__ == '__main__':
    main()

