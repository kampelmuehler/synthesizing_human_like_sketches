import torchvision
import numpy as np
from torchvision.transforms import ToPILImage, Compose
import torch.optim as optim
from utils.data import SketchyDB, UnnormImage
from utils.utils import *
import os
from os.path import isfile
from shutil import copyfile
from time import time
from models.PSim_net import PSimNet
from models.sketch_generator import SketchGenerator
import pickle
from torch.utils.tensorboard import SummaryWriter
import argparse
from models.resnet_classifier import ResnetClassifier

parser = argparse.ArgumentParser(description='Synthesizing human-like sketches',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--seed', default=1337, type=int, help='random seed')
parser.add_argument('--batch_size', default=32, type=int, help='minibatch size for training')
parser.add_argument('--batch_size_test', default=8, type=int, help='minibatch size for testing')
parser.add_argument('--epochs', default=300, type=int, help='number of training epochs')
parser.add_argument('--log_interval', default=10, type=int, help='how often to log (per epoch)')
parser.add_argument('--tensorboard_gridsize', default=8, type=int, help='size of image grid in tensorboard')
parser.add_argument('--test_interval', default=1, type=int, help='how often to test (in epochs)')
parser.add_argument('--save_interval', default=50, type=int, help='how often to keep a checkpoint (in epochs)')
parser.add_argument('--gpu', default=0, type=int, help='which gpu to use')
parser.add_argument('--run_name', default='test-run', type=str, help='name of the rollout')
parser.add_argument('--num_classes', default=125, type=int, help='number of classes in dataset')
parser.add_argument('--dataloader_num_workers', default=4, type=int, help='number of workers for dataloader')
parser.add_argument('--resume_training', default=False, type=bool, help='whether to resume training')
args = parser.parse_args()

IMAGE_TRANSFORM = Compose([UnnormImage(), ToPILImage()])
IMAGE_TRANSFORM_TB = Compose([UnnormImage()])  # for tensorboard

CLASSIFIER_PATH = 'pretrained_models/resnet_classifier.pt'


def test(model, test_loader, outpath, device, epoch, log=None, writer=None, comparator=None):
    log.info(f'Classifier weights: {CLASSIFIER_PATH}')
    classifier = ResnetClassifier()
    classifier_checkpoint = torch.load(CLASSIFIER_PATH, map_location=torch.device("cpu"))
    classifier.load_state_dict(classifier_checkpoint['state_dict'])
    classifier.eval()
    classifier.to(device)
    correct_test_top1 = 0
    correct_test_top5 = 0
    test_loss_acc = 0
    originals = []
    outputs = []
    import random
    sample_batches = random.sample(range(len(test_loader)), args.tensorboard_gridsize)
    for batch_idx, batch in enumerate(test_loader):
        model.eval()
        data, target, target_classifier = batch['image'].to(device), batch['sketch'].float().to(device), batch[
            'labelID'].to(device)
        labels_onehot = torch.zeros(data.size(0), args.num_classes).to(device)
        labels_onehot.scatter_(1, target_classifier.unsqueeze(1), 1.)
        output = model(data, labels_onehot)
        # logging for tensorboard
        if batch_idx in sample_batches:
            sample_idx = random.choice(range(data.size(0)))
            originals.append(IMAGE_TRANSFORM_TB(data[sample_idx, ...].cpu().detach()))
            outputs.append(output[sample_idx, 0, ...].cpu().detach().unsqueeze(0).repeat(3, 1, 1))
            test_loss_acc += comparator(output, target).item()
        # classifier
        output_classifier = classifier(output)
        correct_test_top1 += correct_topk(output_classifier, target_classifier, 1)
        correct_test_top5 += correct_topk(output_classifier, target_classifier, 5)

    # tensorboard logging
    imgrid = originals[:args.tensorboard_gridsize] + outputs[:args.tensorboard_gridsize]
    imgrid = torchvision.utils.make_grid(imgrid, nrow=args.tensorboard_gridsize)
    writer.add_image('test/samples', imgrid, epoch)

    test_loss = test_loss_acc / (len(test_loader))
    top1_accuracy = 100. * correct_test_top1 / len(test_loader.dataset)
    top5_accuracy = 100. * correct_test_top5 / len(test_loader.dataset)
    log.info(f'Test Loss: {test_loss:.4f}')
    log.info(f'Test set accuracy (Top-1): {correct_test_top1}/{len(test_loader.dataset)} ({top1_accuracy:.0f}%)')
    log.info(f'Test set accuracy (Top-5): {correct_test_top5}/{len(test_loader.dataset)} ({top5_accuracy:.0f}%)')
    return top1_accuracy, top5_accuracy, test_loss


def train(model, device, train_loader, optimizer, epoch, log_interval=(),
          comparator=None,
          log=None,
          writer=None):
    loss_acc = 0
    model.train()
    for batch_idx, batch in enumerate(train_loader):
        data, target, labels = batch['image'].to(device), batch['sketch'].float().to(device), batch['labelID'].to(
            device)
        labels_onehot = torch.zeros(data.size(0), args.num_classes).to(device)
        labels_onehot.scatter_(1, labels.unsqueeze(1), 1.)
        optimizer.zero_grad()

        output = model(data, labels_onehot)
        loss = comparator(output, target)
        loss_acc += loss.item()
        loss.backward()

        optimizer.step()
        if batch_idx in log_interval:
            log.info(f'Train Epoch: {epoch}/{args.epochs} '
                     f'[{batch_idx * len(data)}/{len(train_loader.dataset)} '
                     f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                     f'Loss: {loss.item():.4f}')
        if batch_idx == 0:
            # extract imgs from first batch
            originals = [IMAGE_TRANSFORM_TB(data[j, ...].cpu().detach()) for j in range(args.tensorboard_gridsize)]
            targets = [target[j, 0, ...].cpu().detach().unsqueeze(0).repeat(3, 1, 1)
                       for j in range(args.tensorboard_gridsize)]
            outputs = [output[j, 0, ...].cpu().detach().unsqueeze(0).repeat(3, 1, 1)
                       for j in range(args.tensorboard_gridsize)]
            imgrid = originals + targets + outputs
            imgrid = torchvision.utils.make_grid(imgrid, nrow=args.tensorboard_gridsize)
            writer.add_image('train/samples', imgrid, epoch)
    return loss_acc / len(train_loader)


def run_experiment(outpath=None, resume_training=False, log=None):
    torch.manual_seed(args.seed)
    writer = SummaryWriter(join(outpath, 'tensorboard'))
    # list available GPUs with IDs
    listGPUs()
    device = torch.device(f"cuda:{args.gpu}")

    dataset_test = SketchyDB('data',
                             image_type='bbox',
                             sketch_type='centered_scaled',
                             filter_erroneous=True,
                             filter_context=True,
                             filter_ambiguous=True,
                             filter_pose=True,
                             split='test')

    test_loader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size_test,
        shuffle=False,
        num_workers=args.dataloader_num_workers)
    dataset = SketchyDB('data',
                        image_type='bbox',
                        sketch_type='centered_scaled',
                        filter_erroneous=True,
                        filter_context=True,
                        filter_ambiguous=True,
                        filter_pose=True)

    # setup log_interval to generate args.log_interval plots per epoch
    log_interval = np.linspace(0, len(dataset) // args.batch_size, args.log_interval + 1, dtype=np.int).tolist()[:-1]

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers)
    model = SketchGenerator()
    model.to(device)

    comparator = PSimNet(device=device)

    optimizer = optim.Adam(model.decoder.parameters())

    if resume_training is True:
        checkpoint = join(outpath, f'models/{args.run_name}_last_checkpoint.pt')
        print(f'loading checkpoint {checkpoint}')
        checkpoint = torch.load(checkpoint, map_location=torch.device("cpu"))
        start_epoch = checkpoint['epoch']
        state_dict = checkpoint['state_dict']
        model.load_state_dict(state_dict)
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        start_epoch = 1

    # Get number of trainable parameters
    log.info(f'Number of trainable model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')

    epoch_times = []
    train_losses = []
    test_losses = []
    test_accs_top1 = []
    test_accs_top5 = []
    for epoch in range(start_epoch, args.epochs + 1):
        start_time = time()
        train_losses.append(train(model, device, train_loader, optimizer, epoch,
                                  log_interval=log_interval,
                                  comparator=comparator,
                                  log=log,
                                  writer=writer))
        writer.add_scalar('losses/train', train_losses[-1], epoch)

        # save checkpoint every epoch
        fname = join(outpath, f'models/{args.run_name}_last_checkpoint.pt')
        log.info(f'Saving model parameters to {fname}')
        torch.save({'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()},
                   fname)
        # keep models as specified by SAVE_INTERVAL
        if epoch % args.test_interval == 0:
            with torch.no_grad():
                (test_acc_top1, test_acc_top5, test_loss) = test(model, test_loader, outpath, device, epoch,
                                                                 log=log,
                                                                 writer=writer,
                                                                 comparator=comparator)
                test_accs_top1.append(test_acc_top1)
                test_accs_top5.append(test_acc_top5)
                test_losses.append(test_loss)
                writer.add_scalar('losses/test', test_losses[-1], epoch)
                writer.add_scalar('losses/test_top1', test_accs_top1[-1], epoch)
                writer.add_scalar('losses/test_top5', test_accs_top5[-1], epoch)
        if epoch % args.save_interval == 0:
            checkpoint_name = join(outpath, f'models/model_epoch{epoch}.pt')
            log.info(f'Saving model parameters to {checkpoint_name}')
            copyfile(fname, checkpoint_name)
        # timing
        epoch_times.append(time() - start_time)
        time_remaining = (args.epochs - (epoch + 1)) * np.mean(epoch_times)
        log.info(f'Time/epoch: {epoch_times[-1]:.1f} s; approximately {time_remaining / 3600:.1f} h remaining')

    # save loss/accuracy
    pickle.dump({'train': train_losses,
                 'test': test_losses,
                 'test_top1': test_accs_top1,
                 'test_top5': test_accs_top5,
                 'test_interval': args.test_interval,
                 'classifier': CLASSIFIER_PATH}, open(join(outpath, 'losses.pkl'), 'wb'))
    # remove last checkpoint when training is finished
    copyfile(fname, join(outpath, f'models/{args.run_name}.pt'))
    os.remove(fname)


def main():
    # define output directory
    outpath = f'out/{args.run_name}'
    print(f'starting run {args.run_name}')

    # make sure not to overwrite any past runs and continue training per default
    if isfile(join(outpath, f'models/{args.run_name}_last_checkpoint.pt')):
        log = get_logger(outpath)
        run_experiment(outpath=outpath, log=log, resume_training=True)
    elif isfile(join(outpath, f'models/{args.run_name}.pt')):
        print(f"Training finished")
    else:
        setup_output(outpath, overwrite_protection=True)
        log = get_logger(outpath)
        run_experiment(outpath=outpath, log=log)


if __name__ == '__main__':
    main()
