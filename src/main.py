import torch
from torch.utils.data import DataLoader

import clip
import numpy as np
import random
import argparse
import os
from tqdm import tqdm
import yaml
import csv

from datasets import get_dataset_class, TaggedMultipleDataset
from datasets.corruption import MultipleImageFeatDataset, CachedFeatLabelDataset
from methods import get_method_class


def standard_test(datasets, tta_model, args):
    """
    Run the standard test
    Adaptation on each domain is independent, since the tta model is reset
    :param datasets:
    :param tta_model:
    :param args:
    :return:
    """
    dataset_accs = []

    for i, dataset in tqdm(enumerate(datasets)):

        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                pin_memory=True)
        num_correct, num_total = 0, 0
        for image, label in tqdm(dataloader):
            image, label = image.to(args.device), label.to(args.device)
            # print(image.shape, label.shape)
            logits = tta_model(image)

            with torch.no_grad():
                pred = torch.argmax(logits, dim=1)
                num_correct += pred.eq(label).sum().item()
                num_total += label.size(0)

        dataset_accs.append(num_correct / num_total)
        tqdm.write(f"{datasets.environments[i]}: {dataset_accs[-1]}")

        tta_model.reset()

    return dataset_accs


def cache_test(datasets, tta_model, args):
    """
    Run the standard test
    Adaptation on each domain is independent, since the tta model is reset
    :param datasets:
    :param tta_model:
    :param args:
    :return:
    """

    cache_path = os.path.join(args.data_root, 'corruption_cache', args.dataset, f"{args.model.replace('/', '-')}.pkl")
    feat_datasets = CachedFeatLabelDataset(cache_path)
    datasets = MultipleImageFeatDataset(image_datasets=datasets, feat_datasets=feat_datasets)

    dataset_accs = []

    for i, dataset in tqdm(enumerate(datasets)):
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                pin_memory=True)
        num_correct, num_total = 0, 0
        for image, cached_feat, label in tqdm(dataloader):
            image, cached_feat, label = image.to(args.device), cached_feat.to(args.device), label.to(args.device)
            # print(image.shape, label.shape)
            logits = tta_model(image, cached_feat)

            with torch.no_grad():
                pred = torch.argmax(logits, dim=1)
                num_correct += pred.eq(label).sum().item()
                num_total += label.size(0)

        dataset_accs.append(num_correct / num_total)
        tqdm.write(f"{datasets.environments[i]}: {dataset_accs[-1]}")

        tta_model.reset()

    return dataset_accs


def mixture_test(datasets, tta_model, args):
    """
    Run the standard test
    Adaptation on each domain is independent, since the tta model is reset
    :param datasets:
    :param tta_model:
    :param args:
    :return:
    """
    mix_dataset = TaggedMultipleDataset(datasets)

    dataset_num_corrects = torch.zeros(len(datasets), dtype=torch.int)
    dataset_num_samples = np.array([len(dataset) for dataset in datasets])

    dataloader = DataLoader(mix_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                            pin_memory=True)

    for image, label, domain_idx, sample_idx in tqdm(dataloader):
        image, label = image.to(args.device), label.to(args.device)
        # print(image.shape, label.shape)
        logits = tta_model(image)

        with torch.no_grad():
            pred = torch.argmax(logits, dim=1)
            is_correct = pred.eq(label).cpu().int()
            dataset_num_corrects.index_add_(0, domain_idx, is_correct)

    dataset_num_corrects = dataset_num_corrects.numpy()
    dataset_accs = dataset_num_corrects / dataset_num_samples

    tta_model.reset()

    return dataset_accs


def mixture_cache_test(datasets, tta_model, args):
    cache_path = os.path.join(args.data_root, 'corruption_cache', args.dataset, f"{args.model.replace('/', '-')}.pkl")
    feat_datasets = CachedFeatLabelDataset(cache_path)
    datasets = MultipleImageFeatDataset(image_datasets=datasets, feat_datasets=feat_datasets)

    mix_dataset = TaggedMultipleDataset(datasets)

    dataset_num_corrects = torch.zeros(len(datasets), dtype=torch.int)
    dataset_num_samples = np.array([len(dataset) for dataset in datasets])

    dataloader = DataLoader(mix_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                            pin_memory=True)

    for image, cached_feat, label, domain_idx, sample_idx in tqdm(dataloader):
        image, cached_feat, label = image.to(args.device), cached_feat.to(args.device), label.to(args.device)
        # print(image.shape, label.shape)
        logits = tta_model(image, cached_feat)

        with torch.no_grad():
            pred = torch.argmax(logits, dim=1)
            is_correct = pred.eq(label).cpu().int()
            dataset_num_corrects.index_add_(0, domain_idx, is_correct)

    dataset_num_corrects = dataset_num_corrects.numpy()
    dataset_accs = dataset_num_corrects / dataset_num_samples

    tta_model.reset()

    return dataset_accs


def main(args):
    print("Loading model...")
    clip_model, preprocess = clip.load(args.model, device=args.device)

    if args.float:
        clip_model = clip_model.to(torch.float32)

    print("Loading datasets...")
    datasets = get_dataset_class(args.dataset)(root=args.data_root, transform=preprocess)
    print(datasets.environments)

    print("Initializing model")
    tta_model = get_method_class(args.algo)(clip_model, datasets.classes, args.config)

    print("Start testing")

    if args.tta_mode == 'standard':
        dataset_accs = standard_test(datasets, tta_model, args)
    elif args.tta_mode == 'mixture':
        dataset_accs = mixture_test(datasets, tta_model, args)
    elif args.tta_mode == 'cache':
        dataset_accs = cache_test(datasets, tta_model, args)
    elif args.tta_mode == 'mixture_cache':
        dataset_accs = mixture_cache_test(datasets, tta_model, args)

    elif args.tta_mode == 'sfda':  # source free domain adaptation
        # leave the dataloader to algo...
        dataset_accs = tta_model.evaluate(datasets, args)

    print(
        f"--dataset {args.dataset} "
        f"--algo {args.algo} "
        f"--batch_size {args.batch_size} "
        f"--num_workers {args.num_workers} "
        f"--num_threads {args.num_threads} "
        f"--seed {args.seed} "
        f"--tta_mode {args.tta_mode} "
        f"--model {args.model} "
        f"{'--cuda' if args.cuda else ''}"
    )

    for env, acc in zip(datasets.environments, dataset_accs):
        print(f"{env}: {acc:.4f}")

    print(f"total: {np.mean(dataset_accs)}")

    # For formated priting
    for acc in dataset_accs:
        print(f"{acc * 100:.2f}", end=",")

    print(f"{np.mean(dataset_accs) * 100:.2f}")

    # new added to store data into shell/results.csv
    csv_path = args.save_to
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    with open(csv_path, "a", newline='') as f:
        writer = csv.writer(f)

        args_dict = {k: str(v) for k, v in vars(args).items() if isinstance(v, (int, str, float, bool))}
        writer.writerow([f"{k}={v}" for k, v in args_dict.items()])

        writer.writerow([f"{k}={v}" for k, v in args.config.items()])

        writer.writerow(datasets.environments + ["Average"])

        avg_acc = np.mean(dataset_accs)
        accs_formatted = [f"{acc * 100:.2f}" for acc in dataset_accs]
        writer.writerow(accs_formatted + [f"{avg_acc * 100:.2f}"])

        writer.writerow([])


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='CIFAR10C')

    parser.add_argument('--data_root', type=str, default='~/data')

    parser.add_argument('--tta_mode', type=str, default='standard')  # mixture

    parser.add_argument('--model', type=str, default='ViT-B/16',
                        help='model name, a version of CLIP listed by clip.available_models()')

    parser.add_argument('--algo', type=str, default='CLIP',
                        help='tta algorithm name')

    parser.add_argument('--batch_size', type=int, default=1,
                        help='number of images in each mini-batch')

    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers for dataloader')

    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--float', action='store_true', default=False,
                        help='whether to use float32')

    parser.add_argument('--cuda', action='store_true', default=False,
                        help='whether use cuda to train')

    parser.add_argument('--num_threads', type=int, default=4,
                        help='Number of threads')

    parser.add_argument('--config', type=str, default='../cfg')

    parser.add_argument('--save_to', type=str, default='../log/default.csv')

    args = parser.parse_args()

    args.data_root = os.path.expanduser(args.data_root)

    args.device = torch.device('cuda') if torch.cuda.is_available() and args.cuda else torch.device('cpu')

    if args.float:
        args.dtype = torch.float
    else:
        args.dtype = torch.half if torch.cuda.is_available() and args.cuda else torch.float

    filepaths = [
        os.path.expanduser(os.path.join(args.config, args.dataset, args.algo + '.yaml')),
        os.path.expanduser(os.path.join(args.config, 'debug', args.algo + '.yaml')),
    ]

    found_yaml = False

    for filepath in filepaths:
        if os.path.exists(filepath):
            print('Loading config from {}'.format(filepath))
            with open(filepath, 'r') as f:
                args.config = yaml.safe_load(f)

            print(args.config)

            found_yaml = True
            break

    if not found_yaml:
        args.config = {}

    return args


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    args = args_parser()
    setup_seed(args.seed)
    torch.set_num_threads(args.num_threads)
    main(args)
