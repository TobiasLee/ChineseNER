from ner import Ner
import argparse
import os
import shutil

data_augument_tag_list = ['address',
                          'book',
                          'company',
                          'game',
                          'government',
                          'movie',
                          'name',
                          'organization',
                          'position',
                          'scene']

parser = argparse.ArgumentParser()
parser.add_argument('--file_name', type=str, default='train', help='file to be augmented')
parser.add_argument('--dedup', action='store_true', default=False, help='whether deduplicated the entities in each tag')
parser.add_argument('--augument_size', type=int, default=3,
                    help='size of synthesized data. size of total data = (augment_size + 1) * the original data size')
parser.add_argument('--seed', type=int, default=42, help='random seed')

if __name__ == '__main__':
    args = parser.parse_args()
    file_name = args.file_name
    dedup = args.dedup
    augument_size = args.augument_size
    seed = args.seed

    ner = Ner(ner_dir_name='.',
              ignore_tag_list=['O'],
              data_augument_tag_list=data_augument_tag_list,
              augument_size=augument_size, seed=seed, dedup=dedup)

    aug_samples, aug_sample_tags = ner.augment(file_name='%s.txt' % file_name)

    if dedup:
        target_dir = 'aug_dedup_%d' % augument_size
    else:
        target_dir = 'aug_%d' % augument_size
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    with open(os.path.join(target_dir, '%s.txt' % file_name), 'w', encoding='utf-8') as f:
        for tokens, tags in zip(aug_samples, aug_sample_tags):
            for token, tag in zip(tokens, tags):
                f.write("%s %s\n" % (token, tag))
            f.write("\n")

    for file in ['dev.txt', 'labels.txt']:
        shutil.copy(file, os.path.join(target_dir, file))
