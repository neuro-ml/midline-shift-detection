from functools import lru_cache
from argparse import ArgumentParser

from torch.optim import Adam
from dpipe.batch_iter import Infinite, unpack_args, sample
from dpipe.train import train, ConsoleLogger
from dpipe.torch import to_device, save_model_state

from midline_shift_detection import *

parser = ArgumentParser()
parser.add_argument('input', help='Path to the folder containing train data.')
parser.add_argument('output', help='Path to which the model will be saved.')
parser.add_argument('--device', default=None,
                    help='The device to use. E.g. cuda or cpu. CUDA is used by default if possible.')
parser.add_argument('--batch_size', type=int, default=40)
parser.add_argument('--cache', type=bool, default=True, help='Whether to cache the training data to RAM.')
args = parser.parse_args()

lr = 1e-3
n_epochs = 100
batch_size = args.batch_size
samples_per_train = 40 * 32000  # 32000 batches of size 40

paths = gather_train(args.input)
# cache the loader
if args.cache:
    load_pair = lru_cache(None)(load_pair)

# Check out https://deep-pipe.readthedocs.io/en/latest/tutorials/batch_iter.html
# for more details about the batch iterators we use

batch_iter = Infinite(
    # get a random pair of paths
    sample(paths),
    # load the image-contour pair
    unpack_args(load_pair),
    # get a random slice
    unpack_args(get_random_slice),
    # simple augmentation
    unpack_args(random_flip),

    batch_size=batch_size, batches_per_epoch=samples_per_train // (batch_size * n_epochs), combiner=combiner
)

model = to_device(Network(), args.device)
optimizer = Adam(model.parameters(), lr=lr)

# Here we use a general training function with a custom `train_step`.
# See the tutorial for more details: https://deep-pipe.readthedocs.io/en/latest/tutorials/training.html

train(
    train_step, batch_iter, n_epochs=n_epochs, logger=ConsoleLogger(),
    # additional arguments to `train_step`
    model=model, optimizer=optimizer
)
save_model_state(model, args.output)
