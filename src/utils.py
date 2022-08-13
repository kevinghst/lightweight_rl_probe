import torch
from kornia.augmentation import RandomAffine,\
    RandomCrop,\
    RandomResizedCrop, \
    RandomHorizontalFlip
from kornia.filters import GaussianBlur2d
from torch import nn
import numpy as np
import glob
import gzip
import shutil
from pathlib import Path
import os
import src.transforms as T
EPS = 1e-6

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def select_at_indexes(indexes, tensor):
    """Returns the contents of ``tensor`` at the multi-dimensional integer
    array ``indexes``. Leading dimensions of ``tensor`` must match the
    dimensions of ``indexes``.
    """
    dim = len(indexes.shape)
    assert indexes.shape == tensor.shape[:dim]
    num = indexes.numel()
    t_flat = tensor.view((num,) + tensor.shape[dim:])
    s_flat = t_flat[torch.arange(num, device=tensor.device), indexes.view(-1)]
    return s_flat.view(tensor.shape[:dim] + tensor.shape[dim + 1:])


def get_augmentation(augmentation, imagesize):
    if isinstance(augmentation, str):
        augmentation = augmentation.split("_")
    transforms = []
    for aug in augmentation:
        if aug == "affine":
            transformation = RandomAffine(5, (.14, .14), (.9, 1.1), (-5, 5))
        elif aug == "rrc":
            transformation = RandomResizedCrop((imagesize, imagesize), (0.8, 1))
        elif aug == 'rrc_n':
            transformation = T.Compose(
                [
                    T.RandomResize([100], square=True),
                    T.RandomSizeCrop(min_size=74, max_size=95, square=True),
                    T.RandomResize([84], square=True),
                ]
            )
        elif aug == "blur":
            transformation = GaussianBlur2d((5, 5), (1.5, 1.5))
        elif aug == "shift" or aug == "crop":
            transformation = nn.Sequential(nn.ReplicationPad2d(4), RandomCrop((84, 84)))
        elif aug == "intensity":
            transformation = Intensity(scale=0.05)
        elif aug == 'flip':
            transformation = RandomHorizontalFlip()
        elif aug == "none":
            continue
        else:
            raise NotImplementedError()
        transforms.append(transformation)

    return transforms


class Intensity(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        r = torch.randn((x.size(0), 1, 1, 1), device=x.device)
        noise = 1.0 + (self.scale * r.clamp(-2.0, 2.0))
        return x * noise


def maybe_transform(image, transform, p=0.8):
    processed_images = transform(image)
    if p >= 1:
        return processed_images
    else:
        mask = torch.rand((processed_images.shape[0], 1, 1, 1),
                          device=processed_images.device)
        mask = (mask < p).float()
        processed_images = mask * processed_images + (1 - mask) * image
        return processed_images


def renormalize(tensor, first_dim=-3):
    if first_dim < 0:
        first_dim = len(tensor.shape) + first_dim
    flat_tensor = tensor.view(*tensor.shape[:first_dim], -1)
    max = torch.max(flat_tensor, first_dim, keepdim=True).values
    min = torch.min(flat_tensor, first_dim, keepdim=True).values
    flat_tensor = (flat_tensor - min)/(max - min)

    return flat_tensor.view(*tensor.shape)


def from_categorical(distribution, limit=300, logits=True):
    distribution = distribution.float()  # Avoid any fp16 shenanigans
    if logits:
        distribution = torch.softmax(distribution, -1)
    num_atoms = distribution.shape[-1]
    weights = torch.linspace(-limit, limit, num_atoms, device=distribution.device).float()
    return distribution @ weights


def extract_epoch(filename):
    """
    Get the epoch from a model save string formatted as name_Epoch:{seed}.pt
    :param str: Model save name
    :return: epoch (int)
    """

    if "epoch" not in filename.lower():
        return 0

    epoch = int(filename.lower().split("epoch_")[-1].replace(".pt", ""))
    return epoch


def get_last_save(base_pattern, load_best, retry=True):
    base_pattern += '/epoch'
    files = glob.glob(base_pattern+"*.pt")

    if load_best:
        files = [x for x in files if 'best' in x]
        filename = files[0].split('/')[-1]
        epochs = [int(filename.split('_')[1])]
        inds = [0]
    else:
        files = [x for x in files if 'best' not in x]
        epochs = [extract_epoch(path) for path in files]
        inds = np.argsort(-np.array(epochs))

    for ind in inds:
        try:
            print("Attempting to load {}".format(files[ind]))
            if torch.cuda.is_available():
                state_dict = torch.load(Path(files[ind]))
            else:
                state_dict = torch.load(Path(files[ind]), map_location=torch.device('cpu'))
            epoch = epochs[ind]
            return state_dict, epoch
        except Exception as e:
            if retry:
                print("Loading failed: {}".format(e))
            else:
                raise e


def delete_all_but_last(base_pattern, num_to_keep=2):
    files = glob.glob(base_pattern+"*.pt")
    files = [x for x in files if 'best' not in x]
    epochs = [extract_epoch(path) for path in files]

    order = np.argsort(np.array(epochs))

    for i in order[:-num_to_keep]:
        os.remove(files[i])
        print("Deleted old save {}".format(files[i]))

def delete_prior_best(base_pattern):
    files = glob.glob(base_pattern+"*.pt")
    files = [x for x in files if 'best' in x]
    for file in files:
        os.remove(file)

def save_model_fn(model_save, save_only_last=False):
    def save_model(model, optim, epoch, best=False, f1=0):
        if best:
            delete_prior_best(f'{model_save}/epoch')
            path = Path(f'{model_save}/epoch_{epoch}_best_{f1}.pt')
        else:
            path = Path(f'{model_save}/epoch_{epoch}.pt')

        torch.save({"model": model, "optim": optim}, path)
        print("Saved model at {}".format(path))

        if save_only_last and not best:
            delete_all_but_last(f'{model_save}/epoch')

    return save_model


def find_weight_norm(parameters, norm_type=1.0) -> torch.Tensor:
    r"""Finds the norm of an iterable of parameters.

    The norm is computed over all parameterse together, as if they were
    concatenated into a single vector.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor to find norms of
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].device
    if norm_type == np.inf:
        total_norm = max(p.data.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.data.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


class DataWriter:
    def __init__(self,
                 save_data=True,
                 data_dir="/project/rrg-bengioy-ad/schwarzm/atari",
                 save_name="",
                 checkpoint_size=1000000,
                 game="Pong",
                 imagesize=(84, 84),
                 mmap=True):

        self.save_name = save_name
        self.save_data = save_data
        if not self.save_data:
            return

        self.pointer = 0
        self.checkpoint = 0
        self.checkpoint_size = checkpoint_size
        self.imagesize = imagesize
        self.dir = Path(data_dir) / game.replace("_", " ").title().replace(" ", "")
        os.makedirs(self.dir, exist_ok=True)
        self.mmap = mmap
        self.reset()

    def reset(self):
        self.pointer = 0
        obs_data = np.zeros((self.checkpoint_size, *self.imagesize), dtype=np.uint8)
        action_data = np.zeros((self.checkpoint_size,), dtype=np.int32)
        reward_data = np.zeros((self.checkpoint_size,), dtype=np.float32)
        terminal_data = np.zeros((self.checkpoint_size,), dtype=np.uint8)

        self.arrays = []
        self.filenames = []

        for data, filetype in [(obs_data, 'observation'),
                               (action_data, 'action'),
                               (reward_data, 'reward'),
                               (terminal_data, 'terminal')]:
            filename = Path(self.dir / f'{filetype}_{self.checkpoint}{self.save_name}.npy')
            if self.mmap:
                np.save(filename, data)
                data_ = np.memmap(filename, mode="w+", dtype=data.dtype, shape=data.shape,)
                del data
            else:
                data_ = data
            self.arrays.append(data_)
            self.filenames.append(filename)

    def save(self):
        for data, filename in zip(self.arrays, self.filenames):
            if not self.mmap:
                np.save(filename, data)
            del data  # Flushes memmap
            with open(filename, 'rb') as f_in:
                new_filename = os.path.join(self.dir, Path(os.path.basename(filename)[:-4]+".gz"))
                with gzip.open(new_filename, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

            os.remove(filename)

    def write(self, samples):
        if not self.save_data:
            return

        self.arrays[0][self.pointer] = samples.env.observation[0, 0, -1, 0]
        self.arrays[1][self.pointer] = samples.agent.action
        self.arrays[2][self.pointer] = samples.env.reward
        self.arrays[3][self.pointer] = samples.env.done

        self.pointer += 1
        if self.pointer == self.checkpoint_size:
            self.checkpoint += 1
            self.save()
            self.reset()


def update_state_dict_compat(osd, nsd):
    updated_osd = {k.replace("head.advantage", "head.goal_advantage").
                   replace("head.value", "head.goal_value").
                   replace("head.secondary_advantage_head", "head.rl_advantage").
                   replace("head.secondary_value_head", "head.rl_value")
                   : v for k, v in osd.items()}
    filtered_osd = {k: v for k, v in updated_osd.items() if k in nsd}
    missing_items = [k for k, v in updated_osd.items() if k not in nsd]
    if len(missing_items) > 0:
        print("Could not load into new model: {}".format(missing_items))
    nsd.update(filtered_osd)
    return nsd



def discount_return_n_step(reward, done, n_step, discount, return_dest=None,
                           done_n_dest=None, do_truncated=False):
    """Time-major inputs, optional other dimension: [T], [T,B], etc.  Computes
    n-step discounted returns within the timeframe of the of given rewards. If
    `do_truncated==False`, then only compute at time-steps with full n-step
    future rewards are provided (i.e. not at last n-steps--output shape will
    change!).  Returns n-step returns as well as n-step done signals, which is
    True if `done=True` at any future time before the n-step target bootstrap
    would apply (bootstrap in the algo, not here)."""
    rlen = reward.shape[0]
    if not do_truncated:
        rlen -= (n_step - 1)
    return_ = torch.zeros(
        (rlen,) + reward.shape[1:], dtype=reward.dtype, device=reward.device)
    done_n = torch.zeros(
        (rlen,) + reward.shape[1:], dtype=done.dtype, device=done.device)
    return_[:] = reward[:rlen].float()  # 1-step return is current reward.
    done_n[:] = done[:rlen].float()  # True at time t if done any time by t + n - 1

    done_dtype = done.dtype
    done_n = done_n.type(reward.dtype)
    done = done.type(reward.dtype)

    if n_step > 1:
        if do_truncated:
            for n in range(1, n_step):
                return_[:-n] += (discount ** n) * reward[n:n + rlen] * (1 - done_n[:-n])
                done_n[:-n] = torch.max(done_n[:-n], done[n:n + rlen])
        else:
            for n in range(1, n_step):
                return_ += (discount ** n) * reward[n:n + rlen] * (1 - done_n)
                done_n = torch.max(done_n, done[n:n + rlen])  # Supports tensors.
    done_n = done_n.type(done_dtype)
    return return_, done_n

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()