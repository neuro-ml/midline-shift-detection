from pathlib import Path
from argparse import ArgumentParser

from tqdm import tqdm
import nibabel
import torch
from dpipe.medim.io import save_json
from dpipe.predict import preprocess, patches_grid, add_extract_dims
from dpipe.torch import load_model_state, to_var, to_np, to_device

from midline_shift_detection import *

parser = ArgumentParser()
parser.add_argument('input', help='Path to a nifty file or a folder of files.')
parser.add_argument('output', help='Path to the resulting contours, or a folder, if `input` is a folder.')
parser.add_argument('model', help='Path to the trained model.')
parser.add_argument('--device', default=None,
                    help='The device to use. E.g. cuda or cpu. CUDA is used by default if possible.')
args = parser.parse_args()


@rescale_nii
@crop_background
@preprocess(normalize_image)
@patches_grid(1, 1)
@add_extract_dims(2, 1)
def predict(x):
    model.eval()
    x = to_var(x, args.device)
    prediction = model(x[..., 0])
    prediction[:, 1] = torch.sigmoid(prediction[:, 1])
    return to_np(prediction)[..., None]


input_path = Path(args.input)
output_path = Path(args.output)

if input_path.is_dir():
    output_path.mkdir(parents=True, exist_ok=True)
    assert output_path.is_dir()

    files = [[output_path / f'{i}.json', image] for i, image in gather_nifty(input_path)]
else:
    files = [[output_path, input_path]]

bar = tqdm(files)
model = to_device(load_model_state(Network(), args.model), args.device)

for output_file, input_image in bar:
    bar.set_description(input_image.name)

    contours = predict(nibabel.load(str(input_image)))

    # filter out `undefined` points
    # contours = [
    #     [[y, x] for y, x in enumerate(contour) if not np.isnan(x)] for contour in contours
    # ]

    save_json(contours, output_file)
