# Chess AIVA

## Install environment

We'll start from a working python environment with PyTorch installed. For that, follow [this webpage](https://pytorch.org/get-started/locally/). I used Python 3.10, and my command looked like this:

Now run the following command (or use `conda`, `uv`... whatever you want):

```sh
pip install \
    tqdm \
    torchinfo \
    opencv-python \
    numpy \
    pillow \
    cairosvg \
    python-chess \
    mss \
    pandas \
    httpx \
    matplotlib
```

## `*.py` Files

`common.py` contains functions or classes used by multiple files.

1. `gen_new_backgrounds.py` generates the backgrounds to be used in the dataset.
1. `fetcher.py` to download images of pieces and boards. Credits: [link](https://github.com/GiorgioMegrelli/chess.com-boards-and-pieces)
1. `gen_ds.py` generates the synthetic dataset. It uses [this script](https://github.com/GiorgioMegrelli/chess.com-boards-and-pieces/blob/master/render_position.py) with some modifications, located at `render_position.py`.
1. `check.py` helps check a sample of what `gen_ds.py` generates.
1. `make_gray_images.py` converts the dataset to grayscale.
1. `train_cnn.py` trains the model responsible for detecting the board's location.
1. `gen_pieces_ds.py` takes the generated synthetic dataset generated with `gen_ds.py` and generates a dataset of chess pieces by dividing the images in 64 squares.
1. `train_board_model.py` trains the model responsible for detecting the square piece (or empty).
1. `plot_curves.py` plots training curves for all the models in the specified directory.
1. `detector.py` and `detector_screen.py` do the magic. The first one takes a video as input, while the second one grabs a section of your own screen.

## Other `*.py` Files

Attempts were made to train two separate models, one for piece detection and another for classification, in `train_binary_model.py` and `train_piece_model.py`. They were replaced by `train_board_model.py` due to detecting a better performance after training both things at once, rather than separately. Hence, they are forgotten and left as "proof".
