from RanDepict import RandomDepictor
from PIL import Image
from itertools import cycle
from typing import List


def create_depiction_grid(
        images: List,
        im_per_row: int,
        im_per_col: int
        ) -> Image:
    """
    Generates a grid of depictions (see figures in our publication)
    ___
    Important: We assume here that the images all have the same shape.

    Args:
        images (List): List of structure depictions (np.array)
        im_per_row (int): number of images per row in grid
        im_per_col (int): number of images per column in grid

    Returns:
        PIL.Image: Grid of images
    """
    y, x, _ = images[0].shape
    fig = Image.new('RGB', (x * im_per_row, y * im_per_col))
    y_iter = cycle([y * i for i in range(im_per_col)])
    for im_index in range(len(images)):
        x_pos = int(im_index/im_per_col) * x
        y_pos = next(y_iter)
        pos = (x_pos, y_pos)
        fig.paste(Image.fromarray(images[im_index]), pos)
    return fig


def main() -> None:
    """
    This script generates two grids of chemical structure depictions
    (augmented and not augmented) using RanDepict. The images that are
    generated here are guaranteed to be diverse due to the usage of the
    depiction feature fingerprints.
    """
    smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"

    with RandomDepictor(42) as depictor:
        # Depict given SMILES 20 times without additional augmentation
        images = depictor.batch_depict_with_fingerprints([smiles],
                                                         20,
                                                         aug_proportion=0)
        aug_images = depictor.batch_depict_with_fingerprints([smiles],
                                                             20,
                                                             aug_proportion=1)
    # Generate and save grid of structure depictions
    figure_2 = create_depiction_grid(images, 5, 4)
    figure_2.save('figure_2.png')
    figure_3 = create_depiction_grid(aug_images, 5, 4)
    figure_3.save('figure_3.png')


if __name__ == '__main__':
    main()
