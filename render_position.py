#!/usr/bin/env python3
"""
Chess Position Renderer

Generates chess position images by combining board and piece PNGs based on a FEN string.

https://github.com/GiorgioMegrelli/chess.com-boards-and-pieces/blob/master/render_position.py
"""

import sys
from pathlib import Path
from typing import Optional

from PIL import Image
import numpy as np


PIECE_MAP = {
    'r': 'br.png',  # black rook
    'n': 'bn.png',  # black knight
    'b': 'bb.png',  # black bishop
    'q': 'bq.png',  # black queen
    'k': 'bk.png',  # black king
    'p': 'bp.png',  # black pawn
    'R': 'wr.png',  # white rook
    'N': 'wn.png',  # white knight
    'B': 'wb.png',  # white bishop
    'Q': 'wq.png',  # white queen
    'K': 'wk.png',  # white king
    'P': 'wp.png',  # white pawn
}

PIECE_NAMES = [
    "3d_chesskid",
    "3d_plastic",
    "3d_staunton",
    "3d_wood",
    "8_bit",
    "alpha",
    "bases",
    "blindfold",
    "book",
    "bubblegum",
    "cases",
    "classic",
    "club",
    "condal",
    "dash",
    "game_room",
    "glass",
    "gothic",
    "graffiti",
    "icy_sea",
    "light",
    "lolz",
    "marble",
    "maya",
    "metal",
    "modern",
    "nature",
    "neo",
    "neon",
    "neo_wood",
    "newspaper",
    "ocean",
    "sky",
    "space",
    "tigers",
    "tournament",
    "vintage",
    "wood"
]

BOARD_NAMES = [
    "8_bit",
    "bases",
    "blue",
    "brown",
    "bubblegum",
    "burled_wood",
    "dark_wood",
    "dash",
    "glass",
    "graffiti",
    "green",
    "icy_sea",
    "light",
    "lolz",
    "marble",
    "metal",
    "neon",
    "newspaper",
    "orange",
    "overlay",
    "parchment",
    "purple",
    "red",
    "sand",
    "sky",
    "stone",
    "tan",
    "tournament",
    "translucent",
    "walnut"
]


def parse_fen(fen: str) -> list[list[Optional[str]]]:
    """
    Parse FEN string and return 8x8 board representation.

    Returns a 2D list where board[rank][file] contains the FEN character
    for the piece at that position, or None for empty squares.
    Rank 0 = visual top (rank 8), Rank 7 = visual bottom (rank 1).
    """
    # Extract just the board layout part (before first space)
    board_layout = fen.split()[0]

    # Split into ranks (8 to 1, top to bottom)
    ranks = board_layout.split('/')

    if len(ranks) != 8:
        raise ValueError(f"Invalid FEN: expected 8 ranks, got {len(ranks)}")

    board: list[list[Optional[str]]] = []

    for rank in ranks:
        row: list[Optional[str]] = []
        for char in rank:
            if char.isdigit():
                # Empty squares
                row.extend([None] * int(char))
            elif char in PIECE_MAP:
                row.append(char)
            else:
                raise ValueError(f"Invalid FEN character: {char}")

        if len(row) != 8:
            raise ValueError(f"Invalid FEN: rank has {len(row)} squares instead of 8")

        board.append(row)

    return board


def render_position(
    fen: str,
    board_name: str,
    pieces_name: str,
    size: int,
    side: str = 'white'
) -> np.ndarray:
    """
    Render a chess position to an image file.

    Args:
        fen: FEN string representing the position
        board_name: Name of the board style (without .png)
        pieces_name: Name of the pieces directory
        size: Size of each square in pixels
        output: Output filename
        side: Perspective to render from ('white' or 'black', default 'white')
    """
    # Validate paths
    board_path = Path(f"boards/{board_name}.png")
    pieces_dir = Path(f"pieces/{pieces_name}")

    if not board_path.exists():
        raise FileNotFoundError(f"Board not found: {board_path}")

    if not pieces_dir.exists():
        raise FileNotFoundError(f"Pieces directory not found: {pieces_dir}")

    # Parse FEN
    board = parse_fen(fen)

    # Load and resize board
    board_img = Image.open(board_path)
    board_size = size * 8
    board_img = board_img.resize((board_size, board_size), Image.Resampling.LANCZOS)

    # Convert to RGBA to handle transparency
    if board_img.mode != 'RGBA':
        board_img = board_img.convert('RGBA')

    # Rotate board 180° if viewing from black's perspective
    if side == 'black':
        board_img = board_img.rotate(180)

    # Create a new image for compositing
    final_img = board_img.copy()

    # Place pieces on the board
    for rank_idx, rank in enumerate(board):
        for file_idx, piece_char in enumerate(rank):
            if piece_char is not None:
                # Get piece filename
                piece_filename = PIECE_MAP[piece_char]
                piece_path = pieces_dir / piece_filename

                if not piece_path.exists():
                    print(f"Warning: Piece file not found: {piece_path}", file=sys.stderr)
                    continue

                # Load and resize piece
                piece_img = Image.open(piece_path)
                piece_img = piece_img.resize((size, size), Image.Resampling.LANCZOS)

                # Convert to RGBA
                if piece_img.mode != 'RGBA':
                    piece_img = piece_img.convert('RGBA')

                # Calculate position on board
                if side == 'black':
                    # Flip coordinates for black's perspective
                    x = (7 - file_idx) * size
                    y = (7 - rank_idx) * size
                else:
                    x = file_idx * size
                    y = rank_idx * size

                # Paste piece onto board with transparency
                final_img.paste(piece_img, (x, y), piece_img)

    final_img = np.array(final_img)
    final_img = final_img[:, :, :3][:, :, ::-1]  # RGBA to BGR
    return final_img
