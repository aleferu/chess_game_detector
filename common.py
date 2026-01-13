import logging
from typing import Optional
import torch
import torch.nn as nn
import cv2
import numpy as np
import chess
from dataclasses import dataclass
from enum import Enum


def setup_logging():
    """Sets up logging module"""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )


class TinyCNN(nn.Module):
    """A very small CNN for fast training/inference.

    Output: 5 values per image: [board_logit, bbox_raw(4)]
    bbox_raw will be passed through sigmoid during loss/inference to map to (0,1) using get_scaled_sigmoid.
    """

    base_channels: int
    input_size: int
    features: nn.Module
    head: nn.Module

    def __init__(self, in_channels: int = 1, base: int = 32, input_size: int = 256):
        super().__init__()

        self.base_channels = base
        self.input_size = input_size

        c = base
        mid_size = c * 2
        final_size = c * 4

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, c, 5, stride=2, padding=2), nn.BatchNorm2d(c), nn.SiLU(inplace=True),
            nn.Conv2d(c, c, 5, padding=1), nn.BatchNorm2d(c), nn.BatchNorm2d(c), nn.SiLU(inplace=True),

            nn.Conv2d(c, mid_size, 5, stride=2, padding=2), nn.BatchNorm2d(mid_size), nn.SiLU(inplace=True),
            nn.Conv2d(mid_size, mid_size, 5, padding=1), nn.BatchNorm2d(mid_size), nn.SiLU(inplace=True),

            nn.Conv2d(mid_size, final_size, 5, stride=2, padding=2), nn.BatchNorm2d(final_size), nn.SiLU(inplace=True),
            nn.Conv2d(final_size, final_size, 5, padding=1), nn.BatchNorm2d(final_size), nn.SiLU(inplace=True),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, input_size, input_size)
            out = self.features(dummy)
            flat_dim = out.shape[1] * out.shape[2] * out.shape[3]
        hidden_dim = 1024
        self.head = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 5)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(1)
        x = self.head(x)
        return x  # [B,5]


# Helpers for the different classes in a chess board
PIECE_MAP = {
    "": 0, "wp": 1, "wn": 2, "wb": 3, "wr": 4, "wq": 5, "wk": 6,
    "bp": 7, "bn": 8, "bb": 9, "br": 10, "bq": 11, "bk": 12
}
INV_PIECE_MAP = {v: k for k, v in PIECE_MAP.items()}


def get_scaled_sigmoid(t: torch.Tensor, scale: float = 1.2, shift: float = -0.1, clamp: bool = False) -> torch.Tensor:
    """Helper to get the ouput of TinyCNN"""
    result = torch.sigmoid(t) * scale + shift
    if clamp:
        result = result.clamp(0, 1)
    return result


# def build_model(base_channels: int = 32, in_channels: int = 1) -> TinyCNN:
#     return TinyCNN(base=base_channels, in_channels=in_channels)


class ToTensorNormalize:
    """Class in charge of preparing the images for tinyCNNAdapter"""
    def __init__(self):
        pass

    def __call__(self, img: np.ndarray) -> torch.Tensor:
        # img: HxW (grayscale) or HxWxC (BGR/RGB). Convert to grayscale tensor [1,H,W]
        arr = img
        if arr.ndim == 2:
            t = torch.from_numpy(arr)
        else:
            # If color, convert to grayscale using ITU-R BT.601 luma
            # Assume BGR if coming from cv2
            if arr.shape[2] == 3:
                # Convert BGR to gray
                arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
            else:
                arr = arr[:, :, 0]
            t = torch.from_numpy(arr)
        t = t.unsqueeze(0).float() / 255.0
        return t


class TinyCNNAdapter:
    """Adapter for board location detection"""

    device: torch.device
    to_tensor: ToTensorNormalize
    img_size: int
    model: nn.Module

    def __init__(self, model_path: str):
        self.device = torch.device("cuda")
        self.to_tensor = ToTensorNormalize()
        state_dict = torch.load(model_path, map_location=self.device)

        args = state_dict["args"]
        base_channels = args["base_channels"]
        img_size = args["img_size"]
        self.img_size = img_size

        self.model = TinyCNN(in_channels=1, base=base_channels, input_size=self.img_size)
        self.model.load_state_dict(state_dict["model"])
        self.model.to(self.device).eval()

    def _prepare_image(self, img: np.ndarray) -> torch.Tensor:
        if img.ndim == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        return self.to_tensor(img).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def eval_from_img(self, img: np.ndarray) -> tuple[bool, tuple[float, float, float, float]]:
        """Returns a prediction for an image, in the form of [bool, bbox"""
        x = self._prepare_image(img)
        out = self.model(x)
        prob = torch.sigmoid(out[0, 0]).item()
        raw_bbox = get_scaled_sigmoid(out[0, 1:5], clamp=True)
        bbox = raw_bbox.cpu().numpy().tolist()
        return prob > 0.5, tuple(bbox)

    def eval_from_path(self, image_path: str) -> tuple[bool, tuple[float, float, float, float]]:
        """Returns a prediction for an image, in the form of [bool, bbox"""
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")
        return self.eval_from_img(img)

    def __call__(self, img: np.ndarray) -> tuple[bool, tuple[float, float, float, float]]:
        """Returns a prediction for an image, in the form of [bool, bbox"""
        """Calls self.eval_from_img"""
        return self.eval_from_img(img)


@dataclass
class SquareObservation:
    """Dataclass to handle square prediction"""
    piece_id: int
    confidence: float
    probs: np.ndarray


class MicroBoardCNNAdapter:
    """Adapter for per-square piece classification."""

    device: torch.device
    model: "MicroBoardCNN"
    to_tensor: ToTensorNormalize

    def __init__(self, model_path: str):
        self.device = torch.device("cuda")
        ckpt = torch.load(model_path, map_location=self.device)
        self.model = MicroBoardCNN()
        self.model.load_state_dict(ckpt)
        self.model.to(self.device).eval()
        self.to_tensor = ToTensorNormalize()

    @torch.no_grad()
    def __call__(self, square_img: np.ndarray) -> SquareObservation:
        """Single image inference (fallback)."""
        return self.predict_batch([square_img])[0]

    @torch.no_grad()
    def predict_batch(self, square_imgs: list[np.ndarray]) -> list[SquareObservation]:
        """
        Processes a list of images in a single forward pass.
        Returns a list of SquareObservation objects.
        """
        if not square_imgs:
            return list()

        # 1. Pre-process and stack all images into a list, while resizing to (64, 64)
        processed_imgs = [cv2.resize(img, (64, 64)) for img in square_imgs]

        # 2. Convert list of images to a single batch tensor
        # self.to_tensor returns a (C, H, W) tensor; stack makes it (B, C, H, W)
        x = torch.stack([self.to_tensor(img) for img in processed_imgs]).to(self.device)

        # 3. Single forward pass for all 64 squares
        logits = self.model(x)  # Shape: (64, num_classes)
        probs_batch = torch.softmax(logits, dim=1).cpu().numpy()

        # 4. Wrap results back into SquareObservation objects
        observations = list()
        for probs in probs_batch:
            pid = int(np.argmax(probs))
            observations.append(SquareObservation(pid, probs[pid], probs))

        return observations


class MicroBoardCNN(nn.Module):
    """
    Lightweight CNN for Board State Classification (13 classes).
    0: Empty
    1-12: Piece Types
    Input: [B, 1, 64, 64] -> Output: [B, 13]
    """

    features: nn.Module
    classifier: nn.Module

    def __init__(self, input_channels: int = 1, num_classes: int = 13):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # -> 32x32

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # -> 16x16

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # -> 8x8

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)) # -> [B, 256, 1, 1]
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


class StableGameTracker:
    """Handles the game recognition logic"""

    game_board: chess.Board
    initialized: bool
    fen_suffix: str
    stability_threshold: int
    threshold_score: int
    candidate_fen: Optional[str]
    candidate_count: int
    last_stable_fen: Optional[str]
    prev_stable_fen: Optional[str]

    def __init__(self, initial_fen_suffix: str, stability_threshold: int = 5, threshold_score: int = 64):
        self.game_board = chess.Board()                 # Internal logic state
        self.initialized = False                        # Have we locked onto the starting position of the video?
        self.fen_suffix = initial_fen_suffix            # e.g. "w KQkq - 0 1"
        self.stability_threshold = stability_threshold  # Frames to confirm a board state
        self.threshold_score = threshold_score          # If lower than 64, we allow some error
        # Stability Buffers
        self.candidate_fen = None
        self.candidate_count = 0
        # Track the last and second-to-last *accepted* visual FEN to compare against
        self.last_stable_fen = None
        self.prev_stable_fen = None

    def reset(self):
        """Wipes the tracker state to allow re-initialization."""
        self.game_board = chess.Board()
        self.initialized = False
        self.candidate_fen = None
        self.candidate_count = 0
        self.last_stable_fen = None
        self.prev_stable_fen = None
        logging.info("Tracker state has been reset.")

    def get_fen(self) -> str:
        """Returns the FEN of the current game_board"""
        return self.game_board.fen().split(" ")[0]

    def update_fen(self):
        """Updates the last and second-to-last seen FENs"""
        self.last_stable_fen = self.get_fen()
        try:
            last_move = self.game_board.pop()
            self.prev_stable_fen = self.get_fen()
            self.game_board.push(last_move)
        except IndexError:
            pass

    def resolve_move_depth_2(self, target_fen_placement: str) -> Optional[str]:
        """
        Finds a sequence of 2 moves that results in a board state MOST SIMILAR
        to the target_fen_placement.
        """
        try:
            target_board = chess.Board(f"{target_fen_placement} w - - 0 1")
        except ValueError:  # Invalid FEN
            return None

        best_moves = None
        best_score = -1

        # Move 1
        for move1 in self.game_board.legal_moves:
            self.game_board.push(move1)

            # Move 2
            for move2 in self.game_board.legal_moves:
                self.game_board.push(move2)

                # This logic is identical to calculate_similarity method but adapted to this situation (we already have a board)
                # I should refactor this and calculate_similarity to match
                match_score = 0
                for sq in chess.SQUARES:
                    p1 = self.game_board.piece_at(sq)
                    p2 = target_board.piece_at(sq)
                    if p1 == p2:
                        match_score += 1

                if match_score > best_score:
                    best_score = match_score
                    best_moves = (move1, move2)

                self.game_board.pop() # Undo Move 2

                if best_score == 64:
                    break

            self.game_board.pop() # Undo Move 1
            if best_score == 64:
                break

        # Apply and return the best sequence if it meets the threshold
        if best_moves and best_score >= self.threshold_score:
            m1, m2 = best_moves

            # Generate SAN for the moves
            san1 = self.game_board.san(m1)
            self.game_board.push(m1)
            san2 = self.game_board.san(m2)
            self.game_board.push(m2)

            logging.info(f"2-Step Recovery: {san1}, {san2} with score {best_score}/64")
            return f"{san1}, {san2}"

        return None

    def process_frame_fen(self, raw_placement_fen: str) -> Optional[str]:
        """
        Takes the current detected FEN and yields a solution.
        Returns:
          - None: No update
          - "INIT": Initialized successfully
          - "MOVE: <san>": A move was detected
        """
        # Stability Check: Does this frame match the candidate?
        if raw_placement_fen == self.candidate_fen:
            self.candidate_count += 1
        else:
            self.candidate_fen = raw_placement_fen
            self.candidate_count = 1

        # Trigger Logic
        # We only act if we hit the threshold exactly (to avoid spamming and improve performance)
        # OR if we are not initialized, we keep checking to sync up.
        if self.candidate_count >= self.stability_threshold:
            # CASE A: Not Initialized yet -> Force sync
            if not self.initialized:
                # We assume the first stable thing we see is the current game state.
                full_fen = f"{raw_placement_fen} {self.fen_suffix}"
                try:
                    self.game_board = chess.Board(full_fen)
                    self.update_fen()
                    self.initialized = True
                    return "INIT"
                except ValueError:
                    # Invalid FEN (e.g. 10 kings), keep waiting
                    return None

            # CASE B: Already Initialized -> Look for moves
            # Only try to resolve if we just hit the threshold and it's a new attempt
            if self.candidate_count == self.stability_threshold and raw_placement_fen != self.last_stable_fen:
                # 1. Try finding 1 standard move
                move_san = self.resolve_move(raw_placement_fen)
                if move_san:
                    self.update_fen()
                    return f"MOVE: {move_san}"

                # 2. Check for Undo (Direct match to previous FEN)
                elif raw_placement_fen == self.prev_stable_fen:
                    self.game_board.pop()
                    self.update_fen()
                    logging.info("UNDOING LAST MOVE, GOING BACK")
                    return "UNDO"

                # 3. Fallback: Try Replacement -> Then Try 2-Move -> Then Fail
                else:
                    # STEP 3A: Try Replacement (Assume last move was wrong)
                    if self.game_board.move_stack:
                        last_move = self.game_board.pop()
                        # Try to solve from the PREVIOUS state
                        move_san = self.resolve_move(raw_placement_fen)

                        if move_san:
                            logging.info("UNDOING LAST MOVE, REPLACING")
                            self.update_fen()
                            return f"MOVE: {move_san}"

                        # If replacement failed, RESTORE state before trying next trick
                        self.game_board.push(last_move)

                    # STEP 3B: Try 2-Move Recovery (Assume we missed a position or player pre-moved)
                    moves_2_san = self.resolve_move_depth_2(raw_placement_fen)
                    if moves_2_san:
                        self.update_fen()
                        logging.info("TWO MOVES DETECTED!")
                        return f"MOVES: {moves_2_san}"

                    # STEP 3C: Illegal State or we skipeed too many positions
                    logging.warning("Stable change detected, but illegal move.")
                    logging.warning(f"  Internal previous: {self.prev_stable_fen}")
                    logging.warning(f"   Internal current: {self.last_stable_fen}")
                    logging.warning(f"Current observation: {raw_placement_fen}")

        return None

    def calculate_similarity(self, board_fen_b_str: str) -> int:
        """
        Calculates how many squares match between a chess.Board object
        and a raw placement FEN string.
        Returns integer score (0 to 64).
        """
        try:
            board_b = chess.Board(f"{board_fen_b_str} w - - 0 1")
        except ValueError:
            return 0
        match_score = 0
        for sq in chess.SQUARES:
            p1 = self.game_board.piece_at(sq)
            p2 = board_b.piece_at(sq)
            if p1 == p2:
                match_score += 1
        return match_score

    def resolve_move(self, target_fen_placement: str) -> Optional[str]:
        """
        Finds the legal move that results in a board state MOST SIMILAR
        to the target_fen_placement.
        """
        best_move = None
        best_score = -1
        for move in self.game_board.legal_moves:
            self.game_board.push(move)
            score = self.calculate_similarity(target_fen_placement)
            self.game_board.pop()
            if score > best_score:
                best_score = score
                best_move = move
        if best_move and best_score >= self.threshold_score:
            san = self.game_board.san(best_move)
            self.game_board.push(best_move)
            if best_score < 64:
                logging.info(f"Fuzzy Match: Accepted {san} with score {best_score}/64")
            return san
        return None


class BoardOrientation(Enum):
    WHITE_BOTTOM = 0
    BLACK_BOTTOM = 1


def extract_board(frame: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray:
    """Cuts the board from the given image and bbox"""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    x1, x2 = int(x1 * w), int(x2 * w)
    y1, y2 = int(y1 * h), int(y2 * h)
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    return frame[y1:y2, x1:x2]


def split_board_into_squares(board_img: np.ndarray) -> dict[tuple[int, int], np.ndarray]:
    """Gets the 64 images of the detected board squares"""
    h, w = board_img.shape[:2]
    sh, sw = h // 8, w // 8
    squares = dict()
    for r in range(8):
        for c in range(8):
            squares[(r, c)] = board_img[r * sh : (r + 1) * sh, c * sw:(c + 1) * sw]
    return squares


def orient_rc(r: int, c: int, orientation: BoardOrientation):
    """Handles row and column orientation"""
    return (r, c) if orientation == BoardOrientation.WHITE_BOTTOM else (7 - r, 7 - c)


def normalize_obs_board(
    obs_board: dict[tuple[int, int], SquareObservation],
    orientation: BoardOrientation
) -> dict[tuple[int, int], SquareObservation]:
    """Normalizes the detected obs_board given an orientation"""
    return {
        orient_rc(r, c, orientation): obs
        for (r, c), obs in obs_board.items()
    }


def obs_to_fen_placement(obs_board: dict[tuple[int, int], SquareObservation]) -> str:
    """Returns a FEN string given an observation board"""
    rows = list()
    for r in range(8):
        empty = 0
        row = ""
        for c in range(8):
            obs = obs_board.get((r,c))
            pid = obs.piece_id if obs else 0
            if pid == 0:
                empty += 1
            else:
                if empty:
                    row += str(empty)
                    empty = 0
                p_char = INV_PIECE_MAP[pid][1]
                color = INV_PIECE_MAP[pid][0]
                row += p_char.upper() if color == "w" else p_char.lower()
        if empty:
            row += str(empty)
        rows.append(row)
    return "/".join(rows)
