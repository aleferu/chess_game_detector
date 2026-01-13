#!/usr/bin/env python3


import argparse
import logging
import cv2
import chess
import chess.pgn
import mss
import numpy as np

from common import (
    setup_logging,
    TinyCNNAdapter,
    MicroBoardCNNAdapter,
    INV_PIECE_MAP,
    StableGameTracker,
    BoardOrientation,
    extract_board,
    split_board_into_squares,
    normalize_obs_board,
    obs_to_fen_placement
)


def select_screen_roi() -> dict[str, int]:
    """
    Takes a full screenshot, lets the user draw a box with cv2.selectROI,
    and returns the mss monitor dictionary for that region.
    """
    with mss.mss() as sct:
        # Get the primary monitor
        # sct.monitors[0] is all monitors combined
        # sct.monitors[1] is the primary
        monitor_info = sct.monitors[0]

        logging.info("Grabbing screenshot for ROI selection...")
        sct_img = sct.grab(monitor_info)
        img = np.array(sct_img)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)  # Convert for OpenCV

        # Instructions
        cv2.namedWindow("Select ROI", cv2.WINDOW_NORMAL)
        # Try to fit screen if resolution is huge
        h, w = img.shape[:2]
        if w > 1920 or h > 1080:
             cv2.resizeWindow("Select ROI", 1280, 720)

        logging.info("--- INSTRUCTIONS ---")
        logging.info("1. A screenshot window will appear.")
        logging.info("2. Click and drag to draw a box around the CHESS BOARD.")
        logging.info("3. Press ENTER or SPACE to confirm.")
        logging.info("4. Press C to cancel (will exit).")
        logging.info("--------------------")

        # Returns (x, y, w, h)
        rect = cv2.selectROI("Select ROI", img, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow("Select ROI")

        x, y, w, h = rect

        # Handle cancellation
        if w == 0 or h == 0:
            logging.error("No ROI selected. Exiting.")
            exit()

        # Construct the MSS monitor dict
        # We must add the monitor's offset (top/left) to the relative ROI coordinates
        roi_monitor = {
            "top": monitor_info["top"] + y,
            "left": monitor_info["left"] + x,
            "width": w,
            "height": h
        }

        logging.info(f"Selected ROI: {roi_monitor}")
        return roi_monitor


def main(args: argparse.Namespace):
    setup_logging()
    board_model = TinyCNNAdapter(args.board_model)
    piece_model = MicroBoardCNNAdapter(args.piece_model)

    sct = mss.mss()  # MSS for screen capture
    tracker = StableGameTracker(args.initial_fen_suffix)

    orientation_map = {"w": BoardOrientation.WHITE_BOTTOM, "b": BoardOrientation.BLACK_BOTTOM}
    orientation = orientation_map.get(args.orientation.lower(), BoardOrientation.WHITE_BOTTOM)
    logging.debug(f"Using Orientation: {orientation.name}")

    logging.info("Starting Live Tracking... Press 'r' to Reset, ESC to Quit.")
    last_move = None

    monitor_roi = select_screen_roi()
    while True:
        # Start timer for FPS
        timer = cv2.getTickCount()

        # Capture Screen
        sct_img = sct.grab(monitor_roi)
        frame = np.array(sct_img)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        raw_placement_fen = None
        obs_board = None

        # 1. Detection
        has_board, bbox = board_model(frame)

        if has_board:
            # 2. Extract & Classify
            board_img = extract_board(frame, bbox)
            squares = split_board_into_squares(board_img)

            # Squares to list
            square_coords = [(r, c) for r in range(8) for c in range(8)]
            square_imgs = [squares[coord] for coord in square_coords]

            # Batch inference
            batch_results = piece_model.predict_batch(square_imgs)
            obs_board = {coord: res for coord, res in zip(square_coords, batch_results)}

            # Update Tracker
            norm_obs = normalize_obs_board(obs_board, orientation)
            raw_placement_fen = obs_to_fen_placement(norm_obs)

            result = tracker.process_frame_fen(raw_placement_fen)
            if result:
                last_move = result
                logging.info(f"Tracker Update: {result}")

                # if "MOVE" in result:
                #     print(f"--> {result}")

        # Calculate FPS
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        # 4. Visualization
        if args.visualize:
            h, w = frame.shape[:2]
            if has_board:
                x1, y1, x2, y2 = bbox
                x1, x2, y1, y2 = int(x1*w), int(x2*w), int(y1*h), int(y2*h)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                if obs_board:
                    bw, bh = (x2 - x1), (y2 - y1)
                    for r in range(8):
                        for c in range(8):
                            # 4.1. Calculate the bounding coordinates of the square
                            sq_x1 = x1 + int(c * (bw / 8))
                            sq_y1 = y1 + int(r * (bh / 8))
                            sq_x2 = x1 + int((c + 1) * (bw / 8))
                            sq_y2 = y1 + int((r + 1) * (bh / 8))

                            # 4.2. Draw the red grid square (thickness 1 for subtlety)
                            cv2.rectangle(frame, (sq_x1, sq_y1), (sq_x2, sq_y2), (0, 0, 255), 1)

                            # 4.3. Get detection and place text at the top-left of the square
                            obs = obs_board.get((r, c))
                            if obs and obs.piece_id != 0:
                                p_name = INV_PIECE_MAP[obs.piece_id]

                                # Offset text slightly (5px) from the top-left corner
                                cv2.putText(frame, p_name, (sq_x1 + 5, sq_y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

            # HUD
            y_off = 30
            # FPS Counter display
            cv2.putText(frame, f"FPS: {int(fps)}", (w - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Init State
            init_color = (0, 255, 0) if tracker.initialized else (0, 0, 255)
            cv2.putText(frame, "TRACKER INIT", (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.7, init_color, 2)
            y_off += 30

            # Stability
            if raw_placement_fen:
                cnt = tracker.candidate_count
                txt = f"Stable: {cnt}/{tracker.stability_threshold}"
                cv2.putText(frame, txt, (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # Last Move
            if tracker.game_board.move_stack and last_move:
                cv2.putText(frame, f"Last: {last_move}", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.imshow("Chess AI Live", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC to quit
                break
            elif key == ord("r"):  # 'r' to reset
                logging.info("Reset key pressed.")
                tracker.reset()
                last_move = None

    # Cleanup and Save PGN
    if args.pgn_out:
        game = chess.pgn.Game.from_board(tracker.game_board)
        with open(args.pgn_out, "w") as f:
            f.write(str(game))
        logging.info(f"PGN saved to {args.pgn_out}")

    # FEN saving logic: Iterate through move history
    if args.fen_out:
        with open(args.fen_out, "w") as f:
            # Start from the initial position of the current game
            temp_board = tracker.game_board.root()

            # Write the starting position first
            f.write(temp_board.fen() + "\n")

            # Replay every move recorded in the stack
            for move in tracker.game_board.move_stack:
                temp_board.push(move)
                f.write(temp_board.fen() + "\n")
        logging.info(f"FENs saved to {args.fen_out}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--board-model", default="out/base16/best_model.pth")
    parser.add_argument("--piece-model", default="out/board_model_modifiers/best_board_model.pth")
    parser.add_argument("--visualize", action="store_true", default=False)
    parser.add_argument("--initial-fen-suffix", default="w KQkq - 0 1", help="FEN suffix (turn, castling, etc)")
    parser.add_argument("--pgn-out", default="game.pgn", help="Output path for PGN file")
    parser.add_argument("--fen-out", default="fens.txt", help="Output path for FEN list file")
    parser.add_argument("--orientation", default="w", help="w for white and b for black")

    args = parser.parse_args()
    main(args)
