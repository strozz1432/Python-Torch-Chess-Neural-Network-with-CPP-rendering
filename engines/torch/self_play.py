import os
from typing import Callable, Optional

import chess
import chess.pgn
import chess.engine

from .auxiliary_func import board_to_matrix


def play_and_train_vs_stockfish(
    stockfish_path: str,
    num_games: int = 1,
    alternate_colors: bool = True,
    max_moves: int = 200,
    engine_depth: int = 12,
    train_callback: Optional[Callable] = None,
    predict_move_fn: Optional[Callable] = None,
    save_pgn_path: str = None,
    save_json_path: str = None,
    verbose: bool = True,
):


    if save_pgn_path is None:
        # Use package root, not cwd
        pkg_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        save_pgn_path = os.path.join(pkg_root, "data", "self_play_games.pgn")
    
    # Ensure directory exists
    save_dir = os.path.dirname(save_pgn_path)
    os.makedirs(save_dir, exist_ok=True)
    if save_json_path is None:
        save_json_path = os.path.join(save_dir, "self_play_moves.jsonl")
    # ensure json dir
    os.makedirs(os.path.dirname(save_json_path), exist_ok=True)

    total_merits = 0
    games_summary = []

    if not os.path.exists(stockfish_path):
        raise FileNotFoundError(f"stockfish binary not found at {stockfish_path}")

    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)

    for g in range(num_games):
        board = chess.Board()
        game = chess.pgn.Game()
        node = game

        agent_is_white = True if (not alternate_colors or (g % 2 == 0)) else False

        move_count = 0
        merits = 0
        history = []

        while not board.is_game_over() and move_count < max_moves:
            mover_is_white = board.turn

            # Eval before
            info_before = engine.analyse(board, chess.engine.Limit(depth=engine_depth))
            score_before = _score_from_info(info_before)

            # Pre-move board
            board_matrix_before = board_to_matrix(board)

            best_result = engine.play(board, chess.engine.Limit(depth=engine_depth))
            best_move = best_result.move

            # Choose move
            chosen_move = None
            used_callback_move = False
            if (mover_is_white and agent_is_white) or (not mover_is_white and not agent_is_white):
                if predict_move_fn is not None:
                    try:

                        pred = predict_move_fn(board)
                        if pred is None:
                            chosen_move = None
                        elif isinstance(pred, str):
                            try:
                                chosen_move = board.parse_uci(pred)
                            except Exception:
                                chosen_move = None
                        else:
                            # assume
                            chosen_move = pred
                        if chosen_move is not None:
                            used_callback_move = True
                    except Exception:
                        chosen_move = None

            if chosen_move is None:
                chosen_move = best_move

            try:
                san = board.san(chosen_move)
            except Exception:
                san = chosen_move.uci()

            board.push(chosen_move)
            node = node.add_variation(chosen_move)
            move_count += 1


            info_after = engine.analyse(board, chess.engine.Limit(depth=engine_depth))
            score_after = _score_from_info(info_after)


            temp_board = board.copy()
            try:
                temp_board.pop()
            except Exception:
                pass
            temp_board.push(best_move)
            info_best_after = engine.analyse(temp_board, chess.engine.Limit(depth=engine_depth))
            score_best_after = _score_from_info(info_best_after)


            improvement = (score_after - score_before) if mover_is_white else -(score_after - score_before)

            vs_best = (score_after - score_best_after) if mover_is_white else -(score_after - score_best_after)

            is_best = (chosen_move == best_move)
            quality, reason = _classify_move_vs_best(vs_best, is_best)
            delta = int(_merits_for_quality(quality, vs_best))
            merits += delta

            history.append({
                "move": chosen_move.uci(),
                "san": san,
                "qual": quality,
                "improvement": int(improvement),
                "vs_best_cp": int(vs_best),
                "is_best": bool(is_best),
            })

            if verbose:
                if (mover_is_white and agent_is_white) or (not mover_is_white and not agent_is_white):
                    mover_label = "Agent (model)" if used_callback_move else "Agent (fallback)"
                else:
                    mover_label = "Stockfish"

                print(f"[{g+1}] ply {move_count}: {mover_label.lower()} played {chosen_move.uci()} ({san})")

                print(f"    score_before={score_before}, score_after={score_after}, score_best_after={score_best_after}")
                print(f"    chosen_is_best={is_best}, quality={quality} ({reason}), imp={int(improvement)}, vs_best={int(vs_best)}, delta={delta}, merits={merits}")

                try:
                    print(board)
                except Exception:
                    pass


            if train_callback is not None:
                try:

                    train_callback(history, board_matrix_before, chosen_move.uci(), quality, delta)
                except Exception:

                    if verbose:
                        print("error: train_callback")


        result = board.result()
        if result == "1-0":
            if agent_is_white:
                merits += 100
            else:
                merits -= 100
        elif result == "0-1":
            if not agent_is_white:
                merits += 100
            else:
                merits -= 100

        total_merits += merits
        games_summary.append({"game_index": g, "result": result, "merits": merits, "moves": history})

        # Save PGN
        with open(save_pgn_path, "a") as f:
            exporter = chess.pgn.FileExporter(f)
            game.headers["Result"] = result
            game.accept(exporter)

        try:
            import json
            with open(save_json_path, "a") as jf:
                json.dump({"game_index": g, "result": result, "merits": merits, "moves": history}, jf)
                jf.write("\n")
        except Exception:
            pass

        if verbose:

            print(f"game {g+1}/{num_games} finished: result={result}, merits={merits}")

    engine.quit()

    return {"total_merits": total_merits, "games": games_summary}


def _score_from_info(info):
    # Score cp
    score = info.get("score")
    if score is None:
        return 0
    # Score type
    if score.is_mate():
        # Mate cp
        mate = score.white().mate()
        if mate is None:
            return 0
        # Mate mag
        return 100000 if mate > 0 else -100000
    cp = score.white().score()
    return cp if cp is not None else 0


def _classify_move_vs_best(vs_best_cp: float, is_best: bool):
    """
    vs_best_cp: chosen_move_cp - .
    """
    if is_best:
        return "best", "matches engine best"
    # thresholds 
    if vs_best_cp >= -20:
        return "good", "within 20cp of best"
    if vs_best_cp >= -100:
        return "inaccurate", "within 100cp of best"
    if vs_best_cp >= -300:
        return "mistake", "~100-300cp worse than best"
    return "blunder", ">=300cp worse than best"


def _merits_for_quality(quality: str, vs_best_cp: float = 0.0):
    """Return merits"""
    base = {
        "best": 3,
        "good": 2,
        "inaccurate": 0,
        "mistake": -3,
        "blunder": -10,
    }.get(quality, 0)
    # small bonus/penalty (default no extra)
    extra = 0
    # If the chosen move is worse than the engine best, apply a penalty
    if vs_best_cp < 0:
        # penalty proportional to severity (larger magnitude -> larger negative extra)
        extra = int(vs_best_cp / 200.0)
    # If the chosen move is better than the engine best (positive cp), give a small bonus
    elif vs_best_cp > 0:
        extra = int(vs_best_cp / 400.0)

    return base + extra
