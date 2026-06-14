"""
Flask backend for the Go (Baduk) web application.

Endpoints:
    GET  /                -> serve index.html
    POST /api/new_game    -> reset board
    POST /api/move        -> player move + AI response
    POST /api/pass        -> player pass + AI response
    POST /api/analyse     -> positional analysis (top candidates)
    GET  /api/visualize   -> SSE stream of live MCTS stats
"""

from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS
from go_engine import GoBoard, MCTSEngine
import json
import threading
import time
import queue

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = Flask(__name__)
CORS(app)

# Global game state
game_board = GoBoard()
mcts_engine = MCTSEngine(simulations=1000)
board_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def board_response(extra: dict = None) -> dict:
    """Standard JSON payload describing the current board state."""
    data = {
        "board_state": game_board.to_list(),
        "current_player": game_board.current_player,
        "move_count": game_board.move_count,
        "captures": {
            "black": game_board.captures[GoBoard.BLACK],
            "white": game_board.captures[GoBoard.WHITE],
        },
        "game_over": game_board.is_game_over(),
    }
    if game_board.is_game_over():
        score = game_board.calculate_score()
        data["winner"] = score["winner"]
        data["score"] = score
    if extra:
        data.update(extra)
    return data


def ai_respond(simulations: int) -> dict:
    """
    Let the AI (MCTS engine) pick a move and apply it to the global board.
    Returns a dict with the engine's move info.
    """
    color = game_board.current_player

    if game_board.is_game_over():
        return {"engine_move": None}

    # Temporarily adjust simulation count if requested
    old_sims = mcts_engine.simulations
    mcts_engine.simulations = simulations

    move = mcts_engine.get_best_move(game_board, color)

    mcts_engine.simulations = old_sims

    if move is None:
        game_board.pass_move(color)
        return {"engine_move": "pass"}
    else:
        success = game_board.place_stone(move[0], move[1], color)
        if not success:
            # Fallback: if the "best" move is somehow illegal, pass
            game_board.pass_move(color)
            return {"engine_move": "pass"}
        return {"engine_move": list(move)}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Serve the frontend."""
    return render_template("index.html")


@app.route("/api/new_game", methods=["POST"])
def new_game():
    """Reset the board and start a fresh game."""
    global game_board
    with board_lock:
        data = request.get_json(silent=True) or {}
        size = data.get("size", 9)
        if size not in (9, 13, 19):
            size = 9
        game_board = GoBoard(size=size)
    return jsonify(board_response())


@app.route("/api/move", methods=["POST"])
def make_move():
    """
    Accept a player move, validate it, apply it, then run the AI.

    Request JSON: {row: int, col: int, simulations: int (optional)}
    """
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Invalid JSON body"}), 400

    row = data.get("row")
    col = data.get("col")
    simulations = data.get("simulations", mcts_engine.simulations)

    if row is None or col is None:
        return jsonify({"error": "Missing row or col"}), 400

    try:
        row = int(row)
        col = int(col)
        simulations = int(simulations)
    except (TypeError, ValueError):
        return jsonify({"error": "row, col, and simulations must be integers"}), 400

    with board_lock:
        if game_board.is_game_over():
            return jsonify({"error": "Game is already over"}), 400

        color = game_board.current_player
        if not game_board.is_legal(row, col, color):
            return jsonify({"error": f"Illegal move at ({row}, {col})"}), 400

        # Apply player move
        game_board.place_stone(row, col, color)
        player_move = [row, col]

        # AI response
        engine_info = ai_respond(simulations)

    resp = board_response()
    resp["player_move"] = player_move
    resp.update(engine_info)
    return jsonify(resp)


@app.route("/api/pass", methods=["POST"])
def pass_move():
    """Register a player pass, then let the AI respond."""
    data = request.get_json(silent=True) or {}
    simulations = int(data.get("simulations", mcts_engine.simulations))

    with board_lock:
        if game_board.is_game_over():
            return jsonify({"error": "Game is already over"}), 400

        color = game_board.current_player
        game_board.pass_move(color)

        # AI response
        engine_info = ai_respond(simulations)

    resp = board_response()
    resp["player_move"] = "pass"
    resp.update(engine_info)
    return jsonify(resp)


@app.route("/api/analyse", methods=["POST"])
def analyse():
    """
    Run MCTS analysis on the current (or a provided) board state.

    Request JSON:
        board_state: optional 2D list to analyse instead of the live game
        simulations: optional int (default 1000)
    """
    data = request.get_json(silent=True) or {}
    simulations = int(data.get("simulations", mcts_engine.simulations))
    top_n = int(data.get("top_n", 5))

    board_state = data.get("board_state")

    if board_state is not None:
        # Reconstruct a board from the supplied 2D grid
        try:
            size = len(board_state)
            board = GoBoard(size=size)
            for r in range(size):
                for c in range(size):
                    board.grid[r][c] = int(board_state[r][c])
            # Infer current player from stone counts
            blacks = sum(cell == GoBoard.BLACK for row in board.grid for cell in row)
            whites = sum(cell == GoBoard.WHITE for row in board.grid for cell in row)
            board.current_player = GoBoard.BLACK if blacks <= whites else GoBoard.WHITE
        except Exception as exc:
            return jsonify({"error": f"Invalid board_state: {exc}"}), 400
    else:
        board = game_board.copy()

    color = board.current_player

    old_sims = mcts_engine.simulations
    mcts_engine.simulations = simulations
    candidates = mcts_engine.analyse_position(board, color, top_n=top_n)
    mcts_engine.simulations = old_sims

    return jsonify({"candidates": candidates, "current_player": color})


@app.route("/api/visualize")
def visualize():
    """
    Server-Sent Events endpoint.

    Runs MCTS on the current position and streams stats every ~50 iterations.

    Query params:
        simulations: int (default 1000)
    """
    simulations = int(request.args.get("simulations", 1000))

    def generate():
        q: queue.Queue = queue.Queue()

        board = game_board.copy()
        color = board.current_player

        def callback(stats: dict):
            q.put(stats)

        # Run MCTS in a background thread so we can stream results
        def run():
            mcts_engine.get_live_stats(board, color, simulations, callback)
            q.put(None)  # sentinel

        t = threading.Thread(target=run, daemon=True)
        t.start()

        while True:
            try:
                stats = q.get(timeout=30)
            except queue.Empty:
                break
            if stats is None:
                # Final event
                yield "event: done\ndata: {}\n\n"
                break
            yield f"data: {json.dumps(stats)}\n\n"

    return Response(generate(), mimetype="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "X-Accel-Buffering": "no",
                    })


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

@app.errorhandler(404)
def not_found(_e):
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(500)
def server_error(_e):
    return jsonify({"error": "Internal server error"}), 500


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)
