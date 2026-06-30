"""
Go (Baduk) Engine – Python port of the C++ Ray engine.

Architecture mirrors Ray's GoBoard.hpp, MCTSNode.hpp, UCBEvaluation.cpp:
  - Color enum: EMPTY=0, BLACK=1, WHITE=2  (stone enum from Color.hpp)
  - GetOppositeColor: color XOR 0x3
  - Board as 2D grid (replaces char array + OB_SIZE padding)
  - UCB1-Tuned selection formula from UCBEvaluation.cpp
  - MCTS: Selection -> Expansion -> Simulation -> Backpropagation

Performance-critical paths are optimized for Python:
  - Fast has_liberty() with early-exit (no full flood-fill)
  - Simulation uses random position shuffle instead of full legal-move generation
  - is_suicide() checks immediate neighbors before any group traversal
"""

import math
import random
from typing import Optional



# GoBoard  (mirrors game_info_t + board logic from GoBoard.hpp / Color.hpp)


class GoBoard:
    """
    Full Go board with rule enforcement.

    Color constants match Ray's stone enum in Color.hpp:
        S_EMPTY = 0, S_BLACK = 1, S_WHITE = 2
    """

    EMPTY = 0
    BLACK = 1
    WHITE = 2
    BOARD_SIZE = 9
    KOMI = 6.5  # From Constant.hpp


    # Construction / copy


    def __init__(self, size: int = 9):
        self.size = size
        self.grid = [[GoBoard.EMPTY] * size for _ in range(size)]
        self.current_player = GoBoard.BLACK
        self.move_history: list = []
        self.captures = {GoBoard.BLACK: 0, GoBoard.WHITE: 0}
        self.ko_point: Optional[tuple] = None
        self.consecutive_passes = 0
        self.move_count = 0
        # Pre-compute all positions for fast iteration
        self._all_positions = [(r, c) for r in range(size) for c in range(size)]

    def copy(self) -> "GoBoard":
        """Deep copy – used heavily by MCTS for simulations."""
        b = GoBoard.__new__(GoBoard)
        b.size = self.size
        b.grid = [row[:] for row in self.grid]
        b.current_player = self.current_player
        b.move_history = self.move_history[:]
        b.captures = dict(self.captures)
        b.ko_point = self.ko_point
        b.consecutive_passes = self.consecutive_passes
        b.move_count = self.move_count
        b._all_positions = self._all_positions  # shared, immutable
        return b


    # Basic accessors


    @staticmethod
    def opposite_color(color: int) -> int:
        """GetOppositeColor from Color.hpp: color XOR 0x3."""
        return color ^ 0x3

    def get(self, row: int, col: int) -> int:
        """Return stone color at (row, col)."""
        return self.grid[row][col]

    def to_list(self) -> list:
        """2D list for JSON serialization."""
        return [row[:] for row in self.grid]


    # Group / liberty helpers  (mirrors string handling in GoBoard.hpp)


    def _has_liberty(self, row: int, col: int) -> bool:
        """
        Fast check: does the group at (row,col) have at least one liberty?
        Early-exits as soon as any liberty is found – much faster than
        computing the full liberty set.
        """
        color = self.grid[row][col]
        if color == GoBoard.EMPTY:
            return True
        size = self.size
        grid = self.grid
        visited = set()
        stack = [(row, col)]
        while stack:
            r, c = stack.pop()
            if (r, c) in visited:
                continue
            visited.add((r, c))
            # Check four neighbors inline for speed
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < size and 0 <= nc < size:
                    n = grid[nr][nc]
                    if n == GoBoard.EMPTY:
                        return True  # Found a liberty – early exit
                    if n == color and (nr, nc) not in visited:
                        stack.append((nr, nc))
        return False

    def get_group(self, row: int, col: int) -> set:
        """Flood-fill to find the connected group containing (row, col)."""
        color = self.grid[row][col]
        if color == GoBoard.EMPTY:
            return set()
        size = self.size
        grid = self.grid
        visited = set()
        stack = [(row, col)]
        while stack:
            r, c = stack.pop()
            if (r, c) in visited:
                continue
            if not (0 <= r < size and 0 <= c < size):
                continue
            if grid[r][c] != color:
                continue
            visited.add((r, c))
            stack.append((r - 1, c))
            stack.append((r + 1, c))
            stack.append((r, c - 1))
            stack.append((r, c + 1))
        return visited

    def get_liberties(self, row: int, col: int) -> set:
        """Return the set of empty points adjacent to the group at (row, col)."""
        group = self.get_group(row, col)
        liberties: set = set()
        size = self.size
        grid = self.grid
        for r, c in group:
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < size and 0 <= nc < size and grid[nr][nc] == GoBoard.EMPTY:
                    liberties.add((nr, nc))
        return liberties

    def remove_group(self, row: int, col: int) -> int:
        """Remove all stones of the group at (row, col). Return count removed."""
        group = self.get_group(row, col)
        grid = self.grid
        for r, c in group:
            grid[r][c] = GoBoard.EMPTY
        return len(group)


    # Legality checks  (mirrors IsLegal / IsSuicide / IsLegalNotEye)


    def is_suicide(self, row: int, col: int, color: int) -> bool:
        """
        Fast suicide check. Optimized to avoid flood-fills when possible.
        """
        size = self.size
        grid = self.grid
        opp = color ^ 0x3

        # Quick check: if any neighbor is empty, it's never suicide
        for nr, nc in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            if 0 <= nr < size and 0 <= nc < size and grid[nr][nc] == GoBoard.EMPTY:
                return False

        # Temporarily place the stone for group analysis
        grid[row][col] = color

        # Check if we capture any opponent group (-> not suicide)
        captures_any = False
        for nr, nc in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            if 0 <= nr < size and 0 <= nc < size and grid[nr][nc] == opp:
                if not self._has_liberty(nr, nc):
                    captures_any = True
                    break

        if captures_any:
            grid[row][col] = GoBoard.EMPTY
            return False

        # Check if our own group has liberties
        has_lib = self._has_liberty(row, col)
        grid[row][col] = GoBoard.EMPTY
        return not has_lib

    def is_eye(self, row: int, col: int, color: int) -> bool:
        """
        A point is an eye for *color* if:
          1. All orthogonal neighbors are *color* (or off-board).
          2. At most one diagonal neighbor is the opponent (for non-edge),
             or zero for edge/corner points.
        """
        size = self.size
        grid = self.grid
        if grid[row][col] != GoBoard.EMPTY:
            return False

        # All orthogonal neighbors must be own color or off-board
        for nr, nc in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            if 0 <= nr < size and 0 <= nc < size and grid[nr][nc] != color:
                return False

        # Diagonal check
        opp = color ^ 0x3
        off_board = 0
        opp_count = 0
        for nr, nc in ((row - 1, col - 1), (row - 1, col + 1), (row + 1, col - 1), (row + 1, col + 1)):
            if not (0 <= nr < size and 0 <= nc < size):
                off_board += 1
            elif grid[nr][nc] == opp:
                opp_count += 1

        if off_board > 0:
            return opp_count == 0
        return opp_count <= 1

    def is_legal(self, row: int, col: int, color: int) -> bool:
        """Full legality check (empty + not suicide + not ko)."""
        if not (0 <= row < self.size and 0 <= col < self.size):
            return False
        if self.grid[row][col] != GoBoard.EMPTY:
            return False
        if self.ko_point is not None and (row, col) == self.ko_point:
            return False
        if self.is_suicide(row, col, color):
            return False
        return True


    # Move generation


    def get_legal_moves(self, color: int) -> list:
        """Return list of legal (row, col) for *color*."""
        moves = []
        size = self.size
        grid = self.grid
        for r in range(size):
            for c in range(size):
                if grid[r][c] == GoBoard.EMPTY and self.is_legal(r, c, color):
                    moves.append((r, c))
        return moves

    def get_legal_non_eye_moves(self, color: int) -> list:
        """Legal moves excluding own eyes – used by MCTS playouts."""
        moves = []
        size = self.size
        grid = self.grid
        for r in range(size):
            for c in range(size):
                if grid[r][c] == GoBoard.EMPTY and self.is_legal(r, c, color) and not self.is_eye(r, c, color):
                    moves.append((r, c))
        return moves


    # Place stone / pass  (mirrors PutStone from GoBoard.hpp)


    def place_stone(self, row: int, col: int, color: int) -> bool:
        """
        Place a stone. Returns True on success, False if illegal.
        Handles captures, ko detection, and state bookkeeping.
        """
        if not self.is_legal(row, col, color):
            return False

        grid = self.grid
        size = self.size
        grid[row][col] = color
        opp = color ^ 0x3

        # Capture opponent groups with 0 liberties
        captured = 0
        captured_pos = None
        for nr, nc in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            if 0 <= nr < size and 0 <= nc < size and grid[nr][nc] == opp:
                if not self._has_liberty(nr, nc):
                    group = self.get_group(nr, nc)
                    count = len(group)
                    for r, c in group:
                        grid[r][c] = GoBoard.EMPTY
                    captured += count
                    if count == 1:
                        captured_pos = next(iter(group))

        self.captures[color] = self.captures.get(color, 0) + captured

        # Ko detection
        if captured == 1 and captured_pos is not None:
            own_group = self.get_group(row, col)
            own_libs = self.get_liberties(row, col)
            if len(own_group) == 1 and len(own_libs) == 1:
                self.ko_point = captured_pos
            else:
                self.ko_point = None
        else:
            self.ko_point = None

        self.consecutive_passes = 0
        self.move_count += 1
        self.move_history.append((color, row, col))
        self.current_player = color ^ 0x3
        return True

    def place_stone_fast(self, row: int, col: int, color: int) -> bool:
        """
        Fast stone placement for simulations. Skips move history and some bookkeeping.
        Returns True on success, False if illegal.
        """
        grid = self.grid
        size = self.size

        # Inline legality: must be empty, not ko
        if grid[row][col] != GoBoard.EMPTY:
            return False
        if self.ko_point is not None and self.ko_point == (row, col):
            return False

        opp = color ^ 0x3

        # Quick suicide pre-check: if any neighbor is empty, not suicide
        has_empty_neighbor = False
        for nr, nc in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            if 0 <= nr < size and 0 <= nc < size and grid[nr][nc] == GoBoard.EMPTY:
                has_empty_neighbor = True
                break

        if not has_empty_neighbor:
            # Need deeper check
            grid[row][col] = color
            captures_any = False
            for nr, nc in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
                if 0 <= nr < size and 0 <= nc < size and grid[nr][nc] == opp:
                    if not self._has_liberty(nr, nc):
                        captures_any = True
                        break
            if not captures_any and not self._has_liberty(row, col):
                grid[row][col] = GoBoard.EMPTY
                return False  # Suicide
            grid[row][col] = GoBoard.EMPTY

        # Place the stone
        grid[row][col] = color

        # Capture opponent groups
        captured = 0
        captured_pos = None
        for nr, nc in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            if 0 <= nr < size and 0 <= nc < size and grid[nr][nc] == opp:
                if not self._has_liberty(nr, nc):
                    grp = self.get_group(nr, nc)
                    cnt = len(grp)
                    for r, c in grp:
                        grid[r][c] = GoBoard.EMPTY
                    captured += cnt
                    if cnt == 1:
                        captured_pos = next(iter(grp))

        # Ko detection (simplified)
        if captured == 1 and captured_pos is not None:
            own_grp = self.get_group(row, col)
            if len(own_grp) == 1:
                own_libs = self.get_liberties(row, col)
                if len(own_libs) == 1:
                    self.ko_point = captured_pos
                else:
                    self.ko_point = None
            else:
                self.ko_point = None
        else:
            self.ko_point = None

        self.consecutive_passes = 0
        self.current_player = opp
        return True

    def pass_move(self, color: int):
        """Register a pass for *color*."""
        self.consecutive_passes += 1
        self.move_count += 1
        self.move_history.append((color, None, None))
        self.current_player = color ^ 0x3
        self.ko_point = None


    # Game-over / scoring  (mirrors CalculateScore from GoBoard.hpp)


    def is_game_over(self) -> bool:
        return self.consecutive_passes >= 2

    def calculate_score(self) -> dict:
        """
        Chinese rules (area scoring).
        Score = stones on board + territory (empty points enclosed by one color).
        White receives KOMI.
        """
        grid = self.grid
        size = self.size
        black_score = 0
        white_score = 0
        visited = [[False] * size for _ in range(size)]

        for r in range(size):
            row = grid[r]
            for c in range(size):
                v = row[c]
                if v == GoBoard.BLACK:
                    black_score += 1
                elif v == GoBoard.WHITE:
                    white_score += 1

        for r in range(size):
            for c in range(size):
                if grid[r][c] == GoBoard.EMPTY and not visited[r][c]:
                    region = []
                    borders = set()
                    queue = [(r, c)]
                    visited[r][c] = True
                    head = 0
                    while head < len(queue):
                        cr, cc = queue[head]
                        head += 1
                        region.append((cr, cc))
                        for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                            if 0 <= nr < size and 0 <= nc < size:
                                nv = grid[nr][nc]
                                if nv != GoBoard.EMPTY:
                                    borders.add(nv)
                                elif not visited[nr][nc]:
                                    visited[nr][nc] = True
                                    queue.append((nr, nc))

                    if borders == {GoBoard.BLACK}:
                        black_score += len(region)
                    elif borders == {GoBoard.WHITE}:
                        white_score += len(region)

        white_score_with_komi = white_score + GoBoard.KOMI
        winner = "B" if black_score > white_score_with_komi else "W"

        return {
            "black": black_score,
            "white": white_score_with_komi,
            "winner": winner,
        }



# MCTSNode  (mirrors uct_node_t / child_node_t from MCTSNode.hpp)


class MCTSNode:
    """
    A single node in the Monte-Carlo tree.

    Attributes mirror Ray's uct_node_t:
        move          -> previous_move
        visits        -> move_count
        wins          -> win
        children      -> child[]
        untried_moves -> candidates not yet expanded
    """

    __slots__ = ("board", "color", "parent", "move", "children",
                 "visits", "wins", "untried_moves")

    def __init__(self, board: GoBoard, color: int,
                 parent: "MCTSNode" = None, move: tuple = None):
        self.board = board
        self.color = color
        self.parent = parent
        self.move = move
        self.children: list["MCTSNode"] = []
        self.visits = 0
        self.wins = 0.0
        self.untried_moves = board.get_legal_non_eye_moves(color)
        random.shuffle(self.untried_moves)


    # UCB1-Tuned  (from UCBEvaluation.cpp)


    def ucb1(self, exploration_weight: float = 1.414) -> float:
        """
        UCB1-Tuned value from Ray's UCBEvaluation.cpp:
            p   = win / move_count
            div = log(parent_visits) / move_count
            v   = p - p*p + sqrt(2 * div)
            ucb = p + C * sqrt(div * min(0.25, v))
        """
        if self.visits == 0:
            return float("inf")

        p = self.wins / self.visits
        div = math.log(self.parent.visits) / self.visits
        v = p - p * p + math.sqrt(2.0 * div)
        return p + exploration_weight * math.sqrt(div * min(0.25, v))


    # Selection  (pick child with highest UCB)


    def select_child(self) -> "MCTSNode":
        return max(self.children, key=lambda c: c.ucb1())


    # Expansion  (InitializeCandidate + add child)


    def expand(self) -> "MCTSNode":
        """Expand one untried move and return the new child node."""
        move = self.untried_moves.pop()
        new_board = self.board.copy()
        new_board.place_stone(move[0], move[1], self.color)
        child_color = self.color ^ 0x3
        child = MCTSNode(new_board, child_color, parent=self, move=move)
        self.children.append(child)
        return child


    # Simulation  (fast random playout)


    def simulate(self) -> int:
        """
        Run a fast random playout from this node's board state.
        Returns the winning color (BLACK or WHITE).

        Optimization: instead of generating all legal moves each turn,
        we shuffle all positions and try them in order until one works.
        This avoids the O(n^2) legal-move-generation per turn.
        """
        sim_board = self.board.copy()
        current = self.color
        size = sim_board.size
        grid = sim_board.grid
        max_moves = size * size  # Reduced from 2*n^2 for speed
        all_pos = list(sim_board._all_positions)
        consecutive_passes = sim_board.consecutive_passes

        for _ in range(max_moves):
            if consecutive_passes >= 2:
                break

            # Shuffle positions and try to find a playable one
            random.shuffle(all_pos)
            played = False
            for r, c in all_pos:
                if grid[r][c] != GoBoard.EMPTY:
                    continue
                # Skip eyes
                if sim_board.is_eye(r, c, current):
                    continue
                # Try to place
                if sim_board.place_stone_fast(r, c, current):
                    played = True
                    consecutive_passes = 0
                    break

            if not played:
                consecutive_passes += 1
                sim_board.ko_point = None

            current = current ^ 0x3

        score = sim_board.calculate_score()
        return GoBoard.BLACK if score["winner"] == "B" else GoBoard.WHITE


    # Back-propagation  (UpdateResult from MCTSNode.hpp)


    def backpropagate(self, winner: int):
        """Walk up the tree, updating visits and wins."""
        node = self
        while node is not None:
            node.visits += 1
            if node.parent is not None:
                mover = node.color ^ 0x3
                if mover == winner:
                    node.wins += 1.0
            node = node.parent


    # Best child (most visited, robust child selection)


    def best_child(self) -> "MCTSNode":
        """Return child with the most visits (robust selection)."""
        return max(self.children, key=lambda c: c.visits)



# MCTSEngine  (orchestrates the MCTS loop)


class MCTSEngine:
    """
    High-level MCTS controller.

    Mirrors Ray's main search loop:
        for each simulation:
            1. Selection   – walk tree via UCB1-Tuned
            2. Expansion   – add one child
            3. Simulation  – random playout
            4. Back-prop   – update stats up the tree
    """

    def __init__(self, simulations: int = 1000):
        self.simulations = simulations


    # Core search


    def _run_mcts(self, board: GoBoard, color: int,
                  simulations: int, callback=None) -> MCTSNode:
        """
        Run *simulations* iterations of MCTS and return the root node.
        If *callback* is provided it is called every 50 iterations with
        a stats dict for live visualization.
        """
        root = MCTSNode(board.copy(), color)

        for i in range(1, simulations + 1):
            node = root

            # 1. Selection
            while not node.untried_moves and node.children:
                node = node.select_child()

            # 2. Expansion
            if node.untried_moves:
                node = node.expand()

            # 3. Simulation
            winner = node.simulate()

            # 4. Back-propagation
            node.backpropagate(winner)

            # Live callback
            if callback and i % 50 == 0:
                stats = self._build_stats(root, i)
                callback(stats)

        return root


    # Stats helpers


    @staticmethod
    def _child_win_rate(child: MCTSNode) -> float:
        if child.visits == 0:
            return 0.0
        return child.wins / child.visits

    def _build_stats(self, root: MCTSNode, total_sims: int) -> dict:
        """Build a stats dict from the current root."""
        top = sorted(root.children, key=lambda c: c.visits, reverse=True)[:5]
        best = top[0] if top else None
        return {
            "total_simulations": total_sims,
            "best_move": list(best.move) if best else None,
            "best_win_rate": round(self._child_win_rate(best), 4) if best else 0,
            "top_moves": [
                {
                    "move": list(c.move),
                    "visits": c.visits,
                    "win_rate": round(self._child_win_rate(c), 4),
                    "ucb1": round(c.ucb1(), 4) if c.visits > 0 else 0,
                }
                for c in top
            ],
        }


    # Public API


    def get_best_move(self, board: GoBoard, color: int,
                      callback=None) -> tuple:
        """
        Return the best move (row, col) or None (= pass).
        """
        legal = board.get_legal_non_eye_moves(color)
        if not legal:
            return None

        root = self._run_mcts(board, color, self.simulations, callback)

        if not root.children:
            return None

        best = root.best_child()
        return best.move

    def analyse_position(self, board: GoBoard, color: int,
                         top_n: int = 5) -> list:
        """
        Return the top-N candidate moves with statistics.
        """
        root = self._run_mcts(board, color, self.simulations)
        children_sorted = sorted(root.children,
                                 key=lambda c: c.visits, reverse=True)[:top_n]
        return [
            {
                "move": list(c.move),
                "visits": c.visits,
                "win_rate": round(self._child_win_rate(c), 4),
                "ucb1": round(c.ucb1(), 4) if c.visits > 0 else 0,
            }
            for c in children_sorted
        ]

    def get_live_stats(self, board: GoBoard, color: int,
                       simulations: int, callback):
        """
        Run MCTS and invoke *callback(stats_dict)* every 50 iterations.
        """
        self._run_mcts(board, color, simulations, callback)
