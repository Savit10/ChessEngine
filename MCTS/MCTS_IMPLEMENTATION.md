# MCTS Implementation Summary

## Overview

This document summarizes the Monte Carlo Tree Search (MCTS) implementation for the Chess Engine project. The implementation follows the AlphaZero-style MCTS algorithm, integrating neural network policy and value predictions with tree search.

## Architecture

### Core Components

#### 1. Abstract Base Classes (`MCTS/include/state.h`)

- **`MCTS_move`**: Abstract base class for game moves
  - Pure virtual `operator==()` for move comparison
  - Virtual `sprint()` for string representation

- **`MCTS_state`**: Abstract base class for game states
  - `actions_to_try()`: Returns queue of legal moves
  - `next_state()`: Creates new state after applying a move
  - `rollout()`: Performs random simulation, returns [0, 1] winrate
  - `is_terminal()`: Checks if game is over
  - `player1_turn()`: Returns true if it's player 1's turn
  - `print()`: Displays the current state

#### 2. MCTS Node (`MCTS/include/mcts.h`, `MCTS/src/mcts.cpp`)

**`MCTS_node`** represents a single node in the search tree:

**Key Fields:**
- `state`: Current game state
- `move`: Move that led to this node
- `parent`: Pointer to parent node
- `children`: Vector of child nodes
- `untried_actions`: Queue of unexplored moves
- `score`: Accumulated value from simulations
- `number_of_simulations`: Visit count
- `size`: Total size of subtree
- `policy_priors`: Map of move UCI strings to prior probabilities (from NN)
- `nn_value`: Value prediction from neural network
- `has_nn_evaluation`: Flag indicating if node has been evaluated by NN

**Key Methods:**
- `expand()`: Adds one child node (does not evaluate)
- `evaluate(NeuralNetwork* nn)`: Evaluates node with NN or rollout
- `backpropagate_value(double w, int n)`: Backpropagates value up the tree
- `select_best_child(double cpuct)`: Selects best child using PUCT formula
- `advance_tree(const MCTS_move* m)`: Moves root to child node after move
- `get_prior(const MCTS_move* move)`: Retrieves prior probability for a move
- `is_fully_expanded()`: Checks if all moves have been explored
- `is_terminal()`: Checks if state is terminal
- `is_evaluated()`: Checks if node has been evaluated

#### 3. MCTS Tree (`MCTS/include/mcts.h`, `MCTS/src/mcts.cpp`)

**`MCTS_tree`** manages the entire search tree:

**Key Fields:**
- `root`: Root node of the tree
- `nn_`: Pointer to neural network (optional)
- `cpuct_`: PUCT exploration constant

**Key Methods:**
- `select()`: Traverses tree to find leaf node for evaluation/expansion
- `grow_tree(int max_iter, double max_time)`: Main MCTS loop
- `select_best_child()`: Selects best move from root
- `advance_tree(const MCTS_move* move)`: Moves root after opponent's move
- `get_size()`: Returns total tree size
- `get_current_state()`: Returns current game state

#### 4. MCTS Agent (`MCTS/include/mcts.h`, `MCTS/src/mcts.cpp`)

**`MCTS_agent`** provides high-level interface for game play:

**Key Methods:**
- `genmove(const MCTS_move* enemy_move)`: Generates best move
- `get_current_state()`: Returns current game state
- `feedback()`: Prints tree statistics

## Algorithm Details

### MCTS Iteration Loop (`grow_tree()`)

The main MCTS loop follows the AlphaZero pattern:

```cpp
for each iteration:
    1. SELECT: Traverse tree to find leaf node
    2. EVALUATE: If not evaluated, evaluate with NN or rollout
    3. EXPAND: If not fully expanded, add one child
    4. BACKPROPAGATE: Propagate value up the tree
```

**Selection (`select()`):**
- Starts at root
- While node is fully expanded and evaluated:
  - Select best child using PUCT
  - Move to that child
- Returns leaf node (either unevaluated or not fully expanded)

**Evaluation (`evaluate()`):**
- If terminal: Returns game result (0.0, 0.5, or 1.0)
- If NN available: Calls `nn->predict()` to get policy and value
  - Stores policy in `policy_priors` map
  - Stores value in `nn_value`
- Otherwise: Performs random rollout simulation
- Marks node as evaluated

**Expansion (`expand()`):**
- Pops one move from `untried_actions`
- Creates new child node with that move
- Adds child to `children` vector
- Returns new child node

**Backpropagation (`backpropagate_value()`):**
- Updates `score` and `number_of_simulations` for current node
- Recursively updates parent nodes
- Updates subtree sizes

### PUCT Selection Formula

The **PUCT (Polynomial Upper Confidence bound for Trees)** formula balances exploitation and exploration:

```
PUCT(s,a) = Q(s,a) + cpuct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
```

Where:
- **Q(s,a)**: Average value (winrate) from simulations
  - `Q = child->score / child->number_of_simulations`
- **P(s,a)**: Prior probability from neural network
  - Retrieved via `get_prior(child->move)`
  - Falls back to uniform `1/num_children` if no prior found
- **N(s)**: Parent's total visit count
- **N(s,a)**: This child's visit count
- **cpuct**: Exploration constant (typically 1.0-2.0)

**Fallback to UCT:**
If no prior probability is available (P = 0.0), uses standard UCB:
```
UCT(s,a) = Q(s,a) + cpuct * sqrt(log(N(s) + 1) / (1 + N(s,a)))
```

### Tree Reuse Optimization

When opponent makes a move:
- `advance_tree(move)` finds the child node corresponding to that move
- Makes that child the new root
- Deletes all other branches
- Preserves subtree statistics for faster convergence

## Chess Implementation

### Chess Move (`MCTS/Chess/Chess.h`)

**`Chess_move`** wraps `chess::Move`:
- Uses `static_cast<string>(Square)` for UCI string conversion
- Handles promotions (adds 'q', 'n', 'b', 'r' suffix)
- Ensures exact string matching with neural network policy keys

### Chess State (`MCTS/Chess/Chess.h`, `MCTS/Chess/Chess.cpp`)

**`Chess_state`** implements `MCTS_state` using `chess.hpp`:

**Key Features:**
- Uses `chess::Board` for game logic
- `actions_to_try()`: Generates all legal moves using `movegen::legalmoves()`
- `next_state()`: Creates new state by copying board and applying move
- `rollout()`: Random simulation with material-based evaluation
- `evaluate_position()`: Material count evaluation (pawn=100, knight=320, bishop=330, rook=500, queen=900, king=20000)
- `is_terminal()`: Checks `board.isGameOver()`
- `player1_turn()`: Returns `board.sideToMove() == Color::WHITE`

**Rollout Strategy:**
- Simulates random moves up to 500 moves
- If terminal reached: Returns game result (0.0, 0.5, or 1.0)
- If max moves reached: Returns material-based evaluation normalized to [0, 1]

## Neural Network Integration

### Neural Network Wrapper (`MCTS/include/neural_network.h`, `MCTS/src/neural_network.cpp`)

**`NeuralNetwork`** class provides interface to PyTorch model:

**Key Methods:**
- `load_model(string path)`: Loads TorchScript model (`.pt` file)
- `predict(Board& board, map<string, double>& policy_out, double& value_out)`:
  - Encodes board to 12-channel 8x8 tensor
  - Runs model inference
  - Extracts policy logits (4096-dim) and value (1-dim)
  - Maps policy to legal moves (UCI strings)
  - Normalizes policy probabilities
  - Converts value from [-1, 1] to [0, 1] (white's perspective)

**Board Encoding:**
- 12 channels: 6 piece types × 2 colors
- Channels: [white_pawn, white_knight, white_bishop, white_rook, white_queen, white_king,
              black_pawn, black_knight, black_bishop, black_rook, black_queen, black_king]
- No vertical flipping (matches Python training code)
- Uses `Square::index()` for square indexing (matches python-chess)

**Policy Mapping:**
- Policy index: `from_square * 64 + to_square` (4096 dimensions)
- Maps to UCI strings: `"e2e4"`, `"g1f3"`, etc.
- Normalizes over legal moves only

### Integration Points

1. **Node Evaluation**: When a leaf node is evaluated, `evaluate()` calls `nn->predict()` if NN is available
2. **Policy Storage**: Policy map is stored in `node->policy_priors` (keyed by UCI strings)
3. **Prior Retrieval**: `get_prior()` looks up prior probability from parent's `policy_priors`
4. **PUCT Formula**: Uses prior probability `P(s,a)` in exploration term

## Applications

### 1. Self-Play (`MCTS/Chess/main.cpp`)

Two MCTS agents play against each other:
- Configurable iterations/time per move
- Outputs FEN after each move
- Generates PGN at end of game
- Shows statistics: tree size, time, positions/second

**Usage:**
```bash
./chess_mcts [model_path]
```

### 2. Interactive Play (`MCTS/Chess/play.cpp`)

Human vs. MCTS engine:
- User chooses color (White or Black)
- User inputs moves in UCI format
- Engine shows thinking time and statistics
- Displays board, FEN, and last move after each turn
- Generates PGN at end

**Usage:**
```bash
./chess_play [model_path]
```

**Features:**
- Move validation
- Error handling for illegal moves
- Game termination detection
- Last move display (SAN + UCI)

## Configuration Parameters

### MCTS Parameters

- **`MAX_ITERATIONS`**: Maximum simulations per move (default: 5000)
- **`MAX_SECONDS`**: Maximum time per move in seconds (default: 2)
- **`CPUCT`**: PUCT exploration constant (default: 1.0-2.0)
- **`MAX_MOVES`**: Safety limit for game length (default: 1000)

### Neural Network Parameters

- **Model path**: Default `../aznet_traced.pt`, can be specified via command line
- **Policy size**: 4096 dimensions (64×64 from_square × to_square)
- **Value range**: [-1, 1] from current player's perspective, converted to [0, 1] for white

## Key Fixes and Improvements

### 1. Policy Prior Lookup Bug
**Problem**: `get_prior()` was looking in `parent->policy_priors`, but for root node, `parent == nullptr`
**Fix**: Check if at root, use `this->policy_priors`; otherwise use `parent->policy_priors`

### 2. String Format Mismatch
**Problem**: `Chess_move::sprint()` used manual string building, while NN used `static_cast<string>(Square)`
**Fix**: Changed `sprint()` to use `static_cast<string>(Square)` for exact matching

### 3. MCTS Loop Refactoring
**Problem**: Original implementation had double backpropagation and incorrect value handling
**Fix**: Separated `expand()` (add child) from `evaluate()` (get value/policy), following AlphaZero pattern

### 4. PUCT Formula Corrections
**Problem**: 
- Used `sum_b N(s,b)` instead of `N(s)` for parent visits
- Incorrectly flipped Q-values
**Fix**: 
- Changed to `sqrt(N(s))` for parent's total visits
- Removed Q-value flipping (backpropagation handles perspective)

### 5. Exploration Term
**Problem**: When no prior available, exploration was disabled
**Fix**: Added UCT fallback when `P == 0.0` to ensure exploration even without NN

## Statistics and Debugging

### Debug Output

**Root Node Selection:**
- Shows all moves with Q, P, N_a, and PUCT scores
- Sorted by PUCT score (highest first)
- Highlights target move (if specified)

**Root Node Evaluation:**
- Shows policy map size and all policy entries
- Logs when NN prediction fails

**Tree Statistics:**
- Tree size (total nodes)
- Number of simulations
- Branching factor
- Winrate from current player's perspective
- Top moves with winrates

### Performance Metrics

- **Positions/second**: Estimated based on iterations and time
- **Tree size**: Total number of nodes in search tree
- **Time per move**: Actual time taken for move selection
- **Total positions evaluated**: Sum of all simulations across game

## File Structure

```
MCTS/
├── include/
│   ├── state.h              # Abstract MCTS_move and MCTS_state
│   ├── mcts.h               # MCTS_node, MCTS_tree, MCTS_agent
│   ├── neural_network.h     # NeuralNetwork wrapper
│   ├── JobScheduler.h       # Thread pool for parallel rollouts
│   └── chess.hpp            # Chess library
├── src/
│   ├── mcts.cpp             # MCTS algorithm implementation
│   ├── neural_network.cpp   # NN inference and board encoding
│   └── JobScheduler.cpp    # Thread pool implementation
├── Chess/
│   ├── Chess.h              # Chess_move and Chess_state
│   ├── Chess.cpp            # Chess game logic
│   ├── main.cpp             # Self-play between two agents
│   └── play.cpp             # Interactive human vs. engine
└── Makefile                 # Build configuration
```

## Build Instructions

### Prerequisites
- C++17 compiler
- LibTorch (PyTorch C++ API)
- OpenMP (for parallel rollouts)

### Compilation

```bash
cd MCTS
make Play    # Builds interactive play executable
make Chess   # Builds self-play executable
```

### Running

```bash
# Interactive play
./chess_play [model_path]

# Self-play
./chess_mcts [model_path]
```

## Future Improvements

1. **Virtual Loss**: Prevent multiple threads from exploring same path
2. **Dirichlet Noise**: Add noise to root policy for exploration
3. **Temperature**: Use temperature for move selection in early game
4. **Transposition Table**: Cache evaluations for repeated positions
5. **Progressive Widening**: Limit children based on visit count
6. **Time Management**: Adaptive time allocation based on game phase
7. **Opening Book**: Use opening book for first few moves
8. **Endgame Tablebase**: Use tablebase for endgame positions

## References

- **AlphaZero Paper**: Silver et al., "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm"
- **MCTS Survey**: Browne et al., "A Survey of Monte Carlo Tree Search Methods"
- **PUCT Formula**: Rosin, "Multi-armed bandits with episode context"

