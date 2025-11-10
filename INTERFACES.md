# Component Interfaces Specification

This document defines the interfaces between components so everyone can work in parallel without blocking each other.

## 🎯 Overview

All components are **modular and independent**. Each person can work on their component using **dummy/mock data** until integration.

---

## 📊 Data Pipeline Interface (Person 3)

### What Person 3 Can Do **RIGHT NOW** (No Dependencies):

1. **Define Data Structure** ✅
   - Create the sample structure format
   - Implement replay buffer
   - Set up file storage
   - Write logging infrastructure

2. **Create Mock Training Loop** ✅
   - Write `train_step()` that accepts dummy data
   - Implement loss computation (can use random values for now)
   - Set up logging and checkpointing

3. **Test with Dummy Data** ✅
   - Generate fake samples: `{state, policy_target, value_target}`
   - Test replay buffer with dummy data
   - Test training loop with dummy batches

### Expected Data Format (Agreed Upon):

```python
# Training Sample Structure
{
    'state': np.ndarray,        # Shape: (8, 8, 12) - board encoding
    'policy_target': np.ndarray, # Shape: (POLICY_SIZE,) - move probabilities
    'value_target': float        # Range: [-1, 1] - game outcome
}
```

**Person 3 can use this format immediately with dummy data!**

---

## 🧠 Neural Network Interface (Person 1)

### Expected Interface:

```python
# model.py
class ChessModel(nn.Module):
    def forward(self, board_tensor):
        """
        Args:
            board_tensor: torch.Tensor, shape (batch_size, 12, 8, 8)
        
        Returns:
            policy: torch.Tensor, shape (batch_size, POLICY_SIZE)
            value: torch.Tensor, shape (batch_size, 1)
        """
        pass

# board_encoding.py
def encode_board(board: chess.Board) -> np.ndarray:
    """
    Args:
        board: chess.Board object
    
    Returns:
        encoded: np.ndarray, shape (8, 8, 12)
    """
    pass

def predict(board: chess.Board) -> tuple:
    """
    Args:
        board: chess.Board object
    
    Returns:
        policy: np.ndarray, shape (POLICY_SIZE,)
        value: float, range [-1, 1]
    """
    # For Phase 1: return random values
    pass
```

**Person 1 can implement this independently!**

---

## 🌲 Game & MCTS Interface (Person 2)

### Expected Interface:

```python
# game.py
class ChessGame:
    def get_legal_moves(self) -> list:
        """Returns list of legal moves"""
        pass
    
    def apply_move(self, move):
        """Apply a move to the board"""
        pass
    
    def is_game_over(self) -> bool:
        """Check if game is over"""
        pass
    
    def get_result(self) -> float:
        """
        Returns:
            -1.0 if current player lost
            0.0 if draw
            1.0 if current player won
        """
        pass
    
    def get_board(self) -> chess.Board:
        """Get current board state"""
        pass

# mcts.py
class MCTS:
    def search(self, game: ChessGame, num_simulations: int) -> dict:
        """
        Args:
            game: ChessGame object
            num_simulations: int
        
        Returns:
            {
                'policy': np.ndarray,  # Move probabilities, shape (POLICY_SIZE,)
                'value': float         # Estimated value, range [-1, 1]
            }
        """
        # For Phase 1: can use dummy NN predictions
        pass
```

**Person 2 can implement this independently using dummy NN outputs!**

---

## 🔄 Integration Flow

### How Components Connect:

```
┌─────────────┐
│   Person 2  │  Game + MCTS
│  (game.py,  │  ────────────┐
│   mcts.py)  │              │
└─────────────┘              │
                             │
                             ▼
                    ┌─────────────────┐
                    │  Self-Play Loop │
                    │  (Integration)  │
                    └─────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   Person 3      │
                    │  Data Pipeline  │
                    │  (Collects      │
                    │   samples)      │
                    └─────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   Person 3      │
                    │  Training Loop  │
                    │  (train_stub.py)│
                    └─────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   Person 1      │
                    │  Neural Network │
                    │  (model.py)     │
                    └─────────────────┘
```

### Integration Points:

1. **MCTS → Model**: MCTS calls `model.predict(board)` to get policy/value
2. **Game → Data Pipeline**: After each game, collect `(state, policy, value)` samples
3. **Data Pipeline → Training**: Training loop loads batches from replay buffer
4. **Training → Model**: Training loop updates model weights

---

## ✅ What Each Person Can Do Independently

### Person 1 (Neural Network):
- ✅ Implement `board_encoding.py` with `encode_board()`
- ✅ Implement `model.py` with CNN architecture
- ✅ Create `predict()` function that returns random values (for Phase 1)
- ✅ Test with dummy chess boards

### Person 2 (Game Logic & MCTS):
- ✅ Implement `game.py` with chess game API
- ✅ Implement `mcts.py` with MCTS algorithm
- ✅ Use dummy/random NN outputs for MCTS (for Phase 1)
- ✅ Test by running random games to completion

### Person 3 (Data Pipeline):
- ✅ Define data sample structure
- ✅ Implement `data_pipeline.py` with replay buffer
- ✅ Implement `train_stub.py` with mock training loop
- ✅ Test with dummy data samples
- ✅ Set up logging infrastructure
- ✅ **NO WAITING NEEDED!** 🎉

---

## 🧪 Testing Strategy

Each person should:
1. **Test independently** with dummy/mock data
2. **Define clear interfaces** (use this document)
3. **Write unit tests** for their components
4. **Integration testing** happens after all components are ready

---

## 📝 Notes

- **All Phase 1 work uses dummy/random outputs** - no one needs real implementations from others
- **Interfaces are contracts** - as long as everyone follows these interfaces, integration will be smooth
- **Person 3 can start immediately** - they just need to know the data format (defined above)

