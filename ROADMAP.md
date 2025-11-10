# Chess Engine Development Roadmap

## Project Overview
Building a chess engine using deep reinforcement learning with Monte Carlo Tree Search (MCTS) and neural networks.

---

## Phase 1: Bootstrapping 🚀

**Goal**: Create a minimal working prototype where a random neural network + MCTS + data pipeline run end-to-end, producing dummy training samples and a full game loop.

### ✅ End-Goal of Phase 1
- Minimal working prototype with:
  - Random NN + MCTS + data pipeline running end-to-end
  - Dummy training samples generation
  - Full game loop execution

---

## Team Roles & Deliverables

### 🧠 Person 1 — Neural Network Lead

**Goal**: Build the CNN scaffold to process a chess board and output placeholder policy + value.

**Key Tasks**:
- [ ] Encode board → tensor (e.g., 8×8×12)
- [ ] Create CNN with two heads:
  - Policy head (move logits)
  - Value head (win probability)
- [ ] Stub `predict(board_state)` returning random outputs

**Deliverables**:
- `model.py` - Neural network architecture
- `board_encoding.py` - Board to tensor conversion
- Small test script to verify encoding and model structure

---

### 🌲 Person 2 — Search & Game Logic Lead (You)

**Goal**: Build chess environment and MCTS using dummy NN outputs.

**Key Tasks**:
- [ ] Implement basic game API:
  - `get_legal_moves()`
  - `apply_move()`
  - `is_game_over()`
  - `get_result()`
  - etc.
- [ ] Implement MCTS:
  - Node structure
  - Selection via UCB (Upper Confidence Bound)
  - Expansion
  - Rollout/simulation
  - Backpropagation
- [ ] Test by running a random game to completion

**Deliverables**:
- `game.py` - Chess game logic and API
- `mcts.py` - Monte Carlo Tree Search implementation
- CLI demo - Interactive or automated game demonstration

---

### 📊 Person 3 — Data Pipeline & Integration Lead

**Goal**: Create data storage + mock training pipeline linking self-play → dataset → train.

**Key Tasks**:
- [ ] Define sample structure: `{state, policy_target, value_target}`
- [ ] Implement replay buffer and file storage
- [ ] Write fake `train_step()` computing random loss and logging
- [ ] Set up repo, logging, and version control

**Deliverables**:
- `data_pipeline.py` - Data collection and storage
- `train_stub.py` - Mock training loop
- Repo structure + logs folder
- Version control setup

**Note**: 
- Training will be done on Google Colab for GPU access. See `COLAB_SETUP.md` for setup instructions.
- **Person 3 can work independently!** No need to wait for Person 1 or Person 2. See `INTERFACES.md` for data format specifications.

---

## Future Phases (TBD)

### Phase 2: Training Pipeline
- Real neural network training
- Self-play implementation
- Data collection from actual games

### Phase 3: Optimization
- Performance improvements
- Hyperparameter tuning
- Advanced search techniques

### Phase 4: Evaluation & Deployment
- Engine strength evaluation
- UCI protocol support
- Performance benchmarking

---

## Repository Structure

```
ChessEngine/
├── model.py                    # Neural network (Person 1)
├── board_encoding.py           # Board encoding (Person 1)
├── game.py                     # Chess game logic (Person 2)
├── mcts.py                     # MCTS implementation (Person 2)
├── data_pipeline.py            # Data collection (Person 3)
├── train_stub.py               # Training loop (Person 3)
├── config.py                   # Shared configuration
├── sync_colab.py               # Colab sync helper
├── colab_training_template.ipynb # Colab training template
├── logs/                       # Training logs (Person 3)
├── data/                       # Training data (Person 3)
├── models/                     # Saved models
├── tests/                      # Test scripts
├── requirements.txt            # Python dependencies
├── COLAB_SETUP.md              # Colab setup guide
├── INTERFACES.md                # Component interface specifications
├── README.md                   # Project overview
└── ROADMAP.md                  # This file
```

---

## Notes
- All Phase 1 components should work with dummy/random outputs
- Focus on clean interfaces between components
- Each person should test their component independently before integration
- **Everyone can work in parallel!** See `INTERFACES.md` for component interface specifications

