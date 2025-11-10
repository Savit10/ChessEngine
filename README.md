# Chess Engine

A chess engine built using deep reinforcement learning with Monte Carlo Tree Search (MCTS) and neural networks.

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ChessEngine
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # On macOS/Linux:
   source venv/bin/activate
   
   # On Windows:
   venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**
   ```bash
   python -c "import chess; import torch; import numpy; print('All dependencies installed!')"
   ```

## 📁 Project Structure

```
ChessEngine/
├── model.py              # Neural network (Person 1)
├── board_encoding.py     # Board encoding (Person 1)
├── game.py               # Chess game logic (Person 2)
├── mcts.py               # MCTS implementation (Person 2)
├── data_pipeline.py      # Data collection (Person 3)
├── train_stub.py         # Training loop (Person 3)
├── config.py             # Shared configuration
├── logs/                 # Training logs
├── tests/                # Test scripts
├── requirements.txt      # Python dependencies
├── ROADMAP.md            # Development roadmap
├── INTERFACES.md          # Component interface specifications
└── README.md             # This file
```

## 🎯 Development Status

Currently in **Phase 1: Bootstrapping**

See [ROADMAP.md](ROADMAP.md) for detailed development plan and team responsibilities.

## 🛠️ Key Dependencies

- **python-chess**: Chess board representation and move validation
- **PyTorch**: Deep learning framework for neural networks
- **NumPy**: Numerical computing

## 📝 Usage

(To be updated as components are implemented)

## 🔗 Google Colab Integration

For GPU training and data pipelining, use Google Colab. See [COLAB_SETUP.md](COLAB_SETUP.md) for detailed setup instructions.

**Quick Colab Setup:**
1. Open `colab_training_template.ipynb` in Google Colab
2. Clone the repository in Colab
3. Install dependencies: `!pip install -r requirements.txt`
4. Enable GPU: Runtime → Change runtime type → GPU
5. Run training and save results to Google Drive

## 🤝 Contributing

This is a collaborative project. Each team member is responsible for their assigned component:
- Person 1: Neural Network
- Person 2: Game Logic & MCTS
- Person 3: Data Pipeline & Integration

## 📄 License

(To be determined)

