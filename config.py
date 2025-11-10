"""
Shared configuration and constants for the Chess Engine project.
This file should be used by all team members to ensure consistency.
"""

import chess

# Board representation constants
BOARD_SIZE = 8
NUM_CHANNELS = 12  # 6 piece types × 2 colors (or alternative encoding)

# Piece encoding (for board_encoding.py)
PIECE_TYPES = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}

# Colors
WHITE = chess.WHITE
BLACK = chess.BLACK

# MCTS constants
MCTS_SIMULATIONS = 800  # Number of MCTS simulations per move
C_PUCT = 1.0  # Exploration constant for UCB formula
TEMPERATURE = 1.0  # Temperature for policy sampling

# Neural network constants
POLICY_SIZE = 4672  # Maximum number of legal moves in chess (approximate)
VALUE_OUTPUT_SIZE = 1  # Single value output (win probability)

# Data pipeline constants
REPLAY_BUFFER_SIZE = 10000  # Size of replay buffer
BATCH_SIZE = 32  # Training batch size

# File paths
LOGS_DIR = "logs"
DATA_DIR = "data"
MODELS_DIR = "models"

# Game constants
MAX_GAME_LENGTH = 500  # Maximum moves per game (safety limit)

