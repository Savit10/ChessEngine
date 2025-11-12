#include "Chess.h"
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <random>
#include <chrono>

using namespace std;
using namespace chess;

// Constructor: Start from initial position
Chess_state::Chess_state() : MCTS_state(), board_() {
    // Board is initialized to starting position by default
}

// Constructor: Start from FEN string
Chess_state::Chess_state(const string& fen) : MCTS_state(), board_(fen) {
}

// Copy constructor
Chess_state::Chess_state(const Chess_state& other) 
    : MCTS_state(other), board_(other.board_) {
    // chess::Board copy constructor handles everything
}

// Check if game is over
bool Chess_state::is_terminal() const {
    auto [reason, result] = board_.isGameOver();
    return result != GameResult::NONE;
}

// Create new state after applying a move
MCTS_state* Chess_state::next_state(const MCTS_move* move) const {
    const Chess_move* m = static_cast<const Chess_move*>(move);
    
    // Create new state by copying current board
    Chess_state* new_state = new Chess_state(*this);
    
    // Apply the move
    new_state->board_.makeMove(m->move);
    
    return new_state;
}

// Get all legal moves
queue<MCTS_move*>* Chess_state::actions_to_try() const {
    queue<MCTS_move*>* Q = new queue<MCTS_move*>();
    
    // Generate all legal moves
    Movelist movelist;
    movegen::legalmoves(movelist, board_);
    
    // Add each move to the queue
    for (const Move& move : movelist) {
        Q->push(new Chess_move(move));
    }
    
    return Q;
}

// Check if it's player 1's turn (white)
bool Chess_state::player1_turn() const {
    return board_.sideToMove() == Color::WHITE;
}

// Random rollout simulation
double Chess_state::rollout() const {
    // If already terminal, return result from white's perspective
    if (is_terminal()) {
        auto [reason, result] = board_.isGameOver();
        if (result == GameResult::DRAW) {
            return 0.5;
        }
        // WIN/LOSE are from side-to-move's perspective
        // We need white's perspective (player1)
        Color white = Color::WHITE;
        Color side_to_move = board_.sideToMove();
        
        if (result == GameResult::WIN) {
            // Side to move won
            return (side_to_move == white) ? 1.0 : 0.0;
        } else if (result == GameResult::LOSE) {
            // Side to move lost
            return (side_to_move == white) ? 0.0 : 1.0;
        }
        return 0.5;
    }
    
    // Create a copy for simulation (we'll modify it)
    Board sim_board = board_;
    
    // Random number generator (seed with current time)
    static mt19937 rng(static_cast<unsigned>(
        chrono::steady_clock::now().time_since_epoch().count()));
    
    Color white = Color::WHITE;
    
    // Simulate random game until terminal
    int max_moves = 500;  // Safety limit to prevent infinite loops
    int move_count = 0;
    
    while (move_count < max_moves) {
        // Check if game is over
        auto [reason, result] = sim_board.isGameOver();
        if (result != GameResult::NONE) {
            // Game ended - convert to white's perspective
            if (result == GameResult::DRAW) {
                return 0.5;
            }
            
            // WIN/LOSE are from current side-to-move's perspective
            // We need to convert to white's perspective
            Color current_side = sim_board.sideToMove();
            
            if (result == GameResult::WIN) {
                // Current side won
                return (current_side == white) ? 1.0 : 0.0;
            } else if (result == GameResult::LOSE) {
                // Current side lost (opponent won)
                return (current_side == white) ? 0.0 : 1.0;
            }
            return 0.5;
        }
        
        // Get legal moves
        Movelist movelist;
        movegen::legalmoves(movelist, sim_board);
        
        if (movelist.empty()) {
            // No legal moves - should be terminal, but handle gracefully
            return 0.5;
        }
        
        // Pick random move
        uniform_int_distribution<size_t> dist(0, movelist.size() - 1);
        size_t random_idx = dist(rng);
        Move random_move = movelist[random_idx];
        
        // Apply move
        sim_board.makeMove(random_move);
        move_count++;
    }
    
    // Reached max moves - evaluate position instead of assuming draw
    // Simple material-based evaluation (can be improved with piece-square tables, etc.)
    return evaluate_position(sim_board);
}

// Evaluate position using simple material count
double Chess_state::evaluate_position(const Board& board) {
    // Piece values (standard chess values)
    const int PAWN_VALUE = 100;
    const int KNIGHT_VALUE = 320;
    const int BISHOP_VALUE = 330;
    const int ROOK_VALUE = 500;
    const int QUEEN_VALUE = 900;
    const int KING_VALUE = 20000;  // Very high to prioritize king safety
    
    int white_material = 0;
    int black_material = 0;
    
    // Count material for each side
    for (int sq = 0; sq < 64; sq++) {
        Square square = static_cast<Square>(sq);
        Piece piece = board.at(square);
        
        if (piece == Piece::NONE) continue;
        
        int value = 0;
        PieceType pt = piece.type();
        
        if (pt == PieceType::PAWN) value = PAWN_VALUE;
        else if (pt == PieceType::KNIGHT) value = KNIGHT_VALUE;
        else if (pt == PieceType::BISHOP) value = BISHOP_VALUE;
        else if (pt == PieceType::ROOK) value = ROOK_VALUE;
        else if (pt == PieceType::QUEEN) value = QUEEN_VALUE;
        else if (pt == PieceType::KING) value = KING_VALUE;
        
        if (piece.color() == Color::WHITE) {
            white_material += value;
        } else {
            black_material += value;
        }
    }
    
    // Convert material difference to [0, 1] range
    // Normalize: if white is ahead by a lot, return close to 1.0
    // If black is ahead, return close to 0.0
    int material_diff = white_material - black_material;
    
    // Normalize to [0, 1] using sigmoid-like function
    // Scale factor: 2000 points difference = ~0.9 or ~0.1
    double normalized = 0.5 + (material_diff / 2000.0);
    
    // Clamp to [0, 1]
    if (normalized > 1.0) normalized = 1.0;
    if (normalized < 0.0) normalized = 0.0;
    
    return normalized;
}

// Print the board
void Chess_state::print() const {
    cout << board_ << endl;
    cout << "FEN: " << board_.getFen() << endl;
    cout << "Side to move: " << (board_.sideToMove() == Color::WHITE ? "White" : "Black") << endl;
    
    auto [reason, result] = board_.isGameOver();
    if (result != GameResult::NONE) {
        cout << "Game over: ";
        if (result == GameResult::WIN) cout << "Win";
        else if (result == GameResult::LOSE) cout << "Loss";
        else if (result == GameResult::DRAW) cout << "Draw";
        cout << " (" << static_cast<int>(reason) << ")" << endl;
    }
}

