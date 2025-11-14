#ifndef MCTS_CHESS_H
#define MCTS_CHESS_H

#include "../include/state.h"
#include "../include/chess.hpp"
#include <queue>

using namespace std;
using namespace chess;

/**
 * Chess_move - Wraps chess::Move for MCTS
 */
struct Chess_move : public MCTS_move {
    Move move;
    
    Chess_move(const Move& m) : move(m) {}
    Chess_move(const Chess_move& other) : move(other.move) {}
    
    bool operator==(const MCTS_move& other) const override {
        const Chess_move& o = static_cast<const Chess_move&>(other);
        return move == o.move;
    }
    
    string sprint() const override {
        // Use the same format as neural_network.cpp to ensure exact string matching
        // static_cast<string>(Square) produces "e2", "e4", etc.
        string result = static_cast<string>(move.from()) + static_cast<string>(move.to());
    
        if (move.typeOf() == Move::PROMOTION) {
            PieceType pt = move.promotionType();
            char promo = 'q';
            if (pt == PieceType::KNIGHT) promo = 'n';
            else if (pt == PieceType::BISHOP) promo = 'b';
            else if (pt == PieceType::ROOK) promo = 'r';
            result += promo;
        }
    
        return result;
    }
    
};

/**
 * Chess_state - Implements MCTS_state for chess using chess::Board
 */
class Chess_state : public MCTS_state {
private:
    Board board_;
    
    // Helper: Evaluate position from white's perspective
    // Returns: 1.0 (white winning) to 0.0 (black winning), 0.5 (equal)
    static double evaluate_position(const Board& board);
    
public:
    // Constructors
    Chess_state();
    Chess_state(const string& fen);
    Chess_state(const Chess_state& other);
    
    // MCTS_state interface implementation
    bool is_terminal() const override;
    MCTS_state* next_state(const MCTS_move* move) const override;
    queue<MCTS_move*>* actions_to_try() const override;
    double rollout() const override;
    bool player1_turn() const override;
    void print() const override;
    
    // Access to underlying board (useful for debugging)
    const Board& get_board() const { return board_; }
    Board& get_board() { return board_; }
};

#endif // MCTS_CHESS_H

