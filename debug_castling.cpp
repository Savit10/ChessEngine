#include "MCTS/include/chess.hpp"
#include <iostream>
#include <string>

using namespace std;
using namespace chess;

// Simplified version of parse_uci_move for testing
Move parse_uci_move_debug(const Board& board, const string& uci_str) {
    cout << "Parsing move: '" << uci_str << "'" << endl;
    
    if (uci_str.length() < 4 || uci_str.length() > 5) {
        cout << "Invalid length: " << uci_str.length() << endl;
        return Move::NO_MOVE;
    }
    
    try {
        // Parse squares
        Square from = Square(uci_str.substr(0, 2));
        Square to = Square(uci_str.substr(2, 2));
        
        cout << "From: " << static_cast<string>(from) << endl;
        cout << "To: " << static_cast<string>(to) << endl;
        
        // Check if promotion
        if (uci_str.length() == 5) {
            char promo_char = uci_str[4];
            PieceType promo_type = PieceType::QUEEN;
            
            if (promo_char == 'n') promo_type = PieceType::KNIGHT;
            else if (promo_char == 'b') promo_type = PieceType::BISHOP;
            else if (promo_char == 'r') promo_type = PieceType::ROOK;
            
            return Move::make<Move::PROMOTION>(from, to, promo_type);
        }
        
        // Generate all legal moves and check for match
        Movelist legal_moves;
        movegen::legalmoves(legal_moves, board);
        
        cout << "Legal moves count: " << legal_moves.size() << endl;
        cout << "Looking for move from " << static_cast<string>(from) << " to " << static_cast<string>(to) << endl;
        
        for (const Move& m : legal_moves) {
            if (m.from() == from && m.to() == to) {
                cout << "Found matching move! Type: ";
                if (m.typeOf() == Move::CASTLING) {
                    cout << "CASTLING";
                } else if (m.typeOf() == Move::NORMAL) {
                    cout << "NORMAL";
                } else if (m.typeOf() == Move::PROMOTION) {
                    cout << "PROMOTION";
                } else if (m.typeOf() == Move::ENPASSANT) {
                    cout << "ENPASSANT";
                }
                cout << endl;
                return m;
            }
        }
        
        cout << "No matching move found!" << endl;
        cout << "Available moves:" << endl;
        for (const Move& m : legal_moves) {
            string move_str = static_cast<string>(m.from()) + static_cast<string>(m.to());
            cout << "  " << move_str;
            if (m.typeOf() == Move::CASTLING) cout << " (CASTLING)";
            if (m.typeOf() == Move::PROMOTION) cout << " (PROMOTION)";
            cout << endl;
        }
        
        return Move::NO_MOVE;
    } catch (const exception& e) {
        cout << "Exception: " << e.what() << endl;
        return Move::NO_MOVE;
    } catch (...) {
        cout << "Unknown exception" << endl;
        return Move::NO_MOVE;
    }
}

int main() {
    // Create a position where black can castle kingside
    Board board("r1bqk2r/pppp1ppp/2n2n2/2b1p3/4P3/2NP1N2/PPP1BPPP/R1BQK2R b KQkq - 0 5");
    
    cout << "Current position:\n" << board << endl;
    cout << "FEN: " << board.getFen() << endl;
    
    // Test castling move
    Move move = parse_uci_move_debug(board, "e8g8");
    
    if (move == Move::NO_MOVE) {
        cout << "Failed to parse castling move!" << endl;
    } else {
        cout << "Successfully parsed castling move!" << endl;
    }
    
    return 0;
}