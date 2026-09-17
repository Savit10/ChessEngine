#include "MCTS/include/chess.hpp"
#include <iostream>

using namespace std;
using namespace chess;

int main() {
    // Test initial position for castling moves
    Board board;
    
    // Move some pieces to allow castling
    board.makeMove(Move::make(Square::E2, Square::E4));  // e2e4
    board.makeMove(Move::make(Square::E7, Square::E5));  // e7e5
    board.makeMove(Move::make(Square::G1, Square::F3));  // Nf3
    board.makeMove(Move::make(Square::B8, Square::C6));  // Nc6
    board.makeMove(Move::make(Square::F1, Square::E2));  // Be2
    board.makeMove(Move::make(Square::F8, Square::E7));  // Be7
    
    cout << "Current position:\n" << board << endl;
    cout << "FEN: " << board.getFen() << endl;
    
    // Generate all legal moves and look for castling
    Movelist moves;
    movegen::legalmoves(moves, board);
    
    cout << "\nLegal moves that look like castling:\n";
    for (const Move& move : moves) {
        string uci = static_cast<string>(move.from()) + static_cast<string>(move.to());
        
        // Check for potential castling moves
        if (uci == "e1g1" || uci == "e1c1" || uci == "e8g8" || uci == "e8c8") {
            cout << "Move: " << uci << " - Type: ";
            if (move.typeOf() == Move::CASTLING) {
                cout << "CASTLING";
            } else {
                cout << "NORMAL";
            }
            cout << endl;
        }
    }
    
    // Test parsing a castling move
    cout << "\nTesting castling move parsing:\n";
    
    // Try to find kingside castling for white
    Square e1 = Square::E1;
    Square g1 = Square::G1;
    
    for (const Move& move : moves) {
        if (move.from() == e1 && move.to() == g1) {
            cout << "Found e1g1 move: " << static_cast<string>(move.from()) 
                 << static_cast<string>(move.to()) << " - Type: ";
            if (move.typeOf() == Move::CASTLING) {
                cout << "CASTLING";
            } else {
                cout << "NORMAL";
            }
            cout << endl;
        }
    }
    
    return 0;
}