#include "MCTS/include/chess.hpp"
#include <iostream>
#include <string>

using namespace std;
using namespace chess;

int main() {
    // Test castling representation
    Board board("r1bqk2r/pppp1ppp/2n2n2/2b1p3/4P3/2NP1N2/PPP1BPPP/R1BQK2R b KQkq - 0 5");
    
    cout << "Testing castling moves:\n";
    
    // Generate all legal moves
    Movelist legal_moves;
    movegen::legalmoves(legal_moves, board);
    
    for (const Move& m : legal_moves) {
        if (m.typeOf() == Move::CASTLING) {
            cout << "Castling move: " << static_cast<string>(m.from()) 
                 << static_cast<string>(m.to()) << endl;
            
            // Test what UCI string this creates
            string uci = static_cast<string>(m.from()) + static_cast<string>(m.to());
            cout << "  UCI string: " << uci << endl;
            
            // Check if this is kingside or queenside
            Square from = m.from();
            Square to = m.to();
            string from_str = static_cast<string>(from);
            string to_str = static_cast<string>(to);
            
            if (from_str == "e8") {
                if (to_str == "h8") {
                    cout << "  This is BLACK KINGSIDE castling (should be e8g8 in UCI)" << endl;
                } else if (to_str == "a8") {
                    cout << "  This is BLACK QUEENSIDE castling (should be e8c8 in UCI)" << endl;
                }
            } else if (from_str == "e1") {
                if (to_str == "h1") {
                    cout << "  This is WHITE KINGSIDE castling (should be e1g1 in UCI)" << endl;
                } else if (to_str == "a1") {
                    cout << "  This is WHITE QUEENSIDE castling (should be e1c1 in UCI)" << endl;
                }
            }
        }
    }
    
    // Test creating castling moves manually
    cout << "\nTesting move creation:\n";
    
    // Try to create castling moves using string squares
    Square e8("e8");
    Square g8("g8");
    Square h8("h8");
    
    // Test if we can find a move from e8 to g8
    bool found_e8g8 = false;
    bool found_e8h8 = false;
    
    for (const Move& m : legal_moves) {
        if (m.from() == e8 && m.to() == g8) {
            found_e8g8 = true;
            cout << "Found e8g8 - Type: " << (m.typeOf() == Move::CASTLING ? "CASTLING" : "NORMAL") << endl;
        }
        if (m.from() == e8 && m.to() == h8) {
            found_e8h8 = true;
            cout << "Found e8h8 - Type: " << (m.typeOf() == Move::CASTLING ? "CASTLING" : "NORMAL") << endl;
        }
    }
    
    if (!found_e8g8) cout << "No e8g8 move found" << endl;
    if (!found_e8h8) cout << "No e8h8 move found" << endl;
    
    return 0;
}