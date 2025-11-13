#include "../include/neural_network.h"
#include "../include/chess.hpp"
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <torch/script.h>
#include <torch/torch.h>

NeuralNetwork::NeuralNetwork() : loaded_(false) {
    initialize_move_mapping();
}

NeuralNetwork::~NeuralNetwork() {
    // torch::jit::script::Module handles its own cleanup
}

void NeuralNetwork::initialize_move_mapping() {
    // Create mapping from policy index to Move
    // Policy is 64*64 = 4096 dimensions (from_square * 64 + to_square)
    // We'll build this mapping on first use
    policy_to_move_.clear();
    move_to_policy_.clear();
}

int NeuralNetwork::move_to_policy_index(const Move& move) const {
    Square from = move.from();
    Square to = move.to();
    
    // Convert square to index (0-63)
    int from_idx = from.index();
    int to_idx = to.index();
    
    // Policy index = from_square * 64 + to_square
    // Note: Promotions are NOT included in the 4096-dim policy
    // They will need to be handled separately or the model needs to be retrained
    int policy_idx = from_idx * 64 + to_idx;
    
    return policy_idx;
}

std::vector<float> NeuralNetwork::encode_board(const Board& board) const {
    // Encode board as 12-channel 8x8 tensor
    // Channels: [white_pawn, white_knight, white_bishop, white_rook, white_queen, white_king,
    //            black_pawn, black_knight, black_bishop, black_rook, black_queen, black_king]
    
    std::vector<float> tensor(12 * 8 * 8, 0.0f);
    
    for (int rank = 0; rank < 8; rank++) {
        for (int file = 0; file < 8; file++) {
            Square sq = Square(Rank(rank), File(file));
            Piece piece = board.at(sq);
            
            if (piece == Piece::NONE) continue;
            
            // Determine channel based on piece type and color
            int channel = 0;
            PieceType pt = piece.type();
            Color color = piece.color();
            
            // Piece type index (0-5: pawn, knight, bishop, rook, queen, king)
            int piece_idx = 0;
            if (pt == PieceType::PAWN) piece_idx = 0;
            else if (pt == PieceType::KNIGHT) piece_idx = 1;
            else if (pt == PieceType::BISHOP) piece_idx = 2;
            else if (pt == PieceType::ROOK) piece_idx = 3;
            else if (pt == PieceType::QUEEN) piece_idx = 4;
            else if (pt == PieceType::KING) piece_idx = 5;
            
            // Channel = color_offset + piece_idx
            // White: channels 0-5, Black: channels 6-11
            channel = (color == Color::WHITE) ? piece_idx : (piece_idx + 6);
            
            // Set value in tensor (rank 7-rank because board is stored with rank 7 at top)
            int tensor_idx = channel * 64 + (7 - rank) * 8 + file;
            tensor[tensor_idx] = 1.0f;
        }
    }
    
    return tensor;
}

bool NeuralNetwork::load_model(const std::string& model_path) {
    try {
        // Load TorchScript model
        model_ = torch::jit::load(model_path);
        model_.eval();  // Set to evaluation mode
        
        // Verify model loaded successfully
        loaded_ = true;
        std::cout << "Successfully loaded model from: " << model_path << std::endl;
        return true;
    }
    catch (const c10::Error& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        loaded_ = false;
        return false;
    }
    catch (const std::exception& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        loaded_ = false;
        return false;
    }
}

bool NeuralNetwork::predict(const Board& board,
                            std::map<std::string, double>& policy_out,
                            double& value_out) {
    if (!loaded_) {
        std::cerr << "Error: Model not loaded" << std::endl;
        return false;
    }
    
    try {
        // 1. Encode board to tensor
        std::vector<float> board_tensor_data = encode_board(board);
        
        // Create torch tensor: shape (1, 12, 8, 8) - batch_size=1, channels=12, height=8, width=8
        torch::Tensor input_tensor = torch::from_blob(
            board_tensor_data.data(), 
            {1, 12, 8, 8}, 
            torch::kFloat32
        ).clone();  // clone to ensure data persists
        
        // 2. Run model inference
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_tensor);
        
        auto outputs = model_.forward(inputs).toTuple();
        torch::Tensor policy_logits = outputs->elements()[0].toTensor();  // Shape: (1, 4096)
        torch::Tensor value_tensor = outputs->elements()[1].toTensor();   // Shape: (1,)
        
        // 3. Extract value (convert from [-1, 1] to [0, 1] for white's perspective)
        value_out = value_tensor.item<float>();
        
        // Convert value from current player's perspective to white's perspective
        if (board.sideToMove() == Color::BLACK) {
            value_out = -value_out;  // Flip for black
        }
        // Now value_out is from white's perspective: 1.0 = white wins, -1.0 = black wins
        // Convert to [0, 1] range: (value + 1.0) / 2.0
        value_out = (value_out + 1.0) / 2.0;
        
        // 4. Extract policy and map to legal moves
        policy_out.clear();
        
        // Get legal moves
        Movelist legal_moves;
        movegen::legalmoves(legal_moves, board);
        
        if (legal_moves.empty()) {
            std::cerr << "Warning: No legal moves in position" << std::endl;
            return false;
        }
        
        // Apply softmax to policy logits
        torch::Tensor policy_probs = torch::softmax(policy_logits, 1);  // Shape: (1, 4096)
        auto policy_accessor = policy_probs.accessor<float, 2>();
        
        // Sum probabilities for legal moves (for normalization)
        double total_prob = 0.0;
        for (const Move& move : legal_moves) {
            int policy_idx = move_to_policy_index(move);
            if (policy_idx >= 0 && policy_idx < 4096) {
                total_prob += policy_accessor[0][policy_idx];
            }
        }
        
        // Normalize and store policy for legal moves only
        // Convert Move to UCI string for map key
        if (total_prob > 0.0) {
            for (const Move& move : legal_moves) {
                int policy_idx = move_to_policy_index(move);
                if (policy_idx >= 0 && policy_idx < 4096) {
                    double prob = policy_accessor[0][policy_idx] / total_prob;
                    // Convert Move to UCI string for map key
                    std::string move_uci = static_cast<std::string>(move.from()) + static_cast<std::string>(move.to());
                    if (move.typeOf() == Move::PROMOTION) {
                        PieceType pt = move.promotionType();
                        char promo = 'q';
                        if (pt == PieceType::KNIGHT) promo = 'n';
                        else if (pt == PieceType::BISHOP) promo = 'b';
                        else if (pt == PieceType::ROOK) promo = 'r';
                        move_uci += promo;
                    }
                    policy_out[move_uci] = prob;
                }
            }
        } else {
            // Fallback: uniform distribution if no valid policy found
            double uniform_prob = 1.0 / legal_moves.size();
            for (const Move& move : legal_moves) {
                std::string move_uci = static_cast<std::string>(move.from()) + static_cast<std::string>(move.to());
                if (move.typeOf() == Move::PROMOTION) {
                    PieceType pt = move.promotionType();
                    char promo = 'q';
                    if (pt == PieceType::KNIGHT) promo = 'n';
                    else if (pt == PieceType::BISHOP) promo = 'b';
                    else if (pt == PieceType::ROOK) promo = 'r';
                    move_uci += promo;
                }
                policy_out[move_uci] = uniform_prob;
            }
        }
        
        return true;
    }
    catch (const c10::Error& e) {
        std::cerr << "Error during inference: " << e.what() << std::endl;
        return false;
    }
    catch (const std::exception& e) {
        std::cerr << "Error during inference: " << e.what() << std::endl;
        return false;
    }
}

