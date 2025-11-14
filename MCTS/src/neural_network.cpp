#include "../include/neural_network.h"
#include "../include/chess.hpp"
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <utility>
#include <iomanip>
#include <ctime>
#include <sstream>
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
    
    // MATCH PYTHON: move_to_index(move) = move.from_square * 64 + move.to_square
    // where from_square/to_square are python-chess indices (0-63)
    // 
    // chess.hpp Square::index() matches python-chess exactly:
    // - A1 = 0, B1 = 1, ..., H1 = 7
    // - A2 = 8, ..., H8 = 63
    // So we can use index() directly, but let's be explicit with rank/file to match Python's row/col
    
    // Python: row = square // 8, col = square % 8
    // chess.hpp: rank = index() >> 3, file = index() & 7
    // So: idx = rank * 8 + file = index() (they're equivalent)
    int from_idx = from.index();  // This matches python-chess square index
    int to_idx = to.index();      // This matches python-chess square index
    
    // Policy index = from_square * 64 + to_square (matches Python exactly)
    // Note: Promotions are NOT included in the 4096-dim policy
    // They will need to be handled separately or the model needs to be retrained
    int policy_idx = from_idx * 64 + to_idx;
    
    return policy_idx;
}

std::vector<float> NeuralNetwork::encode_board(const Board& board) const {
    // Encode board as 12-channel 8x8 tensor
    // Channels: [white_pawn, white_knight, white_bishop, white_rook, white_queen, white_king,
    //            black_pawn, black_knight, black_bishop, black_rook, black_queen, black_king]
    // 
    // IMPORTANT: This must match Python's board_to_matrix() exactly:
    // - Python: row = square // 8, col = square % 8, mat[plane, row, col] = 1
    // - No vertical flipping! rank 0 = rank 1 (white's first rank)
    
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
            
            // MATCH PYTHON: row = rank, col = file (NO FLIPPING!)
            // Python: idx64 = row * 8 + col where row = square // 8, col = square % 8
            // chess.hpp: Square::index() = file + rank * 8, which matches python-chess
            int idx64 = rank * 8 + file;
            int tensor_idx = channel * 64 + idx64;
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
        std::cerr << "ERROR: Neural network model not loaded. Cannot predict." << std::endl;
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
        double raw_value = value_tensor.item<float>();  // Raw value from model (from current player's perspective)
        value_out = raw_value;
        
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
            std::cerr << "ERROR: No legal moves in position. Cannot predict policy." << std::endl;
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
        
        // ========== LOGGING: Policy and Value ==========
        // Open log file (append mode)
        static std::ofstream log_file;
        static bool log_file_opened = false;
        
        if (!log_file_opened) {
            // Create log file with timestamp in name
            std::time_t now = std::time(nullptr);
            std::tm* local_time = std::localtime(&now);
            std::ostringstream filename;
            filename << "nn_inference_" 
                     << std::setfill('0') << std::setw(4) << (1900 + local_time->tm_year)
                     << std::setw(2) << (local_time->tm_mon + 1)
                     << std::setw(2) << local_time->tm_mday << "_"
                     << std::setw(2) << local_time->tm_hour
                     << std::setw(2) << local_time->tm_min
                     << std::setw(2) << local_time->tm_sec << ".log";
            log_file.open(filename.str(), std::ios::app);
            log_file_opened = true;
            if (log_file.is_open()) {
                log_file << "=== Neural Network Inference Log ===" << std::endl;
                log_file << "Started: " << std::asctime(local_time) << std::endl;
                log_file << "=====================================" << std::endl << std::endl;
            }
        }
        
        // Helper lambda to write to both stderr and file
        auto log_line = [&](const std::string& line) {
            std::cerr << line << std::endl;
            if (log_file.is_open()) {
                log_file << line << std::endl;
            }
        };
        
        // Get current FEN for context
        std::string fen = board.getFen();
        
        log_line("\n=== NN Inference Results ===");
        log_line("FEN: " + fen);
        log_line("Side to move: " + std::string(board.sideToMove() == Color::WHITE ? "White" : "Black"));
        
        // Log value
        std::ostringstream value_line1, value_line2;
        value_line1 << "Value (raw from model): " << std::fixed << std::setprecision(4) << raw_value 
                    << " (from " << (board.sideToMove() == Color::WHITE ? "White" : "Black") << "'s perspective)";
        value_line2 << "Value (normalized to [0,1] from White's perspective): " << std::fixed << std::setprecision(4) << value_out;
        log_line(value_line1.str());
        log_line(value_line2.str());
        
        // Log top policy moves
        std::vector<std::pair<std::string, double>> policy_vec(policy_out.begin(), policy_out.end());
        std::sort(policy_vec.begin(), policy_vec.end(), 
                  [](const std::pair<std::string, double>& a, const std::pair<std::string, double>& b) {
                      return a.second > b.second;
                  });
        
        log_line("Top 15 policy moves:");
        int top_n = std::min(15, static_cast<int>(policy_vec.size()));
        for (int i = 0; i < top_n; i++) {
            std::ostringstream move_line;
            move_line << "  " << std::setw(2) << (i+1) << ". " << std::setw(6) << policy_vec[i].first 
                      << " : " << std::fixed << std::setprecision(4) << policy_vec[i].second 
                      << " (" << std::fixed << std::setprecision(2) << (policy_vec[i].second * 100.0) << "%)";
            log_line(move_line.str());
        }
        
        // Log policy statistics
        if (!policy_vec.empty()) {
            double max_prob = policy_vec[0].second;
            double sum_top3 = 0.0;
            for (int i = 0; i < std::min(3, static_cast<int>(policy_vec.size())); i++) {
                sum_top3 += policy_vec[i].second;
            }
            std::ostringstream stats_line;
            stats_line << "Policy stats: max=" << std::fixed << std::setprecision(4) << max_prob 
                      << ", top3_sum=" << std::fixed << std::setprecision(4) << sum_top3
                      << ", legal_moves=" << legal_moves.size();
            log_line(stats_line.str());
        }
        
        log_line("============================");
        
        // Flush file to ensure data is written
        if (log_file.is_open()) {
            log_file.flush();
        }
        
        return true;
    }
    catch (const c10::Error& e) {
        std::cerr << "ERROR: Neural network inference failed (PyTorch error): " << e.what() << std::endl;
        return false;
    }
    catch (const std::exception& e) {
        std::cerr << "ERROR: Neural network inference failed (exception): " << e.what() << std::endl;
        return false;
    }
}

