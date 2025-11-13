#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "chess.hpp"
#include <string>
#include <vector>
#include <map>
#include <torch/script.h>

using namespace chess;

/**
 * Neural Network wrapper for AlphaZero model
 * Loads TorchScript model and provides inference interface
 */
class NeuralNetwork {
private:
    torch::jit::script::Module model_;
    bool loaded_;
    
    // Move mapping: policy index -> chess::Move
    // Policy is 4096-dim: from_square * 64 + to_square (no promotions in this encoding)
    std::map<int, Move> policy_to_move_;
    std::map<Move, int> move_to_policy_;
    
    void initialize_move_mapping();
    int move_to_policy_index(const Move& move) const;
    
public:
    NeuralNetwork();
    ~NeuralNetwork();
    
    /**
     * Load TorchScript model from file
     * @param model_path Path to aznet_traced.pt
     * @return true if successful
     */
    bool load_model(const std::string& model_path);
    
    /**
     * Encode chess board to 12-channel tensor
     * @param board Chess board
     * @return Vector of floats (12 * 8 * 8 = 768 elements)
     */
    std::vector<float> encode_board(const Board& board) const;
    
    /**
     * Run inference on board position
     * @param board Chess board
     * @param policy_out Output: policy probabilities for each legal move (normalized), keyed by UCI string
     * @param value_out Output: position value in [0, 1] from white's perspective
     * @return true if successful
     */
    bool predict(const Board& board, 
                 std::map<std::string, double>& policy_out, 
                 double& value_out);
    
    bool is_loaded() const { return loaded_; }
};

#endif // NEURAL_NETWORK_H

