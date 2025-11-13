#include <iostream>
#include <cassert>
#include <cmath>
#include <ctime>
#include <algorithm>
#include "../include/mcts.h"
#include "../include/neural_network.h"
#include "../Chess/Chess.h"

#define DEBUG


using namespace std;


/*** MCTS NODE ***/
MCTS_node::MCTS_node(MCTS_node *parent, MCTS_state *state, const MCTS_move *move)
        : parent(parent), state(state), move(move), score(0.0), number_of_simulations(0), size(0),
          nn_value(0.0), has_nn_evaluation(false) {
    children = new vector<MCTS_node *>();
    children->reserve(STARTING_NUMBER_OF_CHILDREN);
    untried_actions = state->actions_to_try();
    terminal = state->is_terminal();
}

MCTS_node::~MCTS_node() {
    delete state;
    delete move;
    for (auto *child : *children) {
        delete child;
    }
    delete children;
    while (!untried_actions->empty()) {
        delete untried_actions->front();    // if a move is here then it is not a part of a child node and needs to be deleted here
        untried_actions->pop();
    }
    delete untried_actions;
}

void MCTS_node::expand(NeuralNetwork* nn) {
    if (is_terminal()) {
        // Terminal node: use actual game result
        if (!has_nn_evaluation) {
            // Get result from terminal state
            Chess_state* chess_state = dynamic_cast<Chess_state*>(state);
            if (chess_state) {
                auto [reason, result] = chess_state->get_board().isGameOver();
                if (result == GameResult::DRAW) {
                    nn_value = 0.5;
                } else if (result == GameResult::WIN) {
                    nn_value = (chess_state->get_board().sideToMove() == Color::WHITE) ? 0.0 : 1.0;
                } else if (result == GameResult::LOSE) {
                    nn_value = (chess_state->get_board().sideToMove() == Color::WHITE) ? 1.0 : 0.0;
                } else {
                    nn_value = 0.5;
                }
            } else {
                nn_value = state->rollout();  // Fallback to rollout
            }
            has_nn_evaluation = true;
            backpropagate(nn_value, 1);
        }
        return;
    } else if (is_fully_expanded()) {
        cerr << "Warning: Cannot expand this node any more!" << endl;
        return;
    }
    
    // If this is a leaf node and we have NN, evaluate it
    if (!has_nn_evaluation && nn != nullptr && children->empty()) {
        Chess_state* chess_state = dynamic_cast<Chess_state*>(state);
        if (chess_state) {
            // Call neural network: (policy, value) = f_θ(s)
            map<string, double> policy_map;
            double value;
            
            if (nn->predict(chess_state->get_board(), policy_map, value)) {
                // Store policy priors and value
                policy_priors = policy_map;
                nn_value = value;
                has_nn_evaluation = true;
                
                // Backpropagate the value immediately
                backpropagate(nn_value, 1);
            } else {
                // NN failed, fallback to rollout
                nn_value = state->rollout();
                has_nn_evaluation = true;
                backpropagate(nn_value, 1);
            }
        } else {
            // Not a chess state, use rollout
            nn_value = state->rollout();
            has_nn_evaluation = true;
            backpropagate(nn_value, 1);
        }
    }
    
    // Expand one child
    if (untried_actions->empty()) {
        return;  // Fully expanded
    }
    
    // Get next untried action
    MCTS_move *next_move = untried_actions->front();
    untried_actions->pop();
    MCTS_state *next_state = state->next_state(next_move);
    
    // Build new MCTS node
    MCTS_node *new_node = new MCTS_node(this, next_state, next_move);
    
    // Set prior probability if we have it
    if (has_nn_evaluation && !policy_priors.empty()) {
        // Convert move to UCI string to look up prior
        Chess_move* chess_move = dynamic_cast<Chess_move*>(next_move);
        if (chess_move) {
            string move_uci = chess_move->sprint();
            if (policy_priors.find(move_uci) != policy_priors.end()) {
                // Prior is already stored in policy_priors map
                // The child node will access it via get_prior()
            }
        }
    }
    
    // If child is terminal, evaluate it immediately
    if (new_node->is_terminal()) {
        Chess_state* child_chess_state = dynamic_cast<Chess_state*>(new_node->state);
        if (child_chess_state) {
            auto [reason, result] = child_chess_state->get_board().isGameOver();
            double child_value = 0.5;
            if (result == GameResult::DRAW) {
                child_value = 0.5;
            } else if (result == GameResult::WIN) {
                child_value = (child_chess_state->get_board().sideToMove() == Color::WHITE) ? 0.0 : 1.0;
            } else if (result == GameResult::LOSE) {
                child_value = (child_chess_state->get_board().sideToMove() == Color::WHITE) ? 1.0 : 0.0;
            }
            new_node->nn_value = child_value;
            new_node->has_nn_evaluation = true;
            new_node->backpropagate(child_value, 1);
        }
    } else if (has_nn_evaluation) {
        // Non-terminal child: will be evaluated when expanded
        // For now, just add it
    }
    
    // Add new node to tree
    children->push_back(new_node);
}

void MCTS_node::rollout() {
#ifdef PARALLEL_ROLLOUTS
    // schedule Jobs
    static JobScheduler scheduler;               // static so that we don't create new threads every time (!)
    double results[NUMBER_OF_THREADS]{-1};
    for (int i = 0 ; i < NUMBER_OF_THREADS ; i++) {
        scheduler.schedule(new RolloutJob(state, &results[i]));
    }
    // wait for all simulations to finish
    scheduler.waitUntilJobsHaveFinished();
    // aggregate results
    double score_sum = 0.0;
    for (int i = 0 ; i < NUMBER_OF_THREADS ; i++) {
        if (results[i] >= 0.0 && results[i] <= 1.0){
            score_sum += results[i];
        } else {    // should not happen
            cerr << "Warning: Invalid result when aggregating parallel rollouts" << endl;
        }
    }
    backpropagate(score_sum, NUMBER_OF_THREADS);
#else
    double w = state->rollout();
    backpropagate(w, 1);
#endif
}

void MCTS_node::backpropagate(double w, int n) {
    score += w;
    number_of_simulations += n;
    if (parent != NULL) {
        parent->size++;
        parent->backpropagate(w, n);
    }
}

bool MCTS_node::is_fully_expanded() const {
    return is_terminal() || untried_actions->empty();
}

bool MCTS_node::is_terminal() const {
    return terminal;
}

unsigned int MCTS_node::get_size() const {
    return size;
}

MCTS_node *MCTS_node::select_best_child(double cpuct) const {
    /** selects best child using PUCT formula:
     * UCB(s,a) = Q(s,a) + cpuct * P(s,a) * sqrt(sum_b N(s,b) + 1) / (1 + N(s,a))
     */
    if (children->empty()) return NULL;
    else if (children->size() == 1) return children->at(0);
    else {
        double puct_score, max = -1;
        MCTS_node *argmax = NULL;
        
        // Calculate sum of visit counts: sum_b N(s,b)
        unsigned int sum_visits = 0;
        for (auto *child : *children) {
            sum_visits += child->number_of_simulations;
        }
        
        for (auto *child : *children) {
            // Q(s,a) = average value (winrate from current player's perspective)
            double Q = 0.0;
            if (child->number_of_simulations > 0) {
                Q = child->score / ((double) child->number_of_simulations);
                // If it's opponent's turn, flip the value
                if (!state->player1_turn()) {
                    Q = 1.0 - Q;
                }
            }
            
            // P(s,a) = prior probability from NN
            double P = 0.0;
            if (child->move != nullptr) {
                P = get_prior(child->move);
            }
            // If no prior found, use uniform (1/num_children)
            if (P == 0.0) {
                P = 1.0 / children->size();
            }
            
            // N(s,a) = visit count for this child
            unsigned int N_a = child->number_of_simulations;
            
            // PUCT formula
            if (cpuct > 0 && number_of_simulations > 0) {
                double sqrt_term = sqrt((double)(sum_visits + 1)) / (1.0 + N_a);
                puct_score = Q + cpuct * P * sqrt_term;
            } else {
                puct_score = Q;  // Pure exploitation
            }
            
            if (puct_score > max) {
                max = puct_score;
                argmax = child;
            }
        }
        return argmax;
    }
}

MCTS_node *MCTS_node::advance_tree(const MCTS_move *m) {
    // Find child with this m and delete all others
    MCTS_node *next = NULL;
    for (auto *child: *children) {
        if (*(child->move) == *(m)) {
            next = child;
        } else {
            delete child;
        }
    }
    // remove children from queue so that they won't be re-deleted by the destructor when this node dies (!)
    this->children->clear();
    // if not found then we have to create a new node
    if (next == NULL) {
        // Note: UCT may lead to not fully explored tree even for short-term children due to terminal nodes being chosen
        cout << "INFO: Didn't find child node. Had to start over." << endl;
        MCTS_state *next_state = state->next_state(m);
        next = new MCTS_node(NULL, next_state, NULL);
    } else {
        next->parent = NULL;     // make parent NULL
        // IMPORTANT: m and next->move can be the same here if we pass the move from select_best_child()
        // (which is what we will typically be doing). If not then it's the caller's responsibility to delete m (!)
    }
    // return the next root
    return next;
}


/*** MCTS TREE ***/
MCTS_node *MCTS_tree::select(double c) {
    MCTS_node *node = root;
    while (!node->is_terminal()) {
        if (!node->is_fully_expanded()) {
            return node;
        } else {
            // Use cpuct_ if NN is available, otherwise use c
            double exploration = (nn_ != nullptr) ? cpuct_ : c;
            node = node->select_best_child(exploration);
        }
    }
    return node;
}

MCTS_tree::MCTS_tree(MCTS_state *starting_state, NeuralNetwork* nn, double cpuct)
    : nn_(nn), cpuct_(cpuct) {
    assert(starting_state != NULL);
    root = new MCTS_node(NULL, starting_state, NULL);
}

MCTS_tree::~MCTS_tree() {
    delete root;
}

double MCTS_node::get_prior(const MCTS_move* move) const {
    // Get prior probability for a move from parent's policy_priors
    if (parent == nullptr || move == nullptr) {
        return 0.0;
    }
    
    // Convert move to UCI string
    Chess_move* chess_move = dynamic_cast<Chess_move*>(const_cast<MCTS_move*>(move));
    if (chess_move) {
        string move_uci = chess_move->sprint();
        auto it = parent->policy_priors.find(move_uci);
        if (it != parent->policy_priors.end()) {
            return it->second;
        }
    }
    return 0.0;
}

void MCTS_tree::grow_tree(int max_iter, double max_time_in_seconds) {
    MCTS_node *node;
    double dt;
    #ifdef DEBUG
    cout << "Growing tree..." << endl;
    #endif
    time_t start_t, now_t;
    time(&start_t);
    for (int i = 0 ; i < max_iter ; i++){
        // select node to expand according to tree policy
        node = select();
        // expand it (this will call NN if available, or perform rollout)
        node->expand(nn_);
        // check if we need to stop
        time(&now_t);
        dt = difftime(now_t, start_t);
        if (dt > max_time_in_seconds) {
            #ifdef DEBUG
            cout << "Early stopping: Made " << (i + 1) << " iterations in " << dt << " seconds." << endl;
            #endif
            break;
        }
    }
    #ifdef DEBUG
    time(&now_t);
    dt = difftime(now_t, start_t);
    cout << "Finished in " << dt << " seconds." << endl;
    #endif
}

unsigned int MCTS_tree::get_size() const {
    return root->get_size();
}

const MCTS_move *MCTS_node::get_move() const {
    return move;
}

const MCTS_state *MCTS_node::get_current_state() const { return state; }

void MCTS_node::print_stats() const {
    #define TOPK 10
    if (number_of_simulations == 0) {
        cout << "Tree not expanded yet" << endl;
        return;
    }
    cout << "___ INFO _______________________" << endl
         << "Tree size: " << size << endl
         << "Number of simulations: " << number_of_simulations << endl
         << "Branching factor at root: " << children->size() << endl
         << "Chances of P1 winning: " << setprecision(4) << 100.0 * (score / number_of_simulations) << "%" << endl;
    // sort children based on winrate of player's turn for this node (!)
    if (state->player1_turn()) {
        std::sort(children->begin(), children->end(), [](const MCTS_node *n1, const MCTS_node *n2){
            return n1->calculate_winrate(true) > n2->calculate_winrate(true);
        });
    } else {
        std::sort(children->begin(), children->end(), [](const MCTS_node *n1, const MCTS_node *n2){
            return n1->calculate_winrate(false) > n2->calculate_winrate(false);
        });
    }
    // print TOPK of them along with their winrates
    cout << "Best moves:" << endl;
    for (int i = 0 ; i < children->size() && i < TOPK ; i++) {
        cout << "  " << i + 1 << ". " << children->at(i)->move->sprint() << "  -->  "
             << setprecision(4) << 100.0 * children->at(i)->calculate_winrate(state->player1_turn()) << "%" << endl;
    }
    cout << "________________________________" << endl;
}

double MCTS_node::calculate_winrate(bool player1turn) const {
    if (player1turn) {
        return score / number_of_simulations;
    } else {
        return 1.0 - score / number_of_simulations;
    }
}

void MCTS_tree::advance_tree(const MCTS_move *move) {
    MCTS_node *old_root = root;
    root = root->advance_tree(move);
    delete old_root;       // this won't delete the new root since we have emptied old_root's children
}

const MCTS_state *MCTS_tree::get_current_state() const { return root->get_current_state(); }

MCTS_node *MCTS_tree::select_best_child() {
    // Use cpuct_ if NN is available, otherwise use 0.0 (pure exploitation)
    double exploration = (nn_ != nullptr) ? cpuct_ : 0.0;
    return root->select_best_child(exploration);
}

void MCTS_tree::print_stats() const { root->print_stats(); }


/*** MCTS agent ***/
MCTS_agent::MCTS_agent(MCTS_state *starting_state, int max_iter, int max_seconds, NeuralNetwork* nn, double cpuct)
: max_iter(max_iter), max_seconds(max_seconds) {
    tree = new MCTS_tree(starting_state, nn, cpuct);
}

const MCTS_move *MCTS_agent::genmove(const MCTS_move *enemy_move) {
    if (enemy_move != NULL) {
        tree->advance_tree(enemy_move);
    }
    // If game ended from opponent move, we can't do anything
    if (tree->get_current_state()->is_terminal()) {
        return NULL;
    }
    #ifdef DEBUG
    cout << "___ DEBUG ______________________" << endl
         << "Growing tree..." << endl;
    #endif
    tree->grow_tree(max_iter, max_seconds);
    #ifdef DEBUG
    cout << "Tree size: " << tree->get_size() << endl
         << "________________________________" << endl;
    #endif
    MCTS_node *best_child = tree->select_best_child();
    if (best_child == NULL) {
        cerr << "Warning: Tree root has no children! Possibly terminal node!" << endl;
        return NULL;
    }
    const MCTS_move *best_move = best_child->get_move();
    tree->advance_tree(best_move);
    return best_move;
}

MCTS_agent::~MCTS_agent() {
    delete tree;
}

const MCTS_state *MCTS_agent::get_current_state() const { return tree->get_current_state(); }