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

MCTS_node *MCTS_node::expand() {
    if (is_terminal() || untried_actions->empty()) {
        return nullptr;
    }

    MCTS_move *next_move = untried_actions->front();
    untried_actions->pop();
    MCTS_state *next_state = state->next_state(next_move);

    MCTS_node *new_node = new MCTS_node(this, next_state, next_move);
    children->push_back(new_node);
    return new_node;
}

double MCTS_node::evaluate(NeuralNetwork* nn) {
    if (has_nn_evaluation) {
        return nn_value;
    }

    double value = 0.5;

    if (is_terminal()) {
        Chess_state* chess_state = dynamic_cast<Chess_state*>(state);
        if (chess_state) {
            auto [reason, result] = chess_state->get_board().isGameOver();
            if (result == GameResult::DRAW) {
                value = 0.5;
            } else if (result == GameResult::WIN) {
                value = (chess_state->get_board().sideToMove() == Color::WHITE) ? 0.0 : 1.0;
            } else if (result == GameResult::LOSE) {
                value = (chess_state->get_board().sideToMove() == Color::WHITE) ? 1.0 : 0.0;
            }
        } else {
            value = state->rollout();
        }
        nn_value = value;
        has_nn_evaluation = true;
        return value;
    }

    if (nn != nullptr) {
        Chess_state* chess_state = dynamic_cast<Chess_state*>(state);
        if (chess_state) {
            map<string, double> policy_map;
            double network_value;
            if (nn->predict(chess_state->get_board(), policy_map, network_value)) {
                policy_priors = policy_map;
                nn_value = network_value;
                has_nn_evaluation = true;
                // Debug: log root node policy (ALWAYS, not just when parent is null, to catch root)
                static bool root_logged = false;
                if (parent == nullptr && !root_logged) {
                    cerr << "[DEBUG] ===== ROOT NODE EVALUATED =====" << endl;
                    cerr << "[DEBUG] Policy map size: " << policy_map.size() << " entries" << endl;
                    for (const auto& [move, prob] : policy_map) {
                        cerr << "[DEBUG]   " << move << ": " << prob << endl;
                    }
                    cerr << "[DEBUG] ================================" << endl;
                    root_logged = true;
                }
                return nn_value;
            } else {
                cerr << "WARNING: Neural network prediction failed for position. Falling back to heuristic rollout." << endl;
            }
        }
    }

    value = state->rollout();
    nn_value = value;
    has_nn_evaluation = true;
    return value;
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
        
        // DEBUG: Log all Q-values and PUCT scores for root node
        bool is_root = (parent == nullptr);
        if (is_root) {
            cerr << "[DEBUG] ===== SELECTING BEST MOVE FROM ROOT =====" << endl;
            cerr << "[DEBUG] Parent visits (N_s): " << number_of_simulations << endl;
            cerr << "[DEBUG] CPUCT: " << cpuct << endl;
            cerr << "[DEBUG] Children count: " << children->size() << endl;
        }
        
        // Store all scores for logging
        vector<pair<string, tuple<double, double, double, double>>> scores; // move, (Q, P, N_a, puct_score)
        
        for (auto *child : *children) {
            // Q(s,a) = average value (winrate from current player's perspective)
            double Q = 0.0;
            if (child->number_of_simulations > 0) {
                Q = child->score / ((double) child->number_of_simulations);
            }
            
            // P(s,a) = prior probability from NN
            double P = 0.0;
            string move_uci = "NULL";
            if (child->move != nullptr) {
                Chess_move* chess_move = dynamic_cast<Chess_move*>(const_cast<MCTS_move*>(child->move));
                if (chess_move) {
                    move_uci = chess_move->sprint();
                }
                P = get_prior(child->move);
            }
            // If no prior found, use uniform (1/num_children)
            double P_original = P;
            if (P == 0.0) {
                P = 1.0 / children->size();
            }
            
            // N(s,a) = visit count for this child
            unsigned int N_a = child->number_of_simulations;
            
            // N(s) = parent's total visits
            double parent_visits = (number_of_simulations > 0) ? number_of_simulations : 1;
            
            puct_score = Q;
            if (cpuct > 0) {
                if (P_original > 0.0) {
                    // PUCT: use prior probability
                    double sqrt_term = sqrt(parent_visits) / (1.0 + N_a);
                    puct_score += cpuct * P_original * sqrt_term;
                } else {
                    // UCT: no prior, use standard UCB exploration
                    double sqrt_term = sqrt(log(parent_visits + 1.0) / (1.0 + N_a));
                    puct_score += cpuct * sqrt_term;
                }
            }
            
            if (is_root) {
                scores.push_back(make_pair(move_uci, make_tuple(Q, P_original, (double)N_a, puct_score)));
            }
            
            if (puct_score > max) {
                max = puct_score;
                argmax = child;
            }
        }
        
        // DEBUG: Print all scores for root node
        if (is_root) {
            // Sort by PUCT score descending
            sort(scores.begin(), scores.end(), [](const auto& a, const auto& b) {
                return get<3>(a.second) > get<3>(b.second);
            });
            cerr << "[DEBUG] Move    |    Q    |    P    |  N_a  | PUCT Score" << endl;
            cerr << "[DEBUG] -------|---------|---------|-------|-----------" << endl;
            for (const auto& [move, vals] : scores) {
                double Q = get<0>(vals);
                double P = get<1>(vals);
                double N_a = get<2>(vals);
                double puct = get<3>(vals);
                cerr << "[DEBUG] " << setw(6) << move << " | " 
                     << setw(7) << fixed << setprecision(4) << Q << " | "
                     << setw(7) << fixed << setprecision(4) << P << " | "
                     << setw(5) << (unsigned int)N_a << " | "
                     << setw(10) << fixed << setprecision(4) << puct;
                if (move == "e2e4") {
                    cerr << " <-- TARGET";
                }
                cerr << endl;
            }
            if (argmax && argmax->move) {
                Chess_move* best_move = dynamic_cast<Chess_move*>(const_cast<MCTS_move*>(argmax->move));
                if (best_move) {
                    cerr << "[DEBUG] Selected: " << best_move->sprint() << " (PUCT=" << max << ")" << endl;
                }
            }
            cerr << "[DEBUG] ============================================" << endl;
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
    // Get prior probability for a move
    // If we're at the root, look in this->policy_priors
    // Otherwise, look in parent->policy_priors
    if (move == nullptr) {
        return 0.0;
    }
    
    // Convert move to UCI string
    Chess_move* chess_move = dynamic_cast<Chess_move*>(const_cast<MCTS_move*>(move));
    if (!chess_move) {
        return 0.0;
    }
    
    string move_uci = chess_move->sprint();
    
    // Determine which policy_priors map to check
    // If parent is nullptr, we're at root - use this->policy_priors
    // Otherwise, use parent->policy_priors (the policy was stored when parent was evaluated)
    const map<string, double>* policy_map = nullptr;
    bool is_root = (parent == nullptr);
    
    if (is_root) {
        policy_map = &policy_priors;  // Root's own policy
    } else {
        policy_map = &(parent->policy_priors);  // Parent's policy (stored when parent was evaluated)
    }
    
    auto it = policy_map->find(move_uci);
    if (it != policy_map->end()) {
        return it->second;
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
        node = select();
        MCTS_node* target = node;
        double value = 0.5;

        if (node->is_terminal()) {
            value = node->evaluate(nn_);
        } else if (!node->is_evaluated()) {
            value = node->evaluate(nn_);
        } else {
            MCTS_node* child = node->expand();
            if (child == nullptr) {
                value = node->evaluate(nn_);
            } else {
                target = child;
                value = child->evaluate(nn_);
            }
        }

        target->backpropagate_value(value);

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
    // Use cpuct_ if NN is available, otherwise use UCT constant (1.41) for exploration
    // This ensures exploration even without NN priors
    double exploration = (nn_ != nullptr) ? cpuct_ : 1.41;
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