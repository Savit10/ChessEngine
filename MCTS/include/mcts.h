#ifndef MCTS_H
#define MCTS_H

#include "state.h"
#include "JobScheduler.h"
#include <vector>
#include <queue>
#include <iomanip>
#include <map>
#include <string>

// Forward declaration
class NeuralNetwork;


#define STARTING_NUMBER_OF_CHILDREN 32   // expected number so that we can preallocate this many pointers
#define PARALLEL_ROLLOUTS                // whether or not to do multiple parallel rollouts


using namespace std;

/** Ideas for improvements:
 * - state should probably be const like move is (currently problematic because of Quoridor's example)
 * - Instead of a FIFO Queue use a Priority Queue with priority on most probable (better) actions to be explored first
  or maybe this should just be an iterable and we let the implementation decide but these have no superclasses in C++ it seems
 * - vectors, queues and these structures allocate data on the heap anyway so there is little point in using the heap for them
 * so use stack instead?
 */


class MCTS_node {
    bool terminal;
    unsigned int size;
    unsigned int number_of_simulations;
    double score;                       // e.g. number of wins (could be int but double is more general if we use evaluation functions)
    MCTS_state *state;                  // current state
    const MCTS_move *move;              // move to get here from parent node's state
    vector<MCTS_node *> *children;
    MCTS_node *parent;
    queue<MCTS_move *> *untried_actions;
    map<string, double> policy_priors;  // Prior probabilities from NN (keyed by move UCI string)
    double nn_value;                    // Value from NN (initialized when node is expanded)
    bool has_nn_evaluation;            // Whether this node has been evaluated by NN
    void backpropagate(double w, int n);
public:
    MCTS_node(MCTS_node *parent, MCTS_state *state, const MCTS_move *move);
    ~MCTS_node();
    bool is_fully_expanded() const;
    bool is_terminal() const;
    const MCTS_move *get_move() const;
    unsigned int get_size() const;
    void expand(NeuralNetwork* nn = nullptr);  // Pass NN for evaluation
    void rollout();  // Keep for backward compatibility, but won't be used with NN
    MCTS_node *select_best_child(double cpuct) const;  // cpuct is the exploration constant
    MCTS_node *advance_tree(const MCTS_move *m);
    const MCTS_state *get_current_state() const;
    void print_stats() const;
    double calculate_winrate(bool player1turn) const;
    double get_prior(const MCTS_move* move) const;  // Get prior probability for a move
};



class MCTS_tree {
    MCTS_node *root;
    NeuralNetwork* nn_;  // Neural network for evaluation
    double cpuct_;       // PUCT exploration constant
public:
    MCTS_tree(MCTS_state *starting_state, NeuralNetwork* nn = nullptr, double cpuct = 1.0);
    ~MCTS_tree();
    MCTS_node *select(double c=1.41);        // select child node to expand according to tree policy (PUCT)
    MCTS_node *select_best_child();          // select the most promising child of the root node
    void grow_tree(int max_iter, double max_time_in_seconds);
    void advance_tree(const MCTS_move *move);      // if the move is applicable advance the tree, else start over
    unsigned int get_size() const;
    const MCTS_state *get_current_state() const;
    void print_stats() const;
    void set_neural_network(NeuralNetwork* nn) { nn_ = nn; }
    void set_cpuct(double cpuct) { cpuct_ = cpuct; }
};


class MCTS_agent {                           // example of an agent based on the MCTS_tree. One can also use the tree directly.
    MCTS_tree *tree;
    int max_iter, max_seconds;
public:
    MCTS_agent(MCTS_state *starting_state, int max_iter = 100000, int max_seconds = 30, 
               NeuralNetwork* nn = nullptr, double cpuct = 1.0);
    ~MCTS_agent();
    const MCTS_move *genmove(const MCTS_move *enemy_move);
    const MCTS_state *get_current_state() const;
    void feedback() const { tree->print_stats(); }
};


class RolloutJob : public Job {             // class for performing parallel simulations using a thread pool
    double *score;
    const MCTS_state *state;
public:
    RolloutJob(const MCTS_state *state, double *score) : Job(), state(state), score(score) {}
    void run() override {
        // put the result in the memory specified at construction time as this Job will be deleted when done (can't store it here)
        *score = state->rollout();
    }
};


#endif