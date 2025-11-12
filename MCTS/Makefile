FLAGS = -O2 -g3 -pedantic -std=c++17 -pthread # -Wall -Wextra

CHESS_EXE = chess_mcts

COMMON_OBJ = JobScheduler.o mcts.o

all: Chess

mcts.o: src/mcts.cpp include/mcts.h include/state.h
	g++ -c $(FLAGS) -I. src/mcts.cpp -o mcts.o

JobScheduler.o: src/JobScheduler.cpp include/JobScheduler.h
	g++ -c $(FLAGS) -I. src/JobScheduler.cpp -o JobScheduler.o

Chess.o: Chess/Chess.cpp Chess/Chess.h include/state.h include/chess.hpp
	g++ -c $(FLAGS) -I. Chess/Chess.cpp -o Chess.o

Chess: mcts.o JobScheduler.o Chess.o Chess/main.cpp
	g++ -o $(CHESS_EXE) $(FLAGS) -I. Chess/main.cpp Chess.o $(COMMON_OBJ)

Example: mcts.o JobScheduler.o Chess.o Chess/example.cpp
	g++ -o chess_example $(FLAGS) -I. Chess/example.cpp Chess.o $(COMMON_OBJ)

clean:
	rm -f *.o $(CHESS_EXE)

.PHONY: all clean Chess

