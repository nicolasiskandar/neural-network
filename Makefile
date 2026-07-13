CXX := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -Iinclude -O2

SRC := src/main.cpp src/neuron.cpp
BIN := nn_demo

all: $(BIN)

$(BIN): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(BIN)

clean:
	rm -f $(BIN)