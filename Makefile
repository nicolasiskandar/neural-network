CXX := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -Iinclude -O2

SRC := src/main_xor.cpp src/neuron.cpp src/layer.cpp
BIN := nn_demo

all: $(BIN)

$(BIN): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(BIN)

clean:
	rm -f $(BIN)