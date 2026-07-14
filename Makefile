CXX := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -Iinclude -O2

LIB_SRC := src/fastLayer.cpp src/neuron.cpp src/layer.cpp

TEST_SRC := src/test_runner.cpp $(LIB_SRC)
TEST_BIN := nn_test

RUN_SRC := src/main_xor.cpp $(LIB_SRC)
RUN_BIN := nn_run

all: $(TEST_BIN) $(RUN_BIN)

$(TEST_BIN): $(TEST_SRC)
	$(CXX) $(CXXFLAGS) $(TEST_SRC) -o $(TEST_BIN)

$(RUN_BIN): $(RUN_SRC)
	$(CXX) $(CXXFLAGS) $(RUN_SRC) -o $(RUN_BIN)

test: $(TEST_BIN)
	./$(TEST_BIN)

run: $(RUN_BIN)
	./$(RUN_BIN)

clean:
	rm -f $(TEST_BIN) $(RUN_BIN)