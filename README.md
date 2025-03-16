# Lua Neural Network Implementation

A simple yet powerful neural network implementation in Lua, capable of learning various logical operations including AND, OR, XOR gates. The implementation includes features like batch training, multiple activation functions, and model persistence.

## Features

- **Flexible Architecture**
  - Configurable input layer size
  - Hidden layer with adjustable number of neurons
  - Single output neuron with sigmoid activation

- **Training Capabilities**
  - Batch training support
  - Multiple activation functions (sigmoid, ReLU, tanh)
  - Configurable learning rate
  - Optional bias nodes

- **Model Persistence**
  - Save/load trained models
  - Training history tracking
  - Best model state preservation
  - CSV format training logs

## Usage

### Basic Example

```lua
local NeuralNet = require("NeuralNet")

-- Configure the network
local config = {
    inputSize = 2,
    hiddenSize = 4,
    learningRate = 0.1,
    activation = "sigmoid",
    useBias = true,
    batchSize = 4
}

-- Create and train the network
local nn = NeuralNet.new(config)

-- Training data (example: XOR gate)
local trainingData = {
    {inputs = {0, 0}, target = 0},
    {inputs = {0, 1}, target = 1},
    {inputs = {1, 0}, target = 1},
    {inputs = {1, 1}, target = 0}
}

-- Train the network
nn:trainBatch(trainingData, 100000)

-- Make predictions
print(nn:predict({0, 1}))  -- Should output close to 1
```

### Configuration Options

- `inputSize`: Number of input neurons (default: 2)
- `hiddenSize`: Number of hidden layer neurons (default: 4)
- `learningRate`: Learning rate for training (default: 0.01)
- `activation`: Activation function ["sigmoid", "relu", "tanh"] (default: "sigmoid")
- `useBias`: Whether to use bias nodes (default: true)
- `batchSize`: Size of training batches (default: 1)

### Model Persistence

Models are automatically saved with incrementing numbers:
```lua
nn:saveModel()  -- Saves as "models/model_N.lua"
nn:loadModel(1) -- Loads "models/model_1.lua"
```

### Training History

Training history is automatically saved in CSV format:
- Error per epoch
- Best model performance
- Best predictions achieved

## Project Structure

- `main.lua`: Example usage and training script
- `neuralnet.lua`: Core neural network implementation
- `models/`: Directory for saved models and training history
  - `model_N.lua`: Saved model states
  - `history_N.txt`: Training history logs

## Implementation Details

The neural network uses:
- Forward propagation for predictions
- Backpropagation for training
- Gradient descent optimization
- Best model state tracking
- Batch processing for improved training efficiency

## Training Tips

1. For XOR problems:
   - Use at least 4 hidden neurons
   - Higher learning rate (0.1-0.5)
   - More training epochs (100,000+)

2. For AND/OR problems:
   - Simpler architecture works (2-4 hidden neurons)
   - Lower learning rate (0.01-0.1)
   - Fewer epochs needed (10,000-50,000)

## Error Handling

The implementation includes robust error handling for:
- Model loading/saving
- Invalid configurations
- Training data validation
- File I/O operations

## License

This project is open source and available under the MIT License.
