local NeuralNet = require("NeuralNet")

-- Neural Network Configuration
local config = {
    inputSize = 2,
    hiddenSize = 4,
    learningRate = 0.1,
    activation = "sigmoid",
    useBias = true,
    batchSize = 4
}

local nn = NeuralNet.new(config)

-- Training Data (AND Gate)
local trainingData = {
    {inputs = {0, 0}, target = 0},
    {inputs = {0, 1}, target = 1},
    {inputs = {1, 0}, target = 1},
    {inputs = {1, 1}, target = 0}
}

-- Training parameters
local epochs = 100000
local currentModelNum = nn:getLatestModelNum()

-- Try to load the latest model
if currentModelNum > 0 and nn:loadModel(currentModelNum) then
    print(string.format("Continuing training from model_%d...", currentModelNum))
    print("Current best predictions from loaded model:")
    for input, prediction in pairs(nn.history.bestPredictions) do
        print(input .. " ->", string.format("%.6f", prediction))
    end
    print("\nContinuing training...\n")
else
    print("Starting fresh training...")
    currentModelNum = currentModelNum + 1
end

-- Train the network
print(string.format("Training model_%d...", currentModelNum))
nn:trainBatch(trainingData, epochs)

-- Save the improved model
if nn:saveModel(currentModelNum) then
    print(string.format("\nModel saved as model_%d!", currentModelNum))
    print(string.format("Best error achieved: %.6f at epoch %d", nn.history.bestError, nn.history.bestEpoch))
end

-- Test Final Predictions
print("\nFinal Predictions:")
print("0,0 ->", string.format("%.6f", nn:predict({0, 0})))
print("0,1 ->", string.format("%.6f", nn:predict({0, 1})))
print("1,0 ->", string.format("%.6f", nn:predict({1, 0})))
print("1,1 ->", string.format("%.6f", nn:predict({1, 1})))

-- Show Best Predictions
print("\nBest Predictions (from epoch " .. nn.history.bestEpoch .. "):")
for input, prediction in pairs(nn.history.bestPredictions) do
    print(input .. " ->", string.format("%.6f", prediction))
end
