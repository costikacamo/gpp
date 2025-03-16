-- NeuralNet.lua
local NeuralNet = {}

-- Activation Functions
local Activations = {
    sigmoid = {
        func = function(x) return 1 / (1 + math.exp(-x)) end,
        derivative = function(x) return x * (1 - x) end
    },
    relu = {
        func = function(x) return math.max(0, x) end,
        derivative = function(x) return x > 0 and 1 or 0 end
    },
    tanh = {
        func = function(x)
            local exp_x = math.exp(x)
            local exp_neg_x = math.exp(-x)
            return (exp_x - exp_neg_x) / (exp_x + exp_neg_x)
        end,
        derivative = function(x) return 1 - x * x end
    }
}

-- Neural Network Constructor
function NeuralNet.new(config)
    local self = {}

    self.inputSize = config.inputSize or 2
    self.hiddenSize = config.hiddenSize or 4  -- Added hidden layer size
    self.learningRate = config.learningRate or 0.01
    self.activation = Activations[config.activation or "sigmoid"]
    self.useBias = config.useBias or true
    self.batchSize = config.batchSize or 1
    self.totalTrainingEpochs = 0
    
    -- Training history
    self.history = {
        errors = {},
        bestError = math.huge,
        bestEpoch = 0,
        bestPredictions = {}
    }

    -- Initialize Weights & Biases
    -- Input to Hidden weights
    self.weightsIH = {}
    for i = 1, self.hiddenSize do
        self.weightsIH[i] = {}
        for j = 1, self.inputSize do
            self.weightsIH[i][j] = math.random() * 2 - 1
        end
    end
    
    -- Hidden to Output weights
    self.weightsHO = {}
    for i = 1, self.hiddenSize do
        self.weightsHO[i] = math.random() * 2 - 1
    end

    -- Biases
    self.biasH = {}
    self.biasO = math.random() * 2 - 1
    for i = 1, self.hiddenSize do
        self.biasH[i] = math.random() * 2 - 1
    end

    -- Forward Pass
    function self:predict(inputs)
        -- Hidden layer
        local hidden = {}
        for i = 1, self.hiddenSize do
            local sum = self.biasH[i]
            for j = 1, self.inputSize do
                sum = sum + inputs[j] * self.weightsIH[i][j]
            end
            hidden[i] = self.activation.func(sum)
        end

        -- Output layer
        local sum = self.biasO
        for i = 1, self.hiddenSize do
            sum = sum + hidden[i] * self.weightsHO[i]
        end
        return self.activation.func(sum), hidden
    end

    -- Training Function (Batch)
    function self:trainBatch(data, epochs)
        local bestError = math.huge
        local bestWeightsIH = {}
        local bestWeightsHO = {}
        local bestBiasH = {}
        local bestBiasO = self.biasO
        local bestPredictions = {}

        -- Reset history for new training session
        self.history.errors = {}
        
        for epoch = 1, epochs do
            local totalError = 0
            
            for i = 1, #data, self.batchSize do
                local batch = {}
                for j = i, math.min(i + self.batchSize - 1, #data) do
                    table.insert(batch, data[j])
                end

                -- Initialize gradients
                local weightsIHGrad = {}
                local weightsHOGrad = {}
                local biasHGrad = {}
                local biasOGrad = 0

                for i = 1, self.hiddenSize do
                    weightsIHGrad[i] = {}
                    for j = 1, self.inputSize do
                        weightsIHGrad[i][j] = 0
                    end
                    weightsHOGrad[i] = 0
                    biasHGrad[i] = 0
                end

                for _, entry in ipairs(batch) do
                    -- Forward pass
                    local output, hidden = self:predict(entry.inputs)
                    local error = entry.target - output
                    totalError = totalError + error^2

                    -- Output layer gradients
                    local deltaO = error * self.activation.derivative(output)
                    biasOGrad = biasOGrad + deltaO

                    -- Hidden layer gradients
                    local deltaH = {}
                    for i = 1, self.hiddenSize do
                        deltaH[i] = deltaO * self.weightsHO[i] * self.activation.derivative(hidden[i])
                        weightsHOGrad[i] = weightsHOGrad[i] + deltaO * hidden[i]
                        biasHGrad[i] = biasHGrad[i] + deltaH[i]
                        
                        for j = 1, self.inputSize do
                            weightsIHGrad[i][j] = weightsIHGrad[i][j] + deltaH[i] * entry.inputs[j]
                        end
                    end
                end

                -- Update weights and biases
                for i = 1, self.hiddenSize do
                    for j = 1, self.inputSize do
                        self.weightsIH[i][j] = self.weightsIH[i][j] + 
                            (self.learningRate * weightsIHGrad[i][j] / #batch)
                    end
                    self.weightsHO[i] = self.weightsHO[i] + 
                        (self.learningRate * weightsHOGrad[i] / #batch)
                    self.biasH[i] = self.biasH[i] + 
                        (self.learningRate * biasHGrad[i] / #batch)
                end
                self.biasO = self.biasO + (self.learningRate * biasOGrad / #batch)
            end

            -- Track history
            table.insert(self.history.errors, totalError / #data)

            -- Track best model
            if totalError < bestError then
                bestError = totalError
                self.history.bestError = totalError / #data
                self.history.bestEpoch = epoch

                -- Save best predictions
                self.history.bestPredictions = {}
                for _, entry in ipairs(data) do
                    local inputStr = string.format("%d,%d", entry.inputs[1], entry.inputs[2])
                    local prediction = self:predict(entry.inputs)
                    self.history.bestPredictions[inputStr] = prediction
                end

                -- Copy best weights and biases
                bestWeightsIH = {}
                bestWeightsHO = {}
                bestBiasH = {}
                
                for i = 1, self.hiddenSize do
                    bestWeightsIH[i] = {}
                    for j = 1, self.inputSize do
                        bestWeightsIH[i][j] = self.weightsIH[i][j]
                    end
                    bestWeightsHO[i] = self.weightsHO[i]
                    bestBiasH[i] = self.biasH[i]
                end
                bestBiasO = self.biasO
            end

            self.totalTrainingEpochs = self.totalTrainingEpochs + 1
            if epoch % (epochs / 10) == 0 then
                print(string.format("Epoch %d (Total: %d), Error: %.6f, Best Error: %.6f", 
                    epoch, self.totalTrainingEpochs, totalError / #data, self.history.bestError))
            end
        end

        -- Use the best model found during training
        if bestError < math.huge then
            self.weightsIH = bestWeightsIH
            self.weightsHO = bestWeightsHO
            self.biasH = bestBiasH
            self.biasO = bestBiasO
        end

        -- Save training history
        self:saveHistory()
    end

    -- Helper function to get latest model number
    function self:getLatestModelNum()
        local latest = 0
        local handle = io.popen('dir /b "models\\model_*.lua" 2>nul')
        if handle then
            for file in handle:lines() do
                local num = file:match("model_(%d+)%.lua")
                if num then
                    local n = tonumber(num)
                    if n and n > latest then
                        latest = n
                    end
                end
            end
            handle:close()
        end
        return latest
    end

    -- Save training history to file
    function self:saveHistory(filename)
        if not filename then
            filename = string.format("models/history_%d.txt", self:getLatestModelNum())
        end
        
        -- Ensure models directory exists
        os.execute("mkdir models 2>nul")
        
        local file = io.open(filename, "w")
        if not file then
            print("Failed to open history file for writing: " .. filename)
            return false
        end

        -- Write header
        file:write("Epoch,Error,BestError\n")
        
        -- Write data
        for epoch, error in ipairs(self.history.errors) do
            file:write(string.format("%d,%.6f,%.6f\n", 
                epoch, 
                error,
                epoch <= self.history.bestEpoch and self.history.bestError or math.huge
            ))
        end

        -- Write best predictions
        file:write("\nBest Model Predictions:\n")
        for input, prediction in pairs(self.history.bestPredictions) do
            file:write(string.format("%s -> %.6f\n", input, prediction))
        end

        file:close()
        print("Training history saved to " .. filename)
        return true
    end

    -- Save Model to File
    function self:saveModel(filename)
        -- If filename is not provided or is a number, generate model_N format
        if not filename or type(filename) == "number" then
            local num = filename or (self:getLatestModelNum() + 1)
            filename = string.format("model_%d", num)
        end

        -- Ensure models directory exists
        os.execute("mkdir models 2>nul")
        
        -- Add .lua extension if not present
        if not filename:match("%.lua$") then
            filename = filename .. ".lua"
        end
        
        -- Ensure path is in models directory
        if not filename:match("^models[/\\]") then
            filename = "models/" .. filename
        end

        local file = io.open(filename, "w")
        if not file then
            error("Failed to open file for writing: " .. filename)
            return false
        end

        -- Create metadata
        local metadata = {
            timestamp = os.time(),
            config = {
                inputSize = self.inputSize,
                hiddenSize = self.hiddenSize,
                learningRate = self.learningRate,
                activation = self.activation == Activations.sigmoid and "sigmoid" 
                         or self.activation == Activations.relu and "relu"
                         or self.activation == Activations.tanh and "tanh",
                useBias = self.useBias,
                batchSize = self.batchSize,
                totalTrainingEpochs = self.totalTrainingEpochs
            },
            history = {
                bestError = self.history.bestError,
                bestEpoch = self.history.bestEpoch,
                bestPredictions = self.history.bestPredictions
            }
        }

        file:write("return {\n")
        -- Save metadata
        file:write("    metadata = {\n")
        file:write(string.format("        timestamp = %d,\n", metadata.timestamp))
        file:write("        config = {\n")
        file:write(string.format("            inputSize = %d,\n", metadata.config.inputSize))
        file:write(string.format("            hiddenSize = %d,\n", metadata.config.hiddenSize))
        file:write(string.format("            learningRate = %f,\n", metadata.config.learningRate))
        file:write(string.format("            activation = '%s',\n", metadata.config.activation))
        file:write(string.format("            useBias = %s,\n", tostring(metadata.config.useBias)))
        file:write(string.format("            batchSize = %d,\n", metadata.config.batchSize))
        file:write(string.format("            totalTrainingEpochs = %d\n", metadata.config.totalTrainingEpochs))
        file:write("        }\n")
        file:write("    },\n")
        
        -- Save weights
        file:write("    weightsIH = {\n")
        for i = 1, self.hiddenSize do
            file:write("        {")
            for j = 1, self.inputSize do
                file:write(string.format("%.6f", self.weightsIH[i][j]))
                if j < self.inputSize then file:write(", ") end
            end
            file:write("}")
            if i < self.hiddenSize then file:write(",") end
            file:write("\n")
        end
        file:write("    },\n")
        
        file:write("    weightsHO = {")
        for i = 1, self.hiddenSize do
            file:write(string.format("%.6f", self.weightsHO[i]))
            if i < self.hiddenSize then file:write(", ") end
        end
        file:write("},\n")

        -- Save biases
        file:write("    biasH = {")
        for i = 1, self.hiddenSize do
            file:write(string.format("%.6f", self.biasH[i]))
            if i < self.hiddenSize then file:write(", ") end
        end
        file:write("},\n")
        file:write(string.format("    biasO = %.6f\n", self.biasO))
        file:write("}\n")
        file:close()
        print("Model saved successfully to " .. filename)
        return true
    end

    -- Load Model from File
    function self:loadModel(filename)
        -- If filename is a number, convert to model_N format
        if type(filename) == "number" then
            filename = string.format("model_%d", filename)
        end

        -- Add .lua extension if not present
        if not filename:match("%.lua$") then
            filename = filename .. ".lua"
        end
        
        -- Ensure path is in models directory
        if not filename:match("^models[/\\]") then
            filename = "models/" .. filename
        end

        local success, model = pcall(loadfile, filename)
        if not success or not model then
            print("No saved model found at " .. filename)
            return false
        end

        local modelData = model()
        
        -- Validate model data
        if not modelData.weightsIH or not modelData.weightsHO or 
           not modelData.biasH or not modelData.biasO then
            print("Invalid model file format")
            return false
        end

        -- Load and validate metadata if available
        if modelData.metadata then
            local meta = modelData.metadata
            if meta.config.inputSize ~= self.inputSize then
                print(string.format("Warning: Loaded model input size (%d) differs from current config (%d)", 
                    meta.config.inputSize, self.inputSize))
            end
            if meta.config.hiddenSize ~= self.hiddenSize then
                print(string.format("Warning: Loaded model hidden size (%d) differs from current config (%d)", 
                    meta.config.hiddenSize, self.hiddenSize))
            end
            
            -- Restore activation function
            if meta.config.activation then
                self.activation = Activations[meta.config.activation]
                if not self.activation then
                    print(string.format("Warning: Unknown activation function '%s', using sigmoid", meta.config.activation))
                    self.activation = Activations.sigmoid
                end
            end
            
            print(string.format("Loading model trained with %s activation function", meta.config.activation))
            print(string.format("Model was saved on: %s", os.date("%Y-%m-%d %H:%M:%S", meta.timestamp)))
            print(string.format("Total previous training epochs: %d", meta.config.totalTrainingEpochs or 0))
            self.totalTrainingEpochs = meta.config.totalTrainingEpochs or 0

            -- Load history if available
            if meta.history then
                self.history.bestError = meta.history.bestError
                self.history.bestEpoch = meta.history.bestEpoch
                self.history.bestPredictions = meta.history.bestPredictions
                print(string.format("Best error achieved: %.6f at epoch %d", 
                    self.history.bestError, self.history.bestEpoch))
            end
        end

        self.weightsIH = modelData.weightsIH
        self.weightsHO = modelData.weightsHO
        self.biasH = modelData.biasH
        self.biasO = modelData.biasO
        print("Model loaded successfully from " .. filename)
        return true
    end

    return self
end

return NeuralNet