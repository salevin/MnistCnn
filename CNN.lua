 -- Created by Sam Levin

mnist = require 'mnist'
require "nn"
require 'optim'
require 'xlua'


 ---- Create MLP ----

mlp = nn.Sequential();  -- make a multi-layer perceptron
epochs=30 -- parameters
mlp:add(nn.SpatialConvultions(32, 1, 5, 5))
mlp:add(nn.SpatialMaxPooling(5,5))
mlp.add(nn.ReLu())
mlp:add(nn.SpatialConvultions(64, 32, 5, 5))
mlp:add(nn.SpatialMaxPooling(5,5))
mlp:add(nn.ReLu())
criterion = nn.CrossEntropyCriterion()
print(mlp)


 ---- Create Identifiers ----

trainset = mnist.traindataset()
testset = mnist.testdataset()
trainData = trainset.data
trainLabel = trainset.label

testData = testset.data
testLabel = testset.label

testSize = testset.size
testInputs = torch.DoubleTensor(testSize, inputs) -- or CudaTensor for GPU training

batchSize = trainset.size
batchInputs = torch.DoubleTensor(batchSize, inputs) -- or CudaTensor for GPU training
batchLabels = torch.DoubleTensor(batchSize) -- or CudaTensor for GPU training


 ---- Load Data ----

print("\n---Loading input---\n")
for i = 1, batchSize do
   local input = trainData[i]:view(inputs)
   local label = (trainLabel[i] + 1)
   batchInputs[i]:copy(input)
   batchLabels[i] = label
   if i % 100 == 0 then
     xlua.progress(i,batchSize)
   end
end


 ---- Initialize Training Vars ----

params, gradParams = mlp:getParameters()
local optimState = {learningRate = 0.01}


 ---- Start Training ----

print("\n---Training---\n")
for epoch = 1, epochs do
   function feval(params)
      gradParams:zero()

      local outputs = mlp:forward(batchInputs)
      local loss = criterion:forward(outputs, batchLabels)
      local dloss_doutputs = criterion:backward(outputs, batchLabels)
      mlp:backward(batchInputs, dloss_doutputs)

      return loss, gradParams
   end
   optim.adam(feval, params, optimState)
   xlua.progress(epoch,epochs)
end

 ---- Start Testing ----

print("\n---Tests---\n")

err = 0

for i = 1, testSize do
   local input = testData[i]:view(inputs)
   testInputs[i]:copy(input)
   curr = mlp:forward(testInputs[i])
   largest = 0
   for i = 1, 10 do
     if curr[i] > largest then
       largest = curr[i]
       num = i - 1
     end
   end
   if num ~= testLabel[i] then
     err = err + 1
   end
   if i % 100 == 0 then
     xlua.progress(i,testSize)
   end
end


 ---- Ouput Info ----
 
print("error is: ", (err/testSize)*10, "%")


