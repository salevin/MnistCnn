 -- Created by Sam Levin

mnist = require 'mnist'
require "nn"
require 'optim'
require 'xlua'


 ---- Create Network ----

cnn = nn.Sequential();  -- make a convultional neural net
outputs = 10; epochs=30 -- parameters
-- First conv layer
cnn:add(nn.SpatialConvolution(1, 28, 5, 5))
cnn:add(nn.ReLU())
cnn:add(nn.SpatialMaxPooling(2,2))
-- Second conv layer
cnn:add(nn.SpatialConvolution(28, 56, 5, 5))
cnn:add(nn.ReLU())
cnn:add(nn.SpatialMaxPooling(2,2))
-- Densenly connected mlp
cnn:add(nn.Reshape(56*4*4)) -- Why does it have to be 4x4? why not 7x6? :(
cnn:add(nn.Linear(56*4*4, 1024))
cnn:add(nn.ReLU())
cnn:add(nn.Linear(1024,10))

criterion = nn.CrossEntropyCriterion()
print(cnn)


 ---- Create Identifiers ----

trainset = mnist.traindataset()
testset = mnist.testdataset()
trainData = trainset.data
trainLabel = trainset.label

testData = testset.data
testLabel = testset.label

testSize = testset.size
testInputs = torch.DoubleTensor(testSize, 1, 28, 28) -- or CudaTensor for GPU training

batchSize =  1000 -- trainset.size if i had more ram :( need to set up minibatches
batchInputs = torch.DoubleTensor(batchSize, 1, 28, 28) -- or CudaTensor for GPU training
batchLabels = torch.DoubleTensor(batchSize) -- or CudaTensor for GPU training


 ---- Load Data ----

print("\n---Loading input---\n")
for i = 1, batchSize do
   local input = trainData[i]
   local label = (trainLabel[i] + 1)
   batchInputs[i][1]:copy(input)
   batchLabels[i] = label
end


 ---- Initialize Training Vars ----

params, gradParams = cnn:getParameters()
local optimState = {learningRate = 0.01}


 ---- Start Training ----

print("\n---Training---\n")
for epoch = 1, epochs do
   function feval(params)
      gradParams:zero()

      local outputs = cnn:forward(batchInputs)
      local loss = criterion:forward(outputs, batchLabels)
      local dloss_doutputs = criterion:backward(outputs, batchLabels)
      cnn:backward(batchInputs, dloss_doutputs)

      return loss, gradParams
   end
   optim.adam(feval, params, optimState)
   xlua.progress(epoch,epochs)
end

 ---- Start Testing ----

print("\n---Tests---\n")

err = 0

for i = 1, testSize do
   local input = testData[i]
   testInputs[i][1]:copy(input)
   curr = cnn:forward(testInputs[i])
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


