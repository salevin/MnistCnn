 -- Created by Sam Levin

mnist = require 'mnist'
require "nn"
require 'optim'
require 'xlua'
require 'cunn'

 ---- Create Network ----

cnn = nn.Sequential();  -- make a convultional neural net
outputs = 10; epochs=2000; minibatches=1000 -- parameters
-- First conv layer
cnn:add(nn.SpatialConvolution(1, 28, 5, 5))
cnn:add(nn.ReLU())
cnn:add(nn.SpatialMaxPooling(2,2))
-- Second conv layer
cnn:add(nn.SpatialConvolution(28, 56, 5, 5))
cnn:add(nn.ReLU())
cnn:add(nn.SpatialMaxPooling(2,2))
-- Densenly connected mlp
cnn:add(nn.Reshape(56*4*4)) 
cnn:add(nn.Linear(56*4*4, 1024))
cnn:add(nn.ReLU())
cnn:add(nn.Linear(1024,10))
cnn:add(nn.LogSoftMax())

criterion = nn.CrossEntropyCriterion():cuda()
print(cnn)
--cudnn.convert(cnn)
cnn:cuda()


 ---- Create Identifiers ----

trainset = mnist.traindataset()
testset = mnist.testdataset()
trainData = trainset.data
trainLabel = trainset.label

testData = testset.data
testLabel = testset.label

testSize = testset.size
testInputs = torch.DoubleTensor(testSize, 1, 28, 28):cuda() -- or CudaTensor for GPU training

batchSize =  trainset.size 
batchInputs = torch.DoubleTensor(batchSize, 1, 28, 28):cuda() -- or CudaTensor for GPU training
batchLabels = torch.DoubleTensor(batchSize):cuda() -- or CudaTensor for GPU training
miniLabels = torch.DoubleTensor(minibatches):cuda()
miniInputs = torch.DoubleTensor(minibatches, 1, 28, 28):cuda()


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

  for minibatch = 1,batchSize,minibatches do
     miniInputs = batchInputs[{{minibatch,minibatch+minibatches-1}}]
     miniLabels = batchLabels[{{minibatch,minibatch+minibatches-1}}]

     function feval(params)
        collectgarbage()

        gradParams:zero()

        local outputs = cnn:forward(miniInputs)
        local loss = criterion:forward(outputs, miniLabels)
        local dloss_doutputs = criterion:backward(outputs, miniLabels)
        cnn:backward(miniInputs, dloss_doutputs)

        return loss, gradParams
     end
  end
   optim.adam(feval, params, optimState)
   xlua.progress(epoch,epochs)
end

 ---- Start Testing ----

print("\n---Tests---\n")

err = 0

p =testSize/8

for i = 1, testSize do
   local input = testData[i]
   testInputs[i][1]:copy(input)
   curr = cnn:forward(testInputs[i])
   curr = torch.exp(curr)
   if i % p == 0 then
     print(curr)
     print(testLabel[i])
     print(i)
     print("-----------------------------------")
     print()
   end
   if i % 10000 then
   end
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
   if i % 10 == 0 then
     xlua.progress(i,testSize)
   end
end


 ---- Ouput Info ----
 
print("error is: ", (err/testSize)*100, "%")


