 -- Created by Sam Levin

mnist = require 'mnist'
require "nn"
require 'optim'
require 'xlua'
require 'cunn'

 ---- Create Network ----

cnn = nn.Sequential();  -- make a convultional neural net
outputs = 10; epochs=20; miniSize=50 -- parameters
-- First conv layer
cnn:add(nn.SpatialConvolution(1, 32, 5, 5, 1, 1, 2))
cnn:add(nn.ReLU())
cnn:add(nn.SpatialMaxPooling(2,2))
-- Second conv layer
cnn:add(nn.SpatialConvolution(32, 64, 5, 5, 1, 1, 2))
cnn:add(nn.ReLU())
cnn:add(nn.SpatialMaxPooling(2,2))
-- Densenly connected mlp
cnn:add(nn.Reshape(64*7*7))
cnn:add(nn.Linear(64*7*7, 1024))
cnn:add(nn.ReLU())
cnn:add(nn.Linear(1024,10))
cnn:add(nn.Dropout())
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

trainSize =  trainset.size
batchInputs = torch.DoubleTensor(trainSize, 1, 28, 28):cuda() -- or CudaTensor for GPU training
batchLabels = torch.DoubleTensor(trainSize):cuda() -- or CudaTensor for GPU training
miniLabels = torch.DoubleTensor(miniSize):cuda()
miniInputs = torch.DoubleTensor(miniSize, 1, 28, 28):cuda()


 ---- Load Data ----

print("\n---Loading input---\n")
for i = 1, trainSize do
   local input = trainData[i]
   local label = (trainLabel[i] + 1)
   batchInputs[i][1]:copy(input)
   batchLabels[i] = label
end

 ---- Initialize Training Vars ----

params, gradParams = cnn:getParameters()
local optimState = {learningRate = 0.0001}


 ---- Record Confusion ----
-- confusion = optim.ConfusionMatrix(ouputs)

 ---- Log results ----
-- trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
-- testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

 ---- Start Training ----

print("\n---Training---\n")
for epoch = 1, epochs do

  print(string.format("\n---Epoch %s / %s---", epoch, epochs))
  avgErr = 0

  for minibatch = 1,trainSize,miniSize do
     miniInputs = batchInputs[{{minibatch,minibatch+miniSize-1}}]
     miniLabels = batchLabels[{{minibatch,minibatch+miniSize-1}}]

     function feval(params)
        collectgarbage()

        gradParams:zero()

        local outputs = cnn:forward(miniInputs)
        local loss = criterion:forward(outputs, miniLabels)
        local dloss_doutputs = criterion:backward(outputs, miniLabels)
        cnn:backward(miniInputs, dloss_doutputs)

        return loss, gradParams
     end
  _, fs = optim.adam(feval, params, optimState)
  xlua.progress((minibatch-1)/miniSize,trainSize/miniSize)
  avgErr = avgErr + fs[1]
  end
  print(string.format("\nAverage cross entropy error was %s", avgErr/(trainSize/miniSize)))
end

 ---- Start Testing ----

print("\n---Tests---\n")

err = 0

p =testSize/20

cnn:evaluate()
print(cnn)

for i = 1, testSize do
   local input = testData[i]
   testInputs[i][1]:copy(input)
   curr = cnn:forward(testInputs[i])
   curr = torch.exp(curr)
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
   if i % p == 0 then
     print(curr)
     print(testLabel[i])
     print(num)
     print(i)
     print("-----------------------------------")
     print()
   end

end


 ---- Ouput Info ----

print("error is: ", (err/testSize)*100, "%")


