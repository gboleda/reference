-- =================== --
----- PRELIMINARIES -----
-- =================== --

require('nn')
require('nngraph')
require('optim')
require('LinearNB') -- for linear mappings without bias

print('reading the model file')
dofile('models.lua')

print('reading the data processing files')
dofile('data.lua')

-- =================== --
----- FUNCTIONS -----
-- =================== --

-- typical generic gradient update function
function gradUpdate(mlp, x, y, criterion, learningRate)
   local pred = mlp:forward(x)
   local err = criterion:forward(pred, y)
   local gradCriterion = criterion:backward(pred, y)
   mlp:zeroGradParameters()
   mlp:backward(x, gradCriterion)
   mlp:updateParameters(learningRate)
end
   
-- =================== --
----- MAIN -----
-- =================== --

local datadir='/Users/gboleda/Desktop/data/reference/exp5-upto5-tiny-nodeviants'
local training_set_size=40
local validation_set_size=40
local image_set_size=125
local word_embedding_file=datadir .. '/word.dm'
local image_embedding_file=datadir .. '/image.dm'
local stimulifile=datadir .. '/stimuli.train'
local stimulitestfile=datadir .. '/stimuli.test'
local test_set_size=30
local normalize_embeddings=0
local margin=1
local learning_rate=0.01
BUFSIZE = 2^23 -- 1MB


-- Creates a criterion that measures the loss given an input x = {x1,
-- x2}, a table of two Tensors of size 1 (they contain only scalars),
-- and a label y (1 or -1). In batch mode, x is a table of two Tensors
-- of size batchsize, and y is a Tensor of size batchsize containing 1
-- or -1 for each corresponding pair of elements in the input Tensor.
crit = nn.MarginRankingCriterion(margin)

-- REAL DATA PROCESSING
-- reading word embeddings
word_embeddings,t_input_size=
   load_embeddings(word_embedding_file,normalize_embeddings)
print('     word embedding dimensionality: ' ..  tostring(t_input_size))
--reading image embeddings
image_embeddings,v_input_size=
   load_embeddings(image_embedding_file,normalize_embeddings)
print('     image embedding dimensionality: ' ..  tostring(v_input_size))

local model_input,gold,_,idx_list=create_input_structures_from_file_for_max_margin(stimulifile,training_set_size,t_input_size,v_input_size)
local model=max_margin_baseline_model(t_input_size, v_input_size,300)
model_weights, model_weight_gradients = model:getParameters()
print('number of parameters in the model: ' .. model_weights:nElement())

for i = 1,5 do
   print('epoch ' .. tostring(i))
   gradUpdate(model, model_input, gold, crit, learning_rate)
   if true then
      -- take forward pass
      local model_prediction=model:forward(model_input)
      local loss = crit:forward(model_prediction,gold)
      -- print('prediction c1: ')
      -- print(model_prediction[1], model_prediction[2])
      print('loss')
      print(loss)
   end
end


-- now model is trained
-- let's test


skp=0
output_print_file='out'
local d=0
-- print(test_data[2][{{}, {1,5}}])
test_data,test_gold,nimgs_per_sequence,idx_list=create_input_structures_from_file_for_max_margin(stimulitestfile,test_set_size,t_input_size,v_input_size)
-- local acc,l = testmmarg(test_data,test_gold,nimgs_per_sequence,output_print_file,skp)
-- print('loss')
-- print(l)
-- print('acc')
-- print(acc)
print(idx_list)
