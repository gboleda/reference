require('nn')
require('nngraph')
require('optim')
require('LinearNB') -- for linear mappings without bias

print('reading the data processing file')
--dofile('data.lua')
dofile('data-max-margin.lua')

-- typical generic gradient update function
function gradUpdate(mlp, x, y, criterion, learningRate)
   local pred = mlp:forward(x)
   local err = criterion:forward(pred, y)
   local gradCriterion = criterion:backward(pred, y)
   mlp:zeroGradParameters()
   mlp:backward(x, gradCriterion)
   mlp:updateParameters(learningRate)
end

function mmarg(t_inp_size,v_inp_size,ref_size)
   local inputs = {}

   -- inputs
   local ling = nn.Identity()()
   table.insert(inputs,ling)
   local target_i = nn.Identity()()
   table.insert(inputs,target_i)
   local confounder_i = nn.Identity()()
   table.insert(inputs,confounder_i)

   -- mappings
   local query = nn.LinearNB(t_inp_size, ref_size)(ling)
   local target_ref = nn.LinearNB(v_inp_size,ref_size)(target_i)
   local confounder_ref = nn.LinearNB(v_inp_size,ref_size)(confounder_i)
   confounder_ref.data.module:share(target_ref.data.module,'weight','bias','gradWeight','gradBias')

   -- reshaping the ref_size-dimensional text vector into 
   -- a 1xref_size-dimensional vector for the multiplication below
   -- NB: setNumInputDims method warns nn that it might get a
   -- two-dimensional object, in which case it has to treat it as a
   -- batch of 1-dimensional objects
   local query_matrix=nn.View(1,-1):setNumInputDims(1)(query)
   local target_matrix=nn.View(1,-1):setNumInputDims(1)(target_ref)
   local confounder_matrix=nn.View(1,-1):setNumInputDims(1)(confounder_ref)

   -- taking the dot product of each reference vector
   -- with the query vector
   local dot_vector_split=nn.MM(false,true)({query_matrix,target_matrix})
   local dot_vector_target=nn.View(-1):setNumInputDims(2)(dot_vector_split) -- reshaping into batch-by-nref matrix for minibatch processing
   dot_vector_split=nn.MM(false,true)({query_matrix,confounder_matrix})
   local dot_vector_confounder=nn.View(-1):setNumInputDims(2)(dot_vector_split) -- reshaping into batch-by-nref matrix for minibatch processing
   
   -- -- dot products
   -- local right = nn.DotProduct()({query, target_ref})
   -- local wrong = nn.DotProduct()({query, confounder_ref})
   
   -- -- wrapping up, here is our model...
   -- return nn.gModule(inputs,{right,wrong}) -- output of the model: dot products
   -- wrapping up, here is our model...
   return nn.gModule(inputs,{dot_vector_target,dot_vector_confounder}) -- output of the model: dot products

end

-- Creates a criterion that measures the loss given an input x = {x1,
-- x2}, a table of two Tensors of size 1 (they contain only scalars),
-- and a label y (1 or -1). In batch mode, x is a table of two Tensors
-- of size batchsize, and y is a Tensor of size batchsize containing 1
-- or -1 for each corresponding pair of elements in the input Tensor.
crit = nn.MarginRankingCriterion(0.1)

local datadir='/Users/gboleda/Desktop/data/reference/exp3-upto5'
local training_set_size=10
local validation_set_size=10
local test_set_size=10
local image_set_size=10
local word_dimensionality=10
local image_dimensionality=5
local word_embedding_file=datadir .. '/toy-word.dm'
local image_embedding_file=datadir .. '/toy-image.dm'
local stimulifile=datadir .. '/toy-stimuli.train'
local normalize_embeddings=0
BUFSIZE = 2^23 -- 1MB

-- REAL DATA PROCESSING
-- reading word embeddings
word_embeddings,t_input_size=
   load_embeddings(word_embedding_file,normalize_embeddings)
--reading image embeddings
image_embeddings,v_input_size=
   load_embeddings(image_embedding_file,normalize_embeddings)

model_input,gold=create_input_structures_from_file_for_max_margin(stimulifile,training_set_size,word_dimensionality,image_dimensionality)

--model_input=

local model=mmarg(word_dimensionality,image_dimensionality,3)

for i = 1,5 do
   print('epoch ' .. tostring(i))
   gradUpdate(model, model_input, gold, crit, 0.01)
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

-- OLD --

-- q=torch.Tensor({-2.2,-0.3,0.0,-0.7,0.6})
-- t=torch.Tensor({-1.4,0.8,1.0,-0.3})
-- c1=torch.Tensor({1.6,0.5,-0.1,0.7})
-- c2=torch.Tensor({-0.7,-0.5,0.1,-1.4})
-- c3=torch.Tensor({-0.3,0.8,0.1,1.7})

-- for i = 1,5 do
--    gradUpdate(model, {q,t,c1}, 1, crit, 0.01)
-- --   gradUpdate(model, {q,t,c2}, 1, crit, 0.01)
-- --   gradUpdate(model, {q,t,c3}, 1, crit, 0.01)
--    if true then
--       -- take forward pass
--       local model_prediction=model:forward({q,t,c1})
--       local loss = crit:forward(model_prediction, -1)
--       print('prediction c1: ')
--       print(model_prediction[1], model_prediction[2])
--       print('loss c1: ')
--       print(loss)
--    end
-- end


