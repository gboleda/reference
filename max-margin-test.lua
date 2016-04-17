-- =================== --
----- PRELIMINARIES -----
-- =================== --

require('nn')
require('nngraph')
require('optim')
require('LinearNB') -- for linear mappings without bias

print('reading the data processing files')
dofile('data.lua')
dofile('data-max-margin.lua')

-- =================== --
----- FUNCTIONS -----
-- =================== --

-- typical generic gradient update function
function gradUpdate(mlp, x, y, criterion, learningRate)
   local pred = mlp:forward(x)
--   print(pred)
   local err = criterion:forward(pred, y)
   local gradCriterion = criterion:backward(pred, y)
   mlp:zeroGradParameters()
   mlp:backward(x, gradCriterion)
   mlp:updateParameters(learningRate)
--   return err
end
   
function mmarg(t_inp_size,v_inp_size,ref_size)
   
   -- BEGIN DEBUG --
   -- -- inputs
   -- local query = nn.Identity()()
   -- local target_ref = nn.Identity()()
   -- local confounder_ref = nn.Identity()()

   -- local inputs = {query, target_ref, confounder_ref}
   
   -- wrapping up, here is our model...
--   return nn.gModule(inputs,{dot_vector_target,dot_vector_confounder}) -- output of the model: table of two dot products
--   return nn.gModule(inputs,{dot_vector_split1,dot_vector_split2})
--   return nn.gModule(inputs,{query,target_ref,confounder_ref})
--   return nn.gModule(inputs,{query_matrix,target_matrix,confounder_matrix})

   -- END DEBUG --

   -- BEGIN AS IT WAS --   
   
   -- inputs
   local ling = nn.Identity()()
   local target_i = nn.Identity()()
   local confounder_i = nn.Identity()()

   local inputs = {ling, target_i, confounder_i}
   
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
   local query_matrix=nn.View(1,-1):setNumInputDims(1)(query);
   local target_matrix=nn.View(1,-1):setNumInputDims(1)(target_ref)
   local confounder_matrix=nn.View(1,-1):setNumInputDims(1)(confounder_ref)

   -- taking the dot product of each reference vector
   -- with the query vector
   local dot_vector_split1=nn.MM(false,true)({query_matrix,target_matrix})
   local dot_vector_target=nn.View(-1)(dot_vector_split1) -- reshaping into batchsize vector for minibatch processing
   local dot_vector_split2=nn.MM(false,true)({query_matrix,confounder_matrix})
   local dot_vector_confounder=nn.View(-1)(dot_vector_split2) -- reshaping into batchsize vector for minibatch processing
   
   -- wrapping up, here is our model...
   return nn.gModule(inputs,{dot_vector_target,dot_vector_confounder}) -- output of the model: table of two dot products
   -- END AS IT WAS --

end

-- =================== --
----- MAIN -----
-- =================== --

-- Creates a criterion that measures the loss given an input x = {x1,
-- x2}, a table of two Tensors of size 1 (they contain only scalars),
-- and a label y (1 or -1). In batch mode, x is a table of two Tensors
-- of size batchsize, and y is a Tensor of size batchsize containing 1
-- or -1 for each corresponding pair of elements in the input Tensor.
crit = nn.MarginRankingCriterion(0.1)

local datadir='/Users/gboleda/Desktop/data/reference/exp5-upto5-tiny-nodeviants'
local training_set_size=40
local validation_set_size=40
local image_set_size=125
local word_embedding_file=datadir .. '/word.dm'
local image_embedding_file=datadir .. '/image.dm'
local stimulifile=datadir .. '/stimuli.train'
local stimulitestfile=datadir .. '/stimuli-tiny.test'
local test_set_size=5
local normalize_embeddings=0
BUFSIZE = 2^23 -- 1MB

-- REAL DATA PROCESSING
-- reading word embeddings
word_embeddings,t_input_size=
   load_embeddings(word_embedding_file,normalize_embeddings)
print('     word embedding dimensionality: ' ..  tostring(t_input_size))
--reading image embeddings
image_embeddings,v_input_size=
   load_embeddings(image_embedding_file,normalize_embeddings)
print('     image embedding dimensionality: ' ..  tostring(v_input_size))

local model_input,gold,_=create_input_structures_from_file_for_max_margin(stimulifile,training_set_size,t_input_size,v_input_size)
local model=mmarg(t_input_size, v_input_size,300)

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


-- now model is trained
-- let's test

function testmmarg(test_input_table,tgold,nimgs,output_print_file,skip_test_loss)

   set_size=nimgs:size()[1]
   -- passing all test samples through the trained network
   local model_prediction=nil
   model_prediction=model:forward(test_input_table)
   print(model_prediction)
   print(model_prediction[1], model_prediction[2])
   qt=model_prediction[1]
   qc=model_prediction[2]
   -- print(tostring(test_input_table[1]:size()[1]))
   -- print(tostring(nimgs:size()[1]))
   
   local sequence_length=0
   local end_at=0
   local right=0 -- torch.Tensor(1)
   -- to compute accuracy, we first get the answers for each sequence
   for i=1,set_size do
      print(tostring('---'))
      print(tostring(i))
      local start_at=end_at+1 -- next time we start after we left off
      end_at=start_at+(nimgs[i]-2) -- nimgs[i] is sequence length; -2 to discount: the target (-1), the +1 that we put in "start at" (-1)
      print(tostring(start_at))
      print(tostring(end_at))
      qt_sequence=qt[{{start_at,end_at}}]
      qc_sequence=qc[{{start_at,end_at}}]
      -- print(qt_sequence)
      -- print(qc_sequence)
      max_confounder=torch.max(qc_sequence)
      print('target:' .. tostring(qt_sequence))
      print('confounder:' .. tostring(qc_sequence))
      print('max target - max confounder: ' .. tostring(qt_sequence[1]) .. ';' .. tostring(max_confounder))
      if qt_sequence[1] > max_confounder then print('right!'); right=right+1 end
      i=i+1
   end
   -- for j=1,test_input_table[1]:size()[1] do
   --    print('target-confounder: ' .. tostring(qt[j]) .. ';' .. tostring(qc[j]))
   -- end


   -- print(#nimgs)
   -- print(tostring(end_at))
   -- local model_max_log_probs,model_guesses=torch.max(model_prediction,2)
   -- local model_max_probs=torch.exp(model_max_log_probs)
   -- -- we then count how often this guesses are the same as the gold
   -- -- (and thus the difference is 0) (note conversions to long because
   -- -- model_guesses is long tensor)
   -- local hit_count = torch.sum(torch.eq(test_index_list:long(),model_guesses))
   -- -- normalizing accuracy by test set size
   -- local accuracy=hit_count/test_word_query_list:size(1)

   -- --if requested, print guesses and their log probs to file
   -- if output_print_file then
   --       local f = io.open(output_print_file,"w")
   -- 	 for i=1,model_max_probs:size(1) do
   -- 	    f:write(model_guesses[i][1]," ",model_max_probs[i][1],"\n")
   -- 	 end
   -- 	 f:flush()
   -- 	 f.close()
   -- end
   -- return average_loss,accuracy
end -- function test

skp=0
output_print_file='out'
local d=0
test_data,test_gold,nimgs_per_sequence=create_input_structures_from_file_for_max_margin(stimulitestfile,test_set_size,t_input_size,v_input_size)

testmmarg(test_data,test_gold,nimgs_per_sequence,output_print_file,skp)

-- print(test_data[2][{{}, {1,5}}])



-- TOY TESTING WITH A MODEL FUNCTION --

-- local t_input_size=2 -- now this is the reference
-- local v_input_size=2
-- local batchsize=2

-- q=torch.Tensor({1,3})
-- t=torch.Tensor({1,2})
-- c1=torch.Tensor({2,3})
-- c2=torch.Tensor({2,5})

-- queries=torch.Tensor(batchsize,t_input_size)
-- targets=torch.Tensor(batchsize,v_input_size)
-- confounders=torch.Tensor(batchsize,v_input_size)
-- for i = 1, batchsize do -- The range includes both ends.
--    queries[i]=q
--    targets[i]=t
-- end
-- confounders[1]=c1
-- confounders[2]=c2

-- batchinputs ={queries,targets,confounders}
-- -- print(batchinputs)
-- -- for i=1,3 do print(batchinputs[i]:dim());print(batchinputs[i]) end
-- gold=torch.Tensor(batchsize):zero()+1

-- local model=mmarg(t_input_size, v_input_size,2)
-- local model_prediction=model:forward(batchinputs)
-- print(model_prediction)
-- print(model_prediction[1])
-- print(model_prediction[2])
-- for i=1,#model_prediction do print(model_prediction[i]) end

-- for i = 1,5 do
--    print('epoch ' .. tostring(i))
--    gradUpdate(model, batchinputs, gold, crit, 0.01)
--    if true then
--       -- take forward pass
--       local model_prediction=model:forward(batchinputs)
--       local loss = crit:forward(model_prediction,gold)
--       -- print('prediction c1: ')
--       -- print(model_prediction[1], model_prediction[2])
--       print('loss')
--       print(loss)
--    end
-- end


-- TOY TESTING WITHOUT A MODEL FUNCTION --

-- -- building {[q,q,q],[t,t,t],[c1,c2,c3]}
-- -- queries=torch.Tensor(3,t_input_size)
-- queries=torch.Tensor(3,v_input_size) -- DUMMY -- tomake sizes match
-- targets=torch.Tensor(3,v_input_size)
-- confounders=torch.Tensor(3,v_input_size)
-- for i = 1, 3 do -- The range includes both ends.
--    --   queries[i]=q
--    queries[i]=t
--    targets[i]=t
-- end
-- confounders[1]=c1
-- confounders[2]=c2
-- confounders[3]=c3

-- rshp = nn.View(1,-1):setNumInputDims(1)
-- dotmat1=nn.MM(false,true)
-- dotmat2=nn.MM(false,true)
-- dotrshp=nn.View(-1):setNumInputDims(2) -- reshaping into batch-by-nref matrix for minibatch processing

-- qrshp=rshp:forward(queries);
-- trshp=rshp:forward(targets); --print(trshp)
-- crshp=rshp:forward(confounders); --print(crshp)
-- -- print(qrshp)
-- -- print(trshp)
-- -- print(crshp)

-- tqdot=dotmat1:forward({qrshp,trshp})
-- tqdotr=dotrshp:forward(tqdot)
-- print(tqdot)
-- print(tqdotr)
-- cqdot=dotmat2:forward({qrshp,crshp})
-- cqdotr=dotrshp:forward(cqdot)
-- print(cqdot)
-- print(cqdotr)

-- crit=nn.MarginRankingCriterion(0.1)

-- -- it works with single items
-- local s1=torch.rand(1)
-- local s2=torch.rand(1)
-- local x={s1,s2}
-- local y=1
-- single_input=crit:forward(x, y)
-- print("single: " .. single_input)

-- -- but it fails with tensors (batches) as input
-- local b1=torch.rand(3)
-- local b2=torch.rand(3)
-- x={b1,b2}
-- y=torch.Tensor(3):zero()+1
-- batch_input=crit:forward(x, y)
-- print("batch: " .. batch_input)
