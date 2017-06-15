-- preambles

require('nn')
require('nngraph')
require('optim')
require('../../LinearNB') -- for linear mappings without bias
require('../../Peek') 
require('../Broadcast')
require('../Normalization')

-- making sure random is random!
math.randomseed(os.time())


-- ******* options *******

cmd = torch.CmdLine()

-- run on GPU? no by default
cmd:option('--use_cuda',0,'is a GPU available? default: nope, specify value different from 0 to use GPU')

-- options concerning input processing 
cmd:option('--protocol_prefix','','prefix for protocol files. Expects files PREFIX.(train|valid) to be in the folder where program is called. Format: one trial per line: first field linguistic query, second field index of the first occurrence of query in the object token sequence (see next), rest of the fields are token sequence (query format: object:att1:att2; object token format: att:object)')
cmd:option('--word_embedding_file','','word embedding file (with word vectors; first field word, rest of the fields vector values)')
cmd:option('--image_embedding_file','','image embedding file (with visual vectors; first field word and image, rest of the fields vector values)')
cmd:option('--normalize_embeddings',0, 'whether to normalize word and image representations, set to 1 to normalize')
cmd:option('--input_sequence_cardinality', 0, 'number of object tokens (exposures) in a sequence')
cmd:option('--training_set_size',0, 'training set size')
cmd:option('--validation_set_size',0, 'validation set size')
-- cmd:option('--test_set_size',0, 'test set size')

-- options concerning output processing
cmd:option('--save_model_to_file','', 'if a string is passed, after training has finished, the trained model is saved as binary file named like the string')
cmd:option('--output_debug_prefix','','if this prefix is defined, at the end of each epoch, we print to one or more files with this prefix (and various suffixes) information that might vary depending on debugging needs (see directly code of this program to check out what it is currently being generated for debugging purposes, if anything)')
-- output files
cmd:option('--output_guesses_file','','if this file is defined, we print to it, as separated space-delimited columns, the index the model returned as its guess for each test item, and the corresponding log probability')

-- model parameters
cmd:option('--model','ff','name of model to be used (currently supported: ff (default), rnn, mm_one_matrix, mm_standard, entity_prediction_image_att_shared,entity_prediction_image_att_shared_neprob)')
cmd:option('--multimodal_size',300, 'size of multimodal vectors')
cmd:option('--dropout_prob',0,'probability of each parameter being dropped, i.e having its commensurate output element be zero; default: equivalent to no dropout; recommended value in torch documentation: 0.5')
cmd:option('--attribute_dropout_prob',0,'probability of each attribute parameter being dropped, i.e having its commensurate output element be zero, for models that allow a different dropout probability for attributes to avoid the perfect matching problem; default: equivalent to no dropout')
---- entity_prediction parameters
cmd:option('--new_mass_aggregation_method','sum','when computing the new entity mass cell, use as input sum (default), max or mean of values in similarity profile')
cmd:option('--new_cell_nonlinearity','none','nonlinear transformation of mapping to predict new cell (options: none (default), sigmoid, relu, tanh)')
cmd:option('--temperature',1,'before transforming the vector of dot products of the query with the object tokens into a softmax, multiply by temperature: the larger the temperature, the more skewed the probability distribution produced by the softmax (default: no rescaling)')
---- ff and rnn parameters
cmd:option('--hidden_size',300, 'size of hidden layer')
cmd:option('--hidden_count',1,'number of hidden layers')
cmd:option('--ff_nonlinearity','none','nonlinear transformation of hidden layers (options: none (default), sigmoid, relu, tanh)') -- NB: despite the name, also used by rnn
cmd:option('--summary_size',300, 'size of history vector of RNN')
cmd:option('--nhops',1, 'number of hops for memory network models')

-- training parameters
-- optimization method: sgd or adam
cmd:option('--optimization_method','sgd','sgd by default, with adam as alternative')
-- optimization hyperparameters (copying defaults from
-- https://github.com/torch/demos/blob/master/logistic-regression/example-logistic-regression.lua)
-- NB: only first will be used for adam
cmd:option('--learning_rate',1e-3, 'learning rate')
cmd:option('--weight_decay',0, 'weight decay')
cmd:option('--momentum',0, 'momentum')
cmd:option('--learning_rate_decay',1e-4, 'learning rate decay')
-- other training parameters
-- magic gradient clipping value copied from Karpathy's char-rnn code
cmd:option('--grad_clip',5, 'value to clip gradients at')
-- minimum number of epochs, that will be executed no matter what
-- even if validation loss doesn't decrease
cmd:option('--min_epochs',1, 'min number of epochs')
-- maximum number of epochs, after which training stops, even if
-- validation loss is still decreasing
cmd:option('--max_epochs',100, 'max number of epochs')
-- number of adjacent epochs in which the validation loss is allowed
-- to increase/be stable, before training is stopped
cmd:option('--max_validation_lull',2,'number of adjacent non-improving epochs before stop')
-- size of a mini-batch
cmd:option('--mini_batch_size',10,'mini batch size')

-- -- testing parameters
-- cmd:option('--test_mode',0, 'if set to 1, the test set will be evaluated; requires a binary of a model produced in training mode (the default mode)')
-- cmd:option('--model_file','', 'name of file storing trained model')



opt = cmd:parse(arg or {})
print(opt)

local starting_entity_creation_weight = 0.1
local shrinking_rate = 0.9999
local shrinking_weight = starting_entity_creation_weight

local output_debug_prefix=nil
if opt.output_debug_prefix~='' then
   output_debug_prefix=opt.output_debug_prefix
end

local output_guesses_file=nil
if opt.output_guesses_file~='' then
   output_guesses_file=opt.output_guesses_file
end

-- ****** other general parameters ******

-- chunks to read files into
BUFSIZE = 2^23 -- 1MB

if (opt.use_cuda ~= 0) then
   require 'cunn'
end



-- ****** checking command-line arguments ******

-- check that batch size is smaller than training set size: if it
-- isn't, set it to training set size
if (opt.mini_batch_size>opt.training_set_size) then
   print('passed mini_batch_size larger than training set size, setting it to training set size')
   opt.mini_batch_size=opt.training_set_size
end

-- also, let's check if training_set_size is not a multiple of batch_size
local number_of_batches=math.floor(opt.training_set_size/opt.mini_batch_size)
print('each epoch will contain ' .. number_of_batches .. ' mini batches')
local left_out_training_samples_size=opt.training_set_size-(number_of_batches*opt.mini_batch_size)
-- local used_training_samples_size=training_set_size-left_out_training_samples_size
if (left_out_training_samples_size>0) then
   print('since training set size is not a multiple of mini batch size, in each epoch we will exclude ' .. left_out_training_samples_size .. ' random training samples')
end

-- for validation, we use the same batch size
local number_of_valid_batches=math.floor(opt.validation_set_size/opt.mini_batch_size)
local left_out_valid_samples_size=opt.validation_set_size-(number_of_valid_batches*opt.mini_batch_size)
if (left_out_valid_samples_size>0) then
   print('since validation set size is not a multiple of validation mini batch size, we will exclude ' .. left_out_valid_samples_size .. ' validation samples')
end


-- ****** loading models, data handling functions ******

print('reading the models file')
print('reading the data processing file')
dofile('data-supervision.lua')
dofile('supervision_model_utils.lua')
dofile('dire_models.lua')

-- ****** input data reading ******

-- NB: This goes before initializations bc some parameters needed to
-- intialize the models are initialized during data reading

print('preparing the data')

-- reading word embeddings
t_input_size = 100
word_embeddings = create_onehots(t_input_size, 'a_')
--reading image embeddings
v_input_size = 1000
image_embeddings = create_onehots(v_input_size, 'e_')

-- reading in the training data
training_input_table,training_output_table=
   create_data_tables_from_file(
      opt.protocol_prefix .. ".train",
      opt.training_set_size,
      opt.input_sequence_cardinality)
      

-- reading in the validation data
validation_input_table,validation_output_table=
   create_data_tables_from_file(
      opt.protocol_prefix .. ".valid",
      opt.validation_set_size,
      opt.input_sequence_cardinality)
      
-- ******* initializations *******

-- initializing the optimization hyperparameters
local optimization_parameters = {}
if (opt.optimization_method=="sgd") then
   -- hyperparameters for plain sgd
   optimization_parameters = {
      learningRate = opt.learning_rate,
      weightDecay = opt.weight_decay,
      momentum = opt.momentum,
      learningRateDecay = opt.learning_rate_decay 
   }
else -- currently only alternative is adam
   optimization_parameters = {
      learningRate = opt.learning_rate,
      weightDecay = opt.weight_decay
   }
end

print('assembling and initializing the model')

-- setting up the criterion
-- we use the negative log-likelihood criterion (which expects LOG probabilities
-- as model outputs!)
local criterion=nn.ClassNLLCriterion()
local entity_creation_criteria = {}
for i=1,opt.input_sequence_cardinality - 1 do
   local entity_criterion = nn.BCECriterion()
   if (opt.use_cuda ~= 0) then
      entity_criterion:cuda()
   end
   table.insert(entity_creation_criteria,entity_criterion)
end
if (opt.use_cuda ~= 0) then
   criterion:cuda()
end

-- initializing the model
-- NB: gpu processing should be done within the model building
-- functions, to make sure weight sharing is handled correctly!

if (opt.model=='entity_prediction_image_att_shared_neprob_supervision') then
   model=entity_prediction_image_att_shared_neprob_supervision(t_input_size,
        v_input_size,
        opt.multimodal_size,
        opt.input_sequence_cardinality,
        opt.temperature,
        opt.dropout_prob,
        opt.use_cuda)
else
   print("wrong model name, program will die")
end

-- getting pointers to the model weights and their gradient
model_weights, model_weight_gradients = model:getParameters()
-- initializing
model_weights:uniform(-0.08, 0.08) -- small uniform numbers, taken from char-rnn

-- following to remain here as an example even when we don't use it,
-- if we need a model for initialization of specific parameters!
-- -- if we are working with entity_prediction model, we want the bias for the new cell to be high and the intercept to be negative
-- if (opt.model=='entity_prediction') then
--    for _,node in ipairs(model.forwardnodes) do
--       if node.data.annotations.name=='raw_new_entity_mass_2' then -- because of parameter sharing, sufficient to set first
--                                                            -- only
--   node.data.module.bias:fill(5)
--   node.data.module.weight:fill(-1)
--       end
--    end
-- end

print('number of parameters in the model: ' .. model_weights:nElement())


-- ******* feval function to perform forward/backward step *******

-- the closure feval computes the loss and the gradients of the
-- loss function with respect to the weights of the model;
-- this closure will be passed as a parameter to the optimization routine
feval = function(x)

   -- in case a set of weights that is different from the one we got
   -- for the model at the last iteration of the optimization process
   -- is passed, we update the model weights to reflect the weights
   -- that are passed (this should not really be necessary, I'm
   -- putting it here only because everybody else does it)
   if x ~= model_weights then
      model_weights:copy(x)
   end
   -- reset gradients
   model_weight_gradients:zero()

   -- we assume existence of global batch_input_representations_table and batch_gold_index_tensor

   -- take forward pass for current training batch
   local all_output=model:forward(batch_input_representations_table)
   local model_prediction = all_output[#all_output]
   

   local prediction_loss = criterion:forward(model_prediction,batch_gold_index_tensor)
--   print("model_prediction: ")
--   print(model_prediction)
--   print("batch_gold_index_tensor: ")
--   print(batch_gold_index_tensor)
   -- note that according to documentation, loss is already normalized by batch size
   -- take backward pass (note that this is implicitly updating the weight gradients)
   
   -- define shrinking weight somewhere (one go down with time and one does not)
   shrinking_weight = shrinking_weight * shrinking_rate
   
   local building_entity_loss = 0
   for i=1,opt.input_sequence_cardinality - 1 do
      building_entity_loss = building_entity_loss + entity_creation_criteria[i]:forward(all_output[i],batch_gold_creation[i]) 
   end
   
   local loss = shrinking_weight * building_entity_loss + prediction_loss 
--   print("shrinking_weight: ")
--   print(shrinking_weight)
--   
--   print("prediction_loss: ")
--   print(prediction_loss)
--   
--   print("loss: ")
--   print(loss)
   
   local loss_prediction_gradient = criterion:backward(model_prediction,batch_gold_index_tensor)
   
   local output_gradient = {}
   
   for i=1,opt.input_sequence_cardinality - 1 do
      local loss_building_entity_gradient = (entity_creation_criteria[i]:backward(all_output[i],batch_gold_creation[i])):mul(shrinking_weight)
      table.insert(output_gradient,loss_building_entity_gradient)
   end
   table.insert(output_gradient, loss_prediction_gradient)
   
   model:backward(batch_input_representations_table,output_gradient)

   -- clip gradients element-wise
   model_weight_gradients:clamp(-opt.grad_clip,opt.grad_clip)
   return loss,model_weight_gradients
end


-- ******* validation function *******
function test(input_table,output_table,valid_batch_size,number_of_valid_batches,valid_set_size,left_out_samples,debug_file_prefix,guesses_file)

   local valid_batch_begin_index = 1
   local cumulative_loss = 0
   local hit_count=0

   -- preparing for debug
   local f1=nil; local f2=nil; local f3=nil;
   if debug_file_prefix then -- debug_file_prefix will be nil if debug mode is not on
      f1 = io.open(debug_file_prefix .. '.simprofiles',"w")
      f2 = io.open(debug_file_prefix .. '.cumsims',"w")
      f3 = io.open(debug_file_prefix .. '.querysims',"w")
   end

   -- preparing for model guesses
   local f4=nil
   if guesses_file then
      print("writing individual model predictions to file " .. guesses_file .. " (the file will be overriden every epoch)")
      f4 = io.open(guesses_file,"w")
   end
   
   -- reading the validation data batch by batch
   while ((valid_batch_begin_index+valid_batch_size-1)<=valid_set_size) do
      local batch_valid_input_representations_table,batch_valid_gold_entity_creation, batch_valid_gold_index_tensor=
           create_input_structures_from_table(input_table,
              output_table,
              torch.range(valid_batch_begin_index,valid_batch_begin_index+valid_batch_size-1),
              valid_batch_size,
              t_input_size,
              v_input_size,
              opt.input_sequence_cardinality,
              opt.use_cuda)

      -- passing current test samples through the trained network
      local all_output=model:forward(batch_valid_input_representations_table)
      local model_prediction = all_output[#all_output]
  
      -- accumulate loss
      -- NB: according to documentation, the criterion function already normalizes loss!
      cumulative_loss = cumulative_loss + criterion:forward(model_prediction, batch_valid_gold_index_tensor)

      -- accumulate hit counts for accuracy
      -- to compute accuracy, we first retrieve list of indices of image
      -- vectors that were preferred by the model
      local model_guesses_probs,model_guesses_indices=torch.max(model_prediction,2)
      -- we then count how often these guesses are the same as the gold
      -- note conversions to long if we're not using cuda as only tensor
      -- type
      if (opt.use_cuda~=0) then
        hit_count=hit_count+torch.sum(torch.eq(batch_valid_gold_index_tensor:type('torch.CudaLongTensor'),model_guesses_indices))
      else
        hit_count=hit_count+torch.sum(torch.eq(batch_valid_gold_index_tensor:long(),model_guesses_indices))
      end

      -- debug from here
      if debug_file_prefix and (opt.model=='entity_prediction_image_att_shared' or opt.model=='entity_prediction_image_att_shared_neprob') then -- debug_file_prefix will be nil if debug mode is not on

   local nodes = model:listModules()[1]['forwardnodes']

   -- collect debug information
   local query_entity_similarity_profile_tensor = nil
   for _,node in ipairs(nodes) do
      if node.data.annotations.name=='query_entity_similarity_profile' then
         query_entity_similarity_profile_tensor=node.data.module.output
      end
   end

   local similarity_profiles_table = {}
   local raw_cumulative_similarity_table = {}
   for i=2,opt.input_sequence_cardinality do
      for _,node in ipairs(nodes) do
         if node.data.annotations.name=='normalized_similarity_profile_' .. i then
      table.insert(similarity_profiles_table,node.data.module.output)
         elseif node.data.annotations.name=='raw_cumulative_similarity_' .. i then
      table.insert(raw_cumulative_similarity_table,node.data.module.output)
         end
         if node.data.annotations.name=='query_entity_similarity_profile' then
      query_entity_similarity_profile_tensor=node.data.module.output
         end
      end
   end

   -- write debug information to files
   for i=1,valid_batch_size do
      for j=1,#similarity_profiles_table do
         local ref_position = j+1
         f1:write("::",ref_position,"::")
         for k=1,similarity_profiles_table[j]:size(2) do
      f1:write(" ",similarity_profiles_table[j][i][k])
         end
         f1:write(" ")
      end
      f1:write("\n")
      for j=1,#raw_cumulative_similarity_table do
         local ref_position = j+1
         f2:write("::",ref_position,":: ",raw_cumulative_similarity_table[j][i][1]," ")
      end
      f2:write("\n")
      for k=1,query_entity_similarity_profile_tensor:size(3) do
         f3:write(query_entity_similarity_profile_tensor[i][1][k]," ")
      end
      f3:write("\n")
   end
      end
      -- debug to here
      
      -- write model guesses and probabilities to file
      if guesses_file then
   for i=1,model_guesses_probs:size(1) do
      f4:write(model_guesses_indices[i][1]," ",model_guesses_probs[i][1],"\n")
   end
      end
      
      valid_batch_begin_index=valid_batch_begin_index+valid_batch_size
   end -- end while

   if debug_file_prefix then
      f1:flush(); f1.close()
      f2:flush(); f2.close()
      f3:flush(); f3.close()
   end
   if guesses_file then
      f4:flush(); f4.close()
   end
   
   local average_loss=cumulative_loss/number_of_valid_batches
   local accuracy=hit_count/(valid_set_size-left_out_samples) -- we discount the samples that don't go into the batches
   return average_loss,accuracy
end


-- ****** here, the actual training and validating process starts ******

print('proceeding to training and validation')

-- we go through the training data for minimally min_epochs, maximally
-- max_epochs, and we stop if there are max_validation_lull in which
-- the validation loss does not decrease
local epoch_counter=1
local continue_training=1
local previous_validation_loss=math.huge -- high loss, to make sure we are going to "improve" on first epoch
local non_improving_epochs_count=0

while (continue_training==1) do

   print('now going through epoch ' .. epoch_counter)

   --resetting variable to keep track of average loss
   local current_loss = 0

   -- getting a shuffled index through the training data,
   -- so that they are processed in a different order at each epoch
   local shuffle = torch.randperm(opt.training_set_size)
   model:training() -- for dropout; make sure we are in training mode

   -- we now start reading batches
   local batch_begin_index = 1
   while ((batch_begin_index+opt.mini_batch_size-1)<=opt.training_set_size) do
      local current_batch_indices=shuffle:narrow(1,batch_begin_index,opt.mini_batch_size)
      batch_input_representations_table,batch_gold_creation, batch_gold_index_tensor=
   create_input_structures_from_table(training_input_table,
              training_output_table,
              current_batch_indices,
              opt.mini_batch_size,
              t_input_size,
              v_input_size,
              opt.input_sequence_cardinality,
              opt.use_cuda)

      local losses={}
      if (opt.optimization_method=="sgd") then
   _,losses = optim.sgd(feval,model_weights,optimization_parameters)
      else -- for now, adam is only alternative
   _,losses = optim.adam(feval,model_weights,optimization_parameters)
      end
      -- sgd and adam actually return only one loss
      current_loss = current_loss + losses[#losses]
      batch_begin_index=batch_begin_index+opt.mini_batch_size
   end
   
   -- average loss on number of batches
   current_loss = current_loss/number_of_batches

   print('done with epoch ' .. epoch_counter .. ' with average training loss ' .. current_loss)

   -- debug information
   local output_debug_prefix_epoch = nil
   if output_debug_prefix and opt.model=='entity_prediction_image_att_shared' then -- if output_debug_prefix is not nil, we are in debug mode
      output_debug_prefix_epoch = output_debug_prefix .. epoch_counter  -- will be used in test function (called below)
      -- this is done once per epoch:
      local nodes = model:listModules()[1]['forwardnodes']
      for _,node in ipairs(nodes) do
   if node.data.annotations.name=='raw_new_entity_mass_2' then
      print('new mass bias is ' .. node.data.module.bias[1])
      print('new mass weight is ' .. node.data.module.weight[1][1])
   end
      end
      print("writing further info for debugging/analysis in file(s) with prefix " .. output_debug_prefix_epoch) -- done in test function (called below)
   end

   -- validation
   model:evaluate() -- for dropout; get into evaluation mode (all weights used)

   local validation_loss,validation_accuracy =
      test(validation_input_table,validation_output_table,opt.mini_batch_size,number_of_valid_batches,opt.validation_set_size,left_out_training_samples_size,output_debug_prefix_epoch,output_guesses_file)
   print('validation loss: ' .. validation_loss)
   print('validation accuracy: ' .. validation_accuracy)

   -- if we are below or at the minumum number of required epochs, we
   -- won't stop no matter what
   -- if we have reached the max number of epochs, we stop no matter what
   if (epoch_counter>=opt.max_epochs) then
      continue_training=0
   -- if we are in the between epoch range, we must check that the
   -- current validation loss has not been in non-improving mode for
      -- max_validation_lull epochs
   elseif (epoch_counter>opt.min_epochs) then
      if (validation_loss>=previous_validation_loss) then
         non_improving_epochs_count=non_improving_epochs_count+1
         if (non_improving_epochs_count>=opt.max_validation_lull) then
           continue_training=0
         end
      else 
         non_improving_epochs_count=0
      end
   end
   previous_validation_loss=validation_loss
   epoch_counter=epoch_counter+1
end

-- training is over, if we were asked to save the model to a file, now it's the
-- time to do it
if (opt.save_model_to_file ~= '') then
   print('saving model to file ' .. opt.save_model_to_file)
   torch.save(opt.save_model_to_file,model)
end
