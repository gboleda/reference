-- preambles

require('nn')
require('nngraph')
require('optim')
require('LinearNB') -- for linear mappings without bias

-- making sure random is random!
math.randomseed(os.time())

--[[
******* options *******
]]--

cmd = torch.CmdLine()
-- temporary flag for debugging, might or might not be used
cmd:option('--debug',0,'call special debug functions')

-- this option is needed to specify the path to the reference codebase,
-- in case this main script is launched from another directory (must include
-- final slash!)
-- commented out because it was complicated to make sure that also the other files
-- called from this one would be found, so Marco decided to keep running the
-- program from its own directory, rather specifying the path to input and
-- output files on the command line
-- cmd:option('--codebase_path','','path to reference codebase, needed if main script is launched from elsewhere')

-- the following options are used both with toy and with real data
cmd:option('--image_set_size', 0, 'max number of images in a set')
cmd:option('--training_set_size',0, 'training set size')
cmd:option('--validation_set_size',0, 'validation set size')

cmd:option('--save_model_to_file','', 'if a string is passed, after training has finished, the trained model is saved as binary file named like the string')

-- decide if we want to work with toy or real data
cmd:option('--toy',0,'work with toy data? (generated within the script); else, real data from files; default: 0; set to 1 if you want to use toy data')

-- the following options are only used if we work with real data
cmd:option('--word_embedding_file','','word embedding file (with word vectors; first field word, rest of the fields vector values)')
cmd:option('--image_embedding_file','','image embedding file (with visual vectors; first field word and image, rest of the fields vector values)')
cmd:option('--normalize_embeddings',0, 'whether to normalize word and image representations, set to 1 to normalize')
cmd:option('--protocol_prefix','','prefix for protocol files. Expects files PREFIX.(train|valid|test) to be in the folder where program is called (train and valid mandatory, test is considered only if test_set_size is larger than 0). Format: one stimulus set per line: first field linguistic referring expression (RE), second field index of the right image for the RE in the image set (see next), rest of the fields image set (n indices of the images in the image dataset')
cmd:option('--modifier_mode',0,'if set to 1, we assume protocol files to have colon-delimited modifiers prefixed to RE and each image')
cmd:option('--test_set_size',0, 'test set size (if 0 as in default, we assume there are no test data)')
cmd:option('--output_guesses_file','','if this file is defined, at test time we print to it, as separated space-delimited columns, the index the model returned as its guess for each test item, and the corresponding log probability')
cmd:option('--skip_test_loss',0,'if set to value different from 0, loss will not be calculated on test data (to deal with deviant conditions)')
-- the following options are only used if we work with toy data such
-- that we generate a training and a validation set, rather than
-- reading them
cmd:option('--min_filled_image_set_size',0, 'number of image slots that must be filled, not padded with zeroes (if it is higher than image set size, it is re-set to the latter')
cmd:option('--t_input_size',0, 'word embedding size; only used in toy data mode')

-- model parameters
local mst = {ff_ref=true, max_margin_bl=true, ff_ref_with_summary=true, ff_ref_deviance=true, ff_ref_sim_sum=true, ff_ref_sim_sum_revert=true}
local msg='model, to choose from: '
for k, _ in pairs(mst) do msg = msg .. k .. ', ' end
cmd:option('--model','ff_ref', msg)
-- the following is only relevant for models with reference vectors
cmd:option('--reference_size',80, 'size of reference vectors; for max margin baseline, it is recycled to give size of mapped vectors')
-- -- options for the model with the deviance detection layer only
cmd:option('--nonlinearity','sigmoid', 'nonlinear transformation to be used for deviance layer model: sigmoid by default, tanh is any other string is passed')
cmd:option('--deviance_size',2,'dimensionality of deviance layer for the relevant model')
-- -- option for ff_ref_sim_sum and ff_ref_sim_sum_revert only
cmd:option('--sum_of_nonlinearities','none','whether in ff_ref_sim_sum model similarities should be filtered by a nonlinearity before being fed to the deviance layer: no filtering by default, with possible options sigmoid and relu')
-- -- option for max_margin_bl only
cmd:option('--margin',0.1,'margin size; 0.1 by default, any other numerical value possible')

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
--
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
cmd:option('--mini_batch_size',2,'mini batch size')
opt = cmd:parse(arg or {})
print(opt)

-- other general parameters
-- chunks to read files into
BUFSIZE = 2^23 -- 1MB

-- here, list models that can handle deviance, for appropriate data
-- reading
model_can_handle_deviance=0
if ((opt.model=="ff_ref_with_summary") or 
      (opt.model=="ff_ref_deviance") or
      (opt.model=="ff_ref_sim_sum") or
   (opt.model=="ff_ref_sim_sum_revert")) then
   model_can_handle_deviance=1
end

-- here, list models that need information about number of input
-- images, so that for the other models we can reset the list
-- containing this information and we feed the right data to the
-- various models
model_needs_real_image_count=0
if (opt.model=="ff_ref_sim_sum") then
   model_needs_real_image_count=1
end

--[[
****** checking command-line arguments ******
--]]
local output_guesses_file=nil
if opt.output_guesses_file~='' then
   output_guesses_file=opt.output_guesses_file
end
-- if test size is 0, we won't output test guesses to a file
-- and it makes no sense to pass the skip_test_loss option
if (opt.test_set_size == 0) then
   if (output_guesses_file) then
      print('WARNING: you requested to print test guesses to file ' .. output_guesses_file .. ' but no test set is expected... ignoring your request')
   end
   if (opt.skip_test_loss ~= 0) then
      print('WARNING: you requested to skip loss calculation in test phase, but no test set is expected... ignoring your request')
   end
end

if not mst[opt.model] then
   error('ERROR: wrong model type: ' .. tostring(opt.model))
end
if opt.toy ~= 0 then
   if opt.test_set_size ~=0 then
      print('WARNING: no testing in toy mode (resetting test_set_size to 0)')
      opt.test_set_size = 0
   end
end
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
   print('since training set size is not a multiple of mini batch size, in each epoch we will exclude '
	    .. left_out_training_samples_size .. ' random training samples')
end

--[[
****** loading models, data handling functions ******
--]]

print('reading the models file')
dofile('models.lua')
dofile('extra_models.lua') -- temp file for work in progress

print('reading the data processing file')
dofile('data.lua')

--[[
****** input data reading ******
--]]

-- NB: This goes before initializations bc some parameters needed to
-- intialize the models are initialized during data reading

print('preparing the data')
local t_input_size=0
local v_input_size=0

if opt.toy ~= 0 then
-- TOY DATA PROCESSING
   t_input_size=opt.t_input_size training_word_query_list,
   training_image_set_list, training_index_list,
   validation_word_query_list, validation_image_set_list,
   validation_index_list, v_input_size=
   generate_toy_data(opt.training_set_size, opt.validation_set_size,
		     opt.t_input_size, opt.image_set_size, opt.min_filled_image_set_size)

-- REAL DATA PROCESSING
else
   -- reading word embeddings
   word_embeddings,t_input_size=load_embeddings(opt.word_embedding_file,opt.normalize_embeddings)
   --reading image embeddings
   image_embeddings,v_input_size=load_embeddings(opt.image_embedding_file,opt.normalize_embeddings)

   if opt.model=="max_margin_bl" then
      train_data,train_gold,_,training_index_list= create_input_structures_from_file_for_max_margin(opt.protocol_prefix .. ".train",opt.training_set_size,t_input_size,v_input_size)
      -- print(train_data)
      -- print(#train_gold)
      test_data,test_gold,nimgs_per_sequence=create_input_structures_from_file_for_max_margin(opt.protocol_prefix .. ".test",opt.test_set_size,t_input_size,v_input_size)
      -- print(test_data)
      -- print(#test_gold)
      valid_data,valid_gold,_= create_input_structures_from_file_for_max_margin(opt.protocol_prefix .. ".valid",opt.validation_set_size,t_input_size,v_input_size)
      -- print(valid_data)
      -- print(#valid_gold)
   else
      -- reading in the training data
      training_input_table, training_index_list=
	 create_input_structures_from_file(
	    opt.protocol_prefix .. ".train",
	    opt.training_set_size,
	    t_input_size,
	    v_input_size,
	    opt.image_set_size)

      -- reading in the validation data
      validation_input_table, validation_index_list=   
	 create_input_structures_from_file(
	 opt.protocol_prefix .. ".valid",
	 opt.validation_set_size,
	 t_input_size,
	 v_input_size,
	 opt.image_set_size)

      -- finally, if we have test data, we load them as well
      if (opt.test_set_size>0) then
	 test_input_table, test_index_list=
	    create_input_structures_from_file(
	       opt.protocol_prefix .. ".test",
	       opt.test_set_size,
	       t_input_size,
	       v_input_size,
	       opt.image_set_size)
      end
   end
end


--[[
******* initializations *******
--]]

-- optimization hyperparameters
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
   }
end

local model_t_embedding_size = t_input_size -- we need to copy it cause otherwise we overwrite t_input_size
local model_v_embedding_size = v_input_size
-- if there are modifier, word input is actually concatenation of 
-- two vectors
if (opt.modifier_mode==1) then
   model_t_embedding_size = t_input_size*2
   model_v_embedding_size = t_input_size+v_input_size
end

print('assembling and initializing the model')

-- default criterion (CAN BE OVERRIDEN BELOW)
-- we use the negative log-likelihood criterion (which expects LOG probabilities
-- as model outputs!)
criterion=nn.ClassNLLCriterion()

-- option-based switch to set model
if opt.model == 'ff_ref' then
   model=ff_reference(model_t_embedding_size,model_v_embedding_size,opt.image_set_size,opt.reference_size)
elseif opt.model == 'max_margin_bl' then
   model=max_margin_baseline_model(model_t_embedding_size,model_v_embedding_size,opt.reference_size)
   -- Creates a criterion that measures the loss given an input x = {x1,
   -- x2}, a table of two Tensors of size 1 (they contain only scalars),
   -- and a label y (1 or -1). In batch mode, x is a table of two Tensors
   -- of size batchsize, and y is a Tensor of size batchsize containing 1
   -- or -1 for each corresponding pair of elements in the input Tensor.
   criterion=nn.MarginRankingCriterion(opt.margin)
elseif opt.model == 'ff_ref_with_summary' then
   model=ff_reference_with_reference_summary(model_t_embedding_size,model_v_embedding_size,opt.image_set_size,opt.reference_size)
elseif opt.model == 'ff_ref_deviance' then
   model=ff_reference_with_deviance_layer(model_t_embedding_size,model_v_embedding_size,opt.image_set_size,opt.reference_size,opt.deviance_size,opt.nonlinearity)
elseif opt.model == 'ff_ref_sim_sum' then
   model=ff_reference_with_similarity_sum_cell(model_t_embedding_size,model_v_embedding_size,opt.image_set_size,opt.reference_size,opt.deviance_size,opt.nonlinearity,opt.sum_of_nonlinearities)
elseif opt.model == 'ff_ref_sim_sum_revert' then
   model=ff_reference_with_similarity_sum_cell_revert(model_t_embedding_size,model_v_embedding_size,opt.image_set_size,opt.reference_size,opt.deviance_size,opt.nonlinearity,opt.sum_of_nonlinearities)
end

-- getting pointers to the model weights and their gradient
model_weights, model_weight_gradients = model:getParameters()
-- initializing
model_weights:uniform(-0.08, 0.08) -- small uniform numbers, taken from char-rnn
print('number of parameters in the model: ' .. model_weights:nElement())


--[[
******* feval function to perform forward/backward step *******
--]]

-- the closure feval computes the loss and the gradients of the
-- loss function with respect to the weights of the model;
-- this closure will be passed as a parameter to the optimization routine
feval = function(x)
   -- in case a set of weights that is different from the one we got
   -- for the model at the last iteration of the optimization process,
   -- we update the model weights to reflect the weights that are
   -- passed (this should not really be necessary, I'm putting it here
   -- only because everybody else does it)
   if x ~= model_weights then
      model_weights:copy(x)
   end
   -- reset gradients
   model_weight_gradients:zero()

   -- this assumes there is a current_batch_indices tensor telling
   -- us which samples are in current batch
   local batch_index_list=training_index_list:index(1,current_batch_indices)
   local batch_input_table={}
   if opt.model=="max_margin_bl" then
      -- batch_index_list=training_index_list:index(1,current_batch_indices) -- ***
      -- table.insert(batch_input_table,batch_index_list)
      local a=1
   else
      for j=1,#training_input_table do
	 table.insert(batch_input_table,training_input_table[j]:index(1,current_batch_indices))
      end
--      print(batch_input_table)
   end

   -- take forward pass for current training batch
   local model_prediction=model:forward(batch_input_table)
   local loss = criterion:forward(model_prediction,batch_index_list)
   -- note that according to documentation, loss is already normalized by batch size
   -- take backward pass (note that this is implicitly updating the weight gradients)
   local loss_gradient = criterion:backward(model_prediction,batch_index_list)
   model:backward(batch_input_table,loss_gradient)

   -- clip gradients element-wise
   model_weight_gradients:clamp(-opt.grad_clip,opt.grad_clip)
   return loss,model_weight_gradients
end

--[[
******* testing/validation function *******
--]]

function test(input_table,index_list,output_print_file,skip_test_loss)

   local model_prediction=model:forward(input_table)
   local average_loss = math.huge
   -- unless we are asked to skip it, compute loss
   if (skip_test_loss == 0) then
      -- NB: according to documentation, the criterion function already normalizes loss!
      average_loss = criterion:forward(model_prediction,index_list)
   end

   -- to compute accuracy, we first retrieve list of indices of image
   -- vectors that were preferred by the model
   local model_max_log_probs,model_guesses=torch.max(model_prediction,2)
   local model_max_probs=torch.exp(model_max_log_probs)
   -- we then count how often this guesses are the same as the gold
   -- (and thus the difference is 0) (note conversions to long because
   -- model_guesses is long tensor)
   local hit_count = torch.sum(torch.eq(index_list:long(),model_guesses))
   -- normalizing accuracy by test set size
   local accuracy=hit_count/index_list:size(1)

   --if requested, print guesses and their log probs to file
   if output_print_file then
         local f = io.open(output_print_file,"w")
	 for i=1,model_max_probs:size(1) do
	    f:write(model_guesses[i][1]," ",model_max_probs[i][1],"\n")
	 end
	 f:flush()
	 f.close()
   end
   return average_loss,accuracy
end

function testmmarg(input_table,tgold,nimgs,skip_test_loss)

   set_size=nimgs:size()[1]
   -- passing all test samples through the trained network
   local model_prediction=model:forward(input_table)
   -- LOSS
   local average_loss = math.huge
   -- unless we are asked to skip it, compute loss
   if (skip_test_loss == 0) then
      -- NB: according to documentation, the criterion function already normalizes loss!
      average_loss = crit:forward(model_prediction,tgold)
   end
   
   -- ACCURACY
   -- print(model_prediction)
   -- print(model_prediction[1], model_prediction[2])
   qt=model_prediction[1]
   qc=model_prediction[2]
   -- print(tostring(input_table[1]:size()[1]))
   -- print(tostring(nimgs:size()[1])) 
   local sequence_length=0
   local end_at=0
   local hit_count=0 -- torch.Tensor(1)
   -- to compute accuracy, we first get the answers for each sequence
   for i=1,set_size do
      -- print(tostring('---'))
      -- print(tostring(i))
      local start_at=end_at+1 -- next time we start after we left off
      end_at=start_at+(nimgs[i]-2) -- nimgs[i] is sequence length; -2 to discount: the target (-1), the +1 that we put in "start at" (-1)
      -- print(tostring(start_at))
      -- print(tostring(end_at))
      qt_sequence=qt[{{start_at,end_at}}]
      qc_sequence=qc[{{start_at,end_at}}]
      -- print(qt_sequence)
      -- print(qc_sequence)
      max_confounder=torch.max(qc_sequence)
      -- print('target:' .. tostring(qt_sequence))
      -- print('confounder:' .. tostring(qc_sequence))
      -- print('max target - max confounder: ' .. tostring(qt_sequence[1]) .. ';' .. tostring(max_confounder))
      if qt_sequence[1] > max_confounder then hit_count=hit_count+1 end
      i=i+1
   end
   local accuracy=hit_count/set_size

   return average_loss,accuracy
end

--[[
****** here, the actual training and validating process starts ******
--]]

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
   local shuffle = torch.randperm(opt.training_set_size):long()
          -- note that shuffle has to be LongTensor for compatibility
          -- with the index function used below

   -- we now start reading batches
   local batch_begin_index = 1
   while ((batch_begin_index+opt.mini_batch_size-1)<=opt.training_set_size) do
      current_batch_indices=shuffle:narrow(1,batch_begin_index,opt.mini_batch_size)
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

   -- validation
   local validation_loss,validation_accuracy=test(validation_input_table,validation_index_list,nil,0)
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

-- training is over, now, if test data are available, we can report
-- performance on them

if (opt.test_set_size>0) then
   print('training done and test data available...')
   local test_loss,test_accuracy=test(test_input_table,test_index_list,output_guesses_file,opt.skip_test_loss)
   print('test loss: ' .. test_loss)
   if (opt.skip_test_loss == 0) then
      print('test accuracy: ' .. test_accuracy)
   end
end

-- finally, if we were asked to save the model to a file, now it's the
-- time to do it
if (opt.save_model_to_file ~= '') then
   torch.save(opt.save_model_to_file,model)
end
