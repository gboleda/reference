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
-- gbt: why is it commented out?
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
cmd:option('--test_set_size',0, 'test set size (if 0 as in default, we assume there are no test data)')
cmd:option('--output_guesses_file','','if this file is defined, at test time we print to it, as separated space-delimited columns, the index the model returned as its guess for each test item, and the corresponding log probability')
cmd:option('--skip_test_loss',0,'if set to value different from 0, loss will not be calculated on test data (to deal with deviant conditions)')
-- the following options are only used if we work with toy data such
-- that we generate a training and a validation set, rather than
-- reading them
cmd:option('--min_filled_image_set_size',0, 'number of image slots that must be filled, not padded with zeroes (if it is higher than image set size, it is re-set to the latter')
cmd:option('--t_input_size',0, 'word embedding size')

-- model parameters
local mst = {ff_ref=true, max_margin_bl=true}
local msg='model, to choose from: '
for k, _ in pairs(mst) do msg = msg .. k .. ', ' end
cmd:option('--model','ff_ref', msg)
-- the following is only relevant for models with reference vectors
cmd:option('--reference_size',80, 'size of reference vectors; for max margin baseline, recycled to give size of mapped vectors')

-- training parameters
-- sgd hyperparameters (copying defaults from
-- https://github.com/torch/demos/blob/master/logistic-regression/example-logistic-regression.lua)
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
cmd:option('--mini_batch_size',2,'mini batch size')
opt = cmd:parse(arg or {})
print(opt)

-- other general parameters
-- chunks to read files into
BUFSIZE = 2^23 -- 1MB


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

print('reading the models file')
dofile('models.lua')
dofile('extra_models.lua') -- temp file for work in progress

print('reading the data processing file')
dofile('data.lua')
dofile('extra_data.lua') -- temp file for work in progress


--[[
****** input data reading ******
--]]

-- NB: This goes before initializations bc some parameters needed to
-- intialize athe models are initialized during data reading

print('preparing the data')
local t_input_size=0
local v_input_size=0

if opt.toy ~= 0 then
-- TOY DATA PROCESSING

   t_input_size=opt.t_input_size
   training_word_query_list,
   training_image_set_list,
   training_index_list,
   validation_word_query_list,
   validation_image_set_list,
   validation_index_list,
   v_input_size=
      generate_toy_data(opt.training_set_size,
			opt.validation_set_size,
			opt.t_input_size,
			opt.image_set_size,
			opt.min_filled_image_set_size
      )
else
-- REAL DATA PROCESSING
   -- reading word embeddings
   word_embeddings,t_input_size=
      load_embeddings(opt.word_embedding_file,opt.normalize_embeddings)
   --reading image embeddings
   image_embeddings,v_input_size=
      load_embeddings(opt.image_embedding_file,opt.normalize_embeddings)

   -- reading in the training data
   training_word_query_list,
   training_image_set_list,
   training_index_list=
      create_input_structures_from_file(
	 opt.protocol_prefix .. ".train",
	 opt.training_set_size,
	 t_input_size,
	 v_input_size,
	 opt.image_set_size)

   -- reading in the validation data
   validation_word_query_list,
   validation_image_set_list,
   validation_index_list=
      create_input_structures_from_file(
	 opt.protocol_prefix .. ".valid",
	 opt.validation_set_size,
	 t_input_size,
	 v_input_size,
	 opt.image_set_size)

   -- finally, if we have test data, we load them as well
   if (opt.test_set_size>0) then
      test_word_query_list,
      test_image_set_list,
      test_index_list=
	 create_input_structures_from_file(
	    opt.protocol_prefix .. ".test",
	    opt.test_set_size,
	    t_input_size,
	    v_input_size,
	    opt.image_set_size)
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
******* initializations *******
--]]

-- the hyperparameters for plain sgd
-- possibly in the future we might want to switch
-- between SGD and other optimization routines
local sgd_parameters = {
   learningRate = opt.learning_rate,
   weightDecay = opt.weight_decay,
   momentum = opt.momentum,
   learningRateDecay = opt.learning_rate_decay 
}
print('assembling and initializing the model')
-- option-based switch to set model
if opt.model == 'ff_ref' then
   model=ff_reference(t_input_size,v_input_size,opt.image_set_size,opt.reference_size)
   -- we use the negative log-likelihood criterion (which expects LOG probabilities
   -- as model outputs!)
   criterion= nn.ClassNLLCriterion()
elseif opt.model == 'max_margin_bl' then
   error('ERROR: still not implemented: ' .. tostring(opt.model))
   -- model=max_margin_baseline_model(t_input_size,v_input_size,opt.image_set_size,opt.reference_size)
   -- criterion=nn.MarginRankingCriterion()
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

   -- this assumes there is a current_batch_indices tensor telling us
   -- which samples are in current batch
   local batch_word_query_list=training_word_query_list:index(1,current_batch_indices)
   local batch_index_list=training_index_list:index(1,current_batch_indices)
   local batch_image_set_list={}
   for j=1,opt.image_set_size do
      table.insert(batch_image_set_list,training_image_set_list[j]:index(1, current_batch_indices))
   end

   -- take forward pass for current training batch
   local model_prediction=model:forward({batch_word_query_list,unpack(batch_image_set_list)})
   local loss = criterion:forward(model_prediction,batch_index_list)
   -- note that according to documentation, loss is already normalized by batch size
   -- take backward pass (note that this is implicitly updating the weight gradients)
   local loss_gradient = criterion:backward(model_prediction,batch_index_list)
   model:backward({batch_word_query_list,unpack(batch_image_set_list)},loss_gradient)

   -- clip gradients element-wise
   model_weight_gradients:clamp(-opt.grad_clip,opt.grad_clip)
   return loss,model_weight_gradients
end


--[[
******* testing/validation function *******
--]]

function test(test_word_query_list,test_image_set_list,test_index_list,output_print_file,skip_test_loss)

   -- passing all test samples through the trained network
   local model_prediction=model:forward({test_word_query_list,unpack(test_image_set_list)})
   local average_loss = math.huge
   -- unless we are asked to skip it, compute loss
   if (skip_test_loss == 0) then
      -- NB: according to documentation, the criterion function already normalizes loss!
      average_loss = criterion:forward(model_prediction,test_index_list)
   end

   -- to compute accuracy, we first retrieve list of indices of image
   -- vectors that were preferred by the model
   local model_max_log_probs,model_guesses=torch.max(model_prediction,2)
   local model_max_probs=torch.exp(model_max_log_probs)
   -- we then count how often this guesses are the same as the gold
   -- (and thus the difference is 0) (note conversions to long because
   -- model_guesses is long tensor)
   local hit_count = torch.sum(torch.eq(test_index_list:long(),model_guesses))
   -- normalizing accuracy by test set size
   local accuracy=hit_count/test_word_query_list:size(1)

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

   
--[[
****** here, the actual training and validating process starts ******
--]]

print('proceeding to training and validation')

-- we go through the training data for minimally min_epochs, maximally
-- max_epochs, and we stop if there are max_validation_lull in which
-- the validation loss does not decrease
local epoch_counter=1
local continue_training=1
local previous_validation_loss=1e10 -- arbitrary high loss, to make sure we are going to "improve" on first epoch *** gbt: set to Huge
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
   while ((batch_begin_index+opt.mini_batch_size-1)<=training_index_list:size(1)) do
      current_batch_indices=shuffle:narrow(1,batch_begin_index,opt.mini_batch_size)
      local _,losses = optim.sgd(feval,model_weights,sgd_parameters)
      -- sgd returns only one loss
      current_loss = current_loss + losses[1]
      batch_begin_index=batch_begin_index+opt.mini_batch_size
   end
   
   -- average loss on number of batches
   current_loss = current_loss/number_of_batches


   print('done with epoch ' .. epoch_counter .. ' with average training loss ' .. current_loss)

   -- validation
   local validation_loss,validation_accuracy=test(validation_word_query_list,validation_image_set_list,validation_index_list,nil,0)
   print('validation loss: ' .. validation_loss)
   print('validation accuracy: ' .. validation_accuracy)
   -- if we are below or at the minumum number of required epochs, we
   -- won't stop no matter what
   -- *** gbt: remove 'if' below (simplify)
   if (epoch_counter<=opt.min_epochs) then
      continue_training=1
   -- if we have reached the max number of epochs, we stop no matter what
   elseif (epoch_counter>=opt.max_epochs) then
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
   local test_loss,test_accuracy=test(test_word_query_list,test_image_set_list,test_index_list,output_guesses_file,opt.skip_test_loss)
   print('test loss: ' .. test_loss)
   if (opt.skip_test_loss == 0) then
      print('test accuracy: ' .. test_accuracy)
   end
end

-- finally, if we were asked to save the model to a file, not it's the
-- time to do it
if (opt.save_model_to_file ~= '') then
   torch.save(opt.save_model_to_file,model)
end
