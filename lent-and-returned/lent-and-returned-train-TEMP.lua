-- preambles

require('nn')
require('nngraph')
require('optim')
require('../LinearNB') -- for linear mappings without bias

-- making sure random is random!
math.randomseed(os.time())


-- ******* options *******

cmd = torch.CmdLine()

-- options concerning input processing 
cmd:option('--protocol_prefix','','prefix for protocol files. Expects files PREFIX.(train|valid) to be in the folder where program is called. Format: one trial per line: first field linguistic query, second field index of the first occurrence of query in the object token sequence (see next), rest of the fields are token sequence (query format: object:att1:att2; object token format: att:object)')
cmd:option('--word_embedding_file','','word embedding file (with word vectors; first field word, rest of the fields vector values)')
cmd:option('--image_embedding_file','','image embedding file (with visual vectors; first field word and image, rest of the fields vector values)')
cmd:option('--normalize_embeddings',0, 'whether to normalize word and image representations, set to 1 to normalize')
cmd:option('--input_sequence_cardinality', 0, 'number of object tokens in a sequence')
cmd:option('--training_set_size',0, 'training set size')
cmd:option('--validation_set_size',0, 'validation set size')

-- options concerning output processing
cmd:option('--save_model_to_file','', 'if a string is passed, after training has finished, the trained model is saved as binary file named like the string')

-- model parameters
cmd:option('--model','entity_prediction','name of model to be used (currently supported: entity_prediction (default), ff, entity_prediction_query_token_mappings, entity_prediction_query_buggy)')
---- entity_prediction parameters
cmd:option('--multimodal_size',300, 'size of multimodal vectors')
cmd:option('--new_mass_aggregation_method','mean','when computing the new entity mass cell, use as input mean (default) or sum of values in similarity profile')
---- ff parameters
cmd:option('--hidden_size',300, 'size of hidden layer')
cmd:option('--hidden_count',1,'number of hidden layers')
cmd:option('--ff_nonlinearity','none','nonlinear transformation of hidden layers (options: none (default), sigmoid, relu, tanh)')

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

opt = cmd:parse(arg or {})
print(opt)

-- ****** other general parameters ******

-- chunks to read files into
BUFSIZE = 2^23 -- 1MB




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

-- ****** loading models, data handling functions ******

print('reading the models file')
dofile('model.lua')

print('reading the data processing file')
dofile('data.lua')

-- ****** input data reading ******

-- NB: This goes before initializations bc some parameters needed to
-- intialize the models are initialized during data reading

print('preparing the data')

-- reading word embeddings
word_embeddings,t_input_size=load_embeddings(opt.word_embedding_file,opt.normalize_embeddings)
--reading image embeddings
image_embeddings,v_input_size=load_embeddings(opt.image_embedding_file,opt.normalize_embeddings)
-- reading in the training data
training_input_table,training_gold_index_list=
   create_input_structures_from_file_BACKUP(
      opt.protocol_prefix .. ".train",
      opt.training_set_size,
      t_input_size,
      v_input_size,
      opt.input_sequence_cardinality)

-- reading in the validation data
validation_input_table,validation_gold_index_list=
   create_input_structures_from_file_BACKUP(
      opt.protocol_prefix .. ".valid",
      opt.validation_set_size,
      t_input_size,
      v_input_size,
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
   }
end

print('assembling and initializing the model')

-- setting up the criterion
-- we use the negative log-likelihood criterion (which expects LOG probabilities
-- as model outputs!)
criterion=nn.ClassNLLCriterion()

-- initializing the model

if (opt.model=='ff') then
   model=ff(t_input_size,
	    v_input_size,
	    opt.hidden_size,
	    opt.input_sequence_cardinality,
	    opt.hidden_count,
	    opt.ff_nonlinearity)
elseif (opt.model=='entity_prediction_query_token_mappings') then
   model=entity_prediction_query_token_mappings(t_input_size,
			   v_input_size,
			   opt.multimodal_size,
			   opt.input_sequence_cardinality)
elseif (opt.model=='entity_prediction_buggy') then
   model=entity_prediction_buggy(t_input_size,
			   v_input_size,
			   opt.multimodal_size,
			   opt.input_sequence_cardinality)
elseif (opt.model=='entity_prediction_old_wrong_sharing') then
   model=entity_prediction_old_wrong_sharing(t_input_size,
			   t_input_size+v_input_size,
			   opt.multimodal_size,
			   opt.input_sequence_cardinality)
else -- default is entity prediction
   model=entity_prediction_separate_mappings(t_input_size,
			   t_input_size+v_input_size,
			   opt.multimodal_size,
			   opt.input_sequence_cardinality)
end

-- getting pointers to the model weights and their gradient
model_weights, model_weight_gradients = model:getParameters()
-- initializing
model_weights:uniform(-0.08, 0.08) -- small uniform numbers, taken from char-rnn
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

   -- in the following we assume there is a current_batch_indices tensor telling
   -- us which samples are in current batch
   local batch_input_table={}
   for j=1,#training_input_table do
      table.insert(batch_input_table,training_input_table[j]:index(1,current_batch_indices))
   end
   batch_gold_index_list=training_gold_index_list:index(1,current_batch_indices)

   -- take forward pass for current training batch
   local model_prediction=model:forward(batch_input_table)
   local loss = criterion:forward(model_prediction,batch_gold_index_list)
   -- note that according to documentation, loss is already normalized by batch size
   -- take backward pass (note that this is implicitly updating the weight gradients)
   local loss_gradient = criterion:backward(model_prediction,batch_gold_index_list)
   model:backward(batch_input_table,loss_gradient)

   -- clip gradients element-wise
   model_weight_gradients:clamp(-opt.grad_clip,opt.grad_clip)
   return loss,model_weight_gradients
end


-- ******* validation function *******
function test(input_table,gold_index_list)

   -- passing all test samples through the trained network
   local model_prediction=model:forward(input_table)

   -- compute loss
   -- NB: according to documentation, the criterion function already normalizes loss!
   local average_loss = criterion:forward(model_prediction,gold_index_list)

   -- compute accuracy
   -- to compute accuracy, we first retrieve list of indices of image
   -- vectors that were preferred by the model
   _,model_guesses_indices=torch.max(model_prediction,2)
   -- we then count how often this guesses are the same as the gold
   -- note conversions to long because
   -- model_guesses_indices is long tensor
   local hit_count=torch.sum(torch.eq(gold_index_list:long(),model_guesses_indices))
   -- normalizing accuracy by test/valid set size
   local accuracy=hit_count/gold_index_list:size(1)

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
   local validation_loss,validation_accuracy=test(validation_input_table,validation_gold_index_list)
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

-- trainign is over, if we were asked to save the model to a file, now it's the
-- time to do it
if (opt.save_model_to_file ~= '') then
   print('saving model to file ' .. opt.save_model_to_file)
   torch.save(opt.save_model_to_file,model)
end
