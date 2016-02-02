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
-- temporary flag for debugging
cmd:option('--debug',0,'call special debug functions')

-- decide if we want to work with toy or real data
cmd:option('--toy',0,'work with toy data? (generated within the script); else, real data from files; default: 0; set to 1 if you want to use toy data')

-- the following options are only used if we work with real data
cmd:option('--word_lexicon_file','','word lexicon file (with word vectors; first field word, rest of the fields vector values)')
cmd:option('--image_dataset_file','','image dataset file (with visual vectors; first field word and image, rest of the fields vector values)')
cmd:option('--stimuli_prefix','','prefix for stimuli files. Expects files PREFIX.(train|valid|text) to be in the folder where program is called. Format: one stimulus set per line: first field linguistic referring expression (RE), second field index of the right image for the RE in the image set (see next), rest of the fields image set (n indices of the images in the image dataset')

-- the following options are only used if we work with toy data such
-- that we generate a training and a validation set, rather than
-- reading them
cmd:option('--training_set_size',10, 'training set size')
cmd:option('--validation_set_size',3, 'validation set size')
cmd:option('--image_set_size', 5, 'max number of images in a set')
cmd:option('--min_filled_image_set_size',3, 'number of image slots that must be filled, not padded with zeroes (if it is higher than image set size, it is re-set to the latter')

-- model parameters (except for reference size, these might be
-- overwritten by data reading routine, if we're working with real
-- data; the image embedding size is also overwritten by the toy data
-- routine!)
cmd:option('--t_input_size',5, 'word embedding size')
cmd:option('--v_input_size', 5, 'image embedding size')
-- the following is only relevant for models with reference vectors
cmd:option('--reference_size',3, 'size of reference vectors')

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
-- to increase/be stable, before training is stop
cmd:option('--max_validation_lull',2,'number of adjacent non-improving epochs before stop')
-- size of a mini-batch
cmd:option('--mini_batch_size',2,'mini batch size')
opt = cmd:parse(arg or {})
print(opt)

print('reading the models file')
dofile('models.lua')

print('reading the data processing file')
dofile('data.lua')

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
   for j=1,image_set_size do
      table.insert(batch_image_set_list,training_image_set_list[j]:index(1, current_batch_indices))
   end

   -- take forward pass for current training batch
   local model_prediction=model:forward({batch_word_query_list,unpack(batch_image_set_list)})
   local loss = nll_criterion:forward(model_prediction,batch_index_list)
   -- note that we don't normalize loss by batch size, since we will later
   -- divide it by the training set size (or rather, the largest multiple of batch size
   -- that is lower or equal to training set size)
   -- take backward pass (note that this is implicitly updating the weight gradients)
   local loss_gradient = nll_criterion:backward(model_prediction,batch_index_list)
   model:backward({batch_word_query_list,unpack(batch_image_set_list)},loss_gradient)

   -- clip gradients element-wise
   model_weight_gradients:clamp(-opt.grad_clip,opt.grad_clip)
   return loss,model_weight_gradients
end


--[[
******* testing function *******
--]]

function test(test_word_query_list,test_image_set_list,test_index_list)
   
   local test_set_size=test_word_query_list:size(1)

   local cumulative_loss=0
   local hit_count=0

   for i=1,test_set_size do
      local current_gold_index=test_index_list[i]
      local model_prediction=model:forward({test_word_query_list[i],unpack(test_image_set_list[i])})
      local loss = nll_criterion:forward(model_prediction,current_gold_index)
      cumulative_loss = loss + cumulative_loss
      -- we retrieve index of image vector that was preferred by model
      local _,guessed_index = torch.max(model_prediction,2)
      -- ... and consider it a hit if it's the same as the gold index
      if (guessed_index[{1,1}]==current_gold_index) then
	 hit_count=hit_count+1
      end
   end

   local average_loss=cumulative_loss/test_set_size
   local accuracy=hit_count/test_set_size
   return average_loss,accuracy
end


--[[
****** input data reading ******
--]]
-- NB: This goes before initializations bc some parameters are
-- initialized during data reading

print('preparing the data')
-- following should be local, not local now because of interaction with dofile!
training_set_size=0
validation_set_size=0
image_set_size=0
t_input_size=0
v_input_size=0
min_filled_image_set_size=0

-- NB, with toy data, command-line v_input size option is overwritten!
if opt.toy ~= 0 then
   dofile('marco_main.lua')
else
   dofile('gemma_main.lua')
end


-- check that batch size is smaller than training set size: if it
-- isn't, set it to training set size
local mini_batch_size=opt.mini_batch_size
if (mini_batch_size>training_set_size) then
   print('passed mini_batch_size larger than training set size, setting it to training set size')
   mini_batch_size=training_set_size
end

-- also, let's check if training_set_size is not a multiple of batch_size
local number_of_batches=math.floor(training_set_size/mini_batch_size)
print('each epoch will contain ' .. number_of_batches .. ' mini batches')
local left_out_training_samples_size=training_set_size-(number_of_batches*mini_batch_size)
local used_training_samples_size=training_set_size-left_out_training_samples_size
if (left_out_training_samples_size>0) then
   print('since training set size is not a multiple of mini batch size, in each epoch we will exclude '
	    .. left_out_training_samples_size 'random training samples')
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
-- here, we will have an option-based switch to decide which model...
-- for now, only debugging
if (opt.debug==1) then
   model=ff_reference_debug(t_input_size,v_input_size,image_set_size,opt.reference_size)
else
   model=ff_reference(t_input_size,v_input_size,image_set_size,opt.reference_size)
end
-- we use the negative log-likelihood criterion (which expects LOG probabilities
-- as model outputs!)
nll_criterion= nn.ClassNLLCriterion()
-- getting pointers to the model weights and their gradient
model_weights, model_weight_gradients = model:getParameters()
-- initializing
model_weights:uniform(-0.08, 0.08) -- small uniform numbers, taken from char-rnn
print('number of parameters in the model: ' .. model_weights:nElement())

   
--[[
****** here, the actual training and validating process starts ******
--]]

print('proceeding to training and validation')

-- we go through the training data for minimally min_epochs, maximally
-- max_epochs, and we stop if there are max_validation_lull in which
-- the validation loss does not decrease
local epoch_counter=1
local continue_training=1
local previous_validation_loss=1e10 -- arbitrary high loss, to make sure we are going to "improve" on first epoch
local non_improving_epochs_count=0
while (continue_training==1) do

   print('now going through epoch ' .. epoch_counter)

   --resetting variable to keep track of average loss
   local current_loss = 0

   -- getting a shuffled index through the training data,
   -- so that they are processed in a different order at each epoch
   local shuffle = torch.randperm(training_set_size):long()
          -- note that shuffle has to be LongTensor for compatibility
          -- with the index function used below

   -- we now start reading batches

   local batch_begin_index = 1
   while ((batch_begin_index+mini_batch_size-1)<=training_index_list:size(1)) do
      current_batch_indices=shuffle:narrow(1,batch_begin_index,mini_batch_size)
      local _,losses = optim.sgd(feval,model_weights,sgd_parameters)
      -- sgd returns only one loss
      current_loss = current_loss + losses[1]
      batch_begin_index=batch_begin_index+mini_batch_size
   end
   
   -- average loss on current epoch
   current_loss = current_loss/used_training_samples_size

   print('done with epoch ' .. epoch_counter .. ' with average training loss ' .. current_loss)

   --[[ DEBUG validation out for now
   -- validation
   local validation_loss,validation_accuracy=test(validation_word_query_list,validation_image_set_list,validation_index_list)
   print('validation loss: ' .. validation_loss)
   print('validation accuracy: ' .. validation_accuracy)
   --]]
   -- if we are below or at the minumum number of required epochs, we
   -- won't stop no matter what
   if (epoch_counter<=opt.min_epochs) then
      continue_training=1
      -- if we have reached the max number of epochs, we stop no matter what
   elseif (epoch_counter>=opt.max_epochs) then
      continue_training=0
   --[===[ DEBUG validation out for now
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
   --]===]
   end
   -- DEBUG validation out for now
   -- previous_validation_loss=validation_loss
   epoch_counter=epoch_counter+1
end


