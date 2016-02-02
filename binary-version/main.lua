-- preambles

require('nn')
require('nngraph')
require('optim')

--[[
******* options *******
]]--

cmd = torch.CmdLine()
-- the following options are only used if we work with toy data such
-- that we generate a training and a validation set, rather than
-- reading them (eventually, we will need a mechanism to decide which
-- way we go)
cmd:option('--training_set_size',5, 'training set size')
cmd:option('--validation_set_size',3, 'validation set size')

-- model parameters
cmd:option('--t_input_size',2, 'word embedding size')
cmd:option('--v_input_size', 2, 'image embedding size')
cmd:option('--image_set_size', 2, 'max number of images in a set')
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

   -- take forward pass for current training sample

   -- this assume that there is an index variable next_sample_index
   -- that tells us what is the next training sample to use

   -- the following weird syntax is to make sure that the single
   -- response value stays a 1x1 tensor instead of becoming a single
   -- numerical value (which the criterion forward function dislikes)
   local current_gold_response=training_response_list[{{next_sample_index,next_sample_index}}]
   local model_prediction=model:forward({training_word_query_list[next_sample_index],unpack(training_image_set_list[next_sample_index])})
   local loss = binary_cross_entropy_criterion:forward(model_prediction,current_gold_response)
   -- take backward pass (note that this is implicitly updating the weight gradients)
   local loss_gradient = binary_cross_entropy_criterion:backward(model_prediction,current_gold_response)
   model:backward({training_word_query_list[next_sample_index],unpack(training_image_set_list[next_sample_index])},loss_gradient)
   -- clip gradients element-wise
   model_weight_gradients:clamp(-opt.grad_clip,opt.grad_clip)
   return loss,model_weight_gradients
end


--[[
******* testing function *******
--]]

function test(test_word_query_list,test_image_set_list,test_response_list)
   
   local test_set_size=test_word_query_list:size(1)

   local cumulative_loss=0
   local hit_count=0

   for i=1,test_set_size do
      local current_gold_response=test_response_list[{{i,i}}]
      local model_prediction=model:forward({test_word_query_list[i],unpack(test_image_set_list[i])})
      local loss = binary_cross_entropy_criterion:forward(model_prediction,current_gold_response)
      cumulative_loss = loss + cumulative_loss

      if ((model_prediction[1][1]>0.5 and current_gold_response[1]==1) or 
	 (model_prediction[1][1]<=0.5 and current_gold_response[1]==0)) then 
	 hit_count=hit_count+1
      end
   end

   local average_loss=cumulative_loss/test_set_size
   local accuracy=hit_count/test_set_size
   return average_loss,accuracy
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
model=ff_reference(opt.t_input_size,opt.v_input_size,opt.image_set_size,opt.reference_size)
-- we use the binary cross-entropy criterion
binary_cross_entropy_criterion= nn.BCECriterion()
-- getting pointers to the model weights and their gradient
model_weights, model_weight_gradients = model:getParameters()
-- initializing
model_weights:uniform(-0.08, 0.08) -- small uniform numbers, taken from char-rnn
print('number of parameters in the model: ' .. model_weights:nElement())

--[[
****** input data reading ******
--]]

print('preparing the data')
-- here, we will need to decide if we should use toy or real data
training_word_query_list,
validation_word_query_list,
training_image_set_list,
validation_image_set_list,
training_response_list,
validation_response_list=
   generate_toy_data(opt.training_set_size,
		     opt.validation_set_size,
		     opt.image_set_size,
		     opt.t_input_size,
		     opt.v_input_size)

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
   local shuffle = torch.randperm(opt.training_set_size)

   -- we apply "proper" stochastic gradient descent, going through the
   -- examples one by one
   for i=1,opt.training_set_size do
      next_sample_index=shuffle[i]
      local _,losses = optim.sgd(feval,model_weights,sgd_parameters)
      -- sgd returns only one loss
      current_loss = current_loss + losses[1]
   end
   
   -- average loss on current epoch
   current_loss = current_loss/opt.training_set_size

   print('done with epoch ' .. epoch_counter .. ' with average training loss ' .. current_loss)

   -- validation
   local validation_loss,validation_accuracy=test(validation_word_query_list,validation_image_set_list,validation_response_list)
   print('validation loss: ' .. validation_loss)
   print('validation accuracy: ' .. validation_accuracy)

   -- if we are below or at the minumum number of required epochs, we
   -- won't stop no matter what
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
