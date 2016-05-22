-- preambles

require('nn')
require('nngraph')
require('optim')
require('LinearNB') -- for linear mappings without bias

-- making sure random is random!
math.randomseed(os.time())


-- ******* options *******

cmd = torch.CmdLine()

-- options concerning input processing 
cmd:option('--protocol_prefix','','prefix for protocol files. Expects files PREFIX.(train|valid) to be in the folder where program is called. Format: one trial per line: first field linguistic query, second field index of the first occurrence of query in the object token sequence (see next), rest of the fields are token sequence (query format: att1:att2:object; object token format: att:object)')
cmd:option('--word_embedding_file','','word embedding file (with word vectors; first field word, rest of the fields vector values)')
cmd:option('--image_embedding_file','','image embedding file (with visual vectors; first field word and image, rest of the fields vector values)')
cmd:option('--normalize_embeddings',0, 'whether to normalize word and image representations, set to 1 to normalize')
cmd:option('--input_sequence_cardinality', 0, 'number of object tokens in a sequence')
cmd:option('--training_set_size',0, 'training set size')
cmd:option('--validation_set_size',0, 'validation set size')

-- options concerning output processing
cmd:option('--save_model_to_file','', 'if a string is passed, after training has finished, the trained model is saved as binary file named like the string')

-- model parameters
cmd:option('--multimodal_size',300, 'size of multimodal vectors')

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
   print('since training set size is not a multiple of mini batch size, in each epoch we will exclude '
	    .. left_out_training_samples_size .. ' random training samples')
end
