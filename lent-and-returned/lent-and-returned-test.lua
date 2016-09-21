-- preambles

require('nn')
require('nngraph')
require('../LinearNB') -- for linear mappings without bias

-- ******* options *******

cmd = torch.CmdLine()

-- model file to be read
cmd:option('--model_file','', 'name of file storing trained model generated by lent-and-returned-main.lua')
-- run on GPU? no by default
cmd:option('--use_cuda',0,'is a GPU available? default: nope, specify value different from 0 to use GPU')
-- data files and data characteristics
cmd:option('--word_embedding_file','','word embedding file (with word vectors; first field word, rest of the fields vector values)... must be coherent with representations used at model training!')
cmd:option('--image_embedding_file','','image embedding file (with visual vectors; first field word and image, rest of the fields vector values)... must be coherent with representations used at model training!')
cmd:option('--normalize_embeddings',0, 'whether to normalize word and image representations, set to 1 to normalize: must be coherent with choice at model training')
cmd:option('--input_sequence_cardinality', 0, 'number of object tokens in a sequence')
cmd:option('--candidate_cardinality', 0, 'number of images in the output set to pick from')
cmd:option('--test_file','/Users/gboleda/Desktop/love-project/data/binding/exp-tiny/stimuli.test','name of test file; format: same as that of protocol files used for training')
cmd:option('--test_set_size',0, 'test set size')
-- output files
cmd:option('--output_guesses_file','','if this file is defined, we print to it, as separated space-delimited columns, the index the model returned as its guess for each test item, and the corresponding log probability')
cmd:option('--output_debug_file','','if this file is defined, we print to it some information that might vary depending on debugging needs (see directly code of this program to check out what it is currently printing for debugging purposes, if anything)')

opt = cmd:parse(arg or {})
print(opt)


local output_guesses_file=nil
if opt.output_guesses_file~='' then
   output_guesses_file=opt.output_guesses_file
end

local output_debug_file=nil
if opt.output_debug_file~='' then
   output_debug_file=opt.output_debug_file
end



-- other general parameters
-- chunks to read files into
BUFSIZE = 2^23 -- 1MB

-- ******* test function *******
-- (to be copied and adapted from test function in lent-and-returned-train.lua every time)
function test_and_inspect(i_table,gold_idx_list)

   -- passing all test samples through the trained network
   local model_prediction=model:forward(i_table)

   -- compute accuracy
   -- to compute accuracy, we first retrieve list of indices of image
   -- vectors that were preferred by the model
   -- *** change specific to this test function
   local model_guesses_probs,model_guesses_indices=torch.max(model_prediction,2)
   -- we then count how often these guesses are the same as the gold
   -- note conversions to long if we're not using cuda as only tensor
   -- type
   local hit_count=0
   if (opt.use_cuda~=0) then
      hit_count=torch.sum(torch.eq(gold_idx_list:type('torch.CudaLongTensor'),model_guesses_indices))
   else
      hit_count=torch.sum(torch.eq(gold_idx_list:long(),model_guesses_indices))
   end
   -- normalizing accuracy by test/valid set size
   local accuracy=hit_count/gold_idx_list:size(1)

   -- *** change specific to this test function: we return different information
   return accuracy,model_guesses_probs,model_guesses_indices
end

-- ****** loading models, data handling functions ******

print('reading the models file')
dofile('model-new.lua')

print('reading the data processing file')
dofile('data-new.lua')

-- ****** input data reading ******

-- reading word embeddings
word_embeddings,t_input_size=load_embeddings(opt.word_embedding_file,opt.normalize_embeddings)
--reading image embeddings
image_embeddings,v_input_size=load_embeddings(opt.image_embedding_file,opt.normalize_embeddings)
-- reading in the test data
input_table,gold_index_list=
   create_input_structures_from_file(
      opt.test_file,
      opt.test_set_size,
      t_input_size,
      v_input_size,
      opt.input_sequence_cardinality,
      opt.candidate_cardinality)

-- ****** model reading ******

print('reading in the model from file ' .. opt.model_file)
model = torch.load(opt.model_file)
model:evaluate() -- turns off dropout in test mode

-- *** computing model predictions and accuracy

print('computing model predictions and accuracy on test data')

local acc,guesses_probs,guesses_indices=test_and_inspect(input_table,gold_index_list)

-- OLD -- TO REMOVE WHEN TESTED
-- -- to compute accuracy, we first retrieve list of indices of items
-- -- that were preferred by the model
-- local model_guesses_probs,model_guesses_indices=torch.max(model_predictions,2)
-- -- we then count how often these guesses are the same as the gold
-- -- (and thus the difference is 0) (note conversions to long because
-- -- model_guesses_indices is long tensor)
-- local hit_count=torch.sum(torch.eq(gold_index_list:long(),model_guesses_indices))
-- -- normalizing accuracy by test set size
-- local accuracy=hit_count/opt.test_set_size

print('test set accuracy is ' .. acc)

--if requested, print guesses, their log probs and the overall prob distribution 
--to file
if output_guesses_file then
   print("writing individual model predictions to file " .. output_guesses_file)

   local f = io.open(output_guesses_file,"w")
   for i=1,guesses_probs:size(1) do
      f:write(guesses_indices[i][1]," ",guesses_probs[i][1],"\n")
   end
   f:flush()
   f.close()
end

-- if requested, print relevant information to a debug file (this might
-- change from time to time, based on debugging needs)
if output_debug_file then
   print("writing further information in " .. output_debug_file)

   local similarity_profiles_table = {}
   local nodes = model:listModules()[1]['forwardnodes']
   for i=2,opt.input_sequence_cardinality do
      local target_annotation = 'normalized_similarity_profile_' .. i
      for _,node in ipairs(nodes) do
	 if node.data.annotations.name==target_annotation then
	    table.insert(similarity_profiles_table,node.data.module.output)
	 end
      end
   end
   local f = io.open(output_debug_file,"w")
   for i=1,opt.test_set_size do
      for j=1,#similarity_profiles_table do
	 local ref_position = j+1
	 f:write("::",ref_position,"::")
	 for k=1,similarity_profiles_table[j]:size(2) do
	    f:write(" ",similarity_profiles_table[j][i][k])
	 end
	 f:write(" ")
      end
      f:write("\n")
   end
   f:flush()
   f.close()
end

