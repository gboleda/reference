-- preambles

require('nn')
require('nngraph')
require('../LinearNB') -- for linear mappings without bias

-- ******* options *******

cmd = torch.CmdLine()

-- model file to be read
cmd:option('--model_file','', 'name of file storing trained model generated by lent-and-returned-train.lua')
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
cmd:option('--output_debug_prefix','','if this prefix is defined, we print to one or more files with this prefix and various suffixes information that might vary depending on debugging needs (see directly code of this program to check out what it is currently being generated for debugging purposes, if anything)')

opt = cmd:parse(arg or {})
print(opt)


local output_guesses_file=nil
if opt.output_guesses_file~='' then
   output_guesses_file=opt.output_guesses_file
end

local output_debug_prefix=nil
if opt.output_debug_prefix~='' then
   output_debug_prefix=opt.output_debug_prefix
   print("further info for debugging/analysis will be written in file(s) with prefix " .. output_debug_prefix) -- done in test function (called below)

end

-- other general parameters
-- chunks to read files into
BUFSIZE = 2^23 -- 1MB

-- we are not using the loss for the moment -- just in case (also cause test function needs it)
-- setting up the criterion
-- we use the negative log-likelihood criterion (which expects LOG probabilities
-- as model outputs!)
local criterion=nn.ClassNLLCriterion()
if (opt.use_cuda ~= 0) then
   criterion:cuda()
end

-- ****** model reading ******

print('reading in the model from file ' .. opt.model_file)
local model = torch.load(opt.model_file)
model:evaluate() -- turns off dropout in test mode

-- ******* validation function ******* (test function, copied every time from lent-and-returned-train.lua)

function test(input_table,gold_index_list,valid_batch_size,number_of_valid_batches,valid_set_size,left_out_samples,debug_file_prefix,guesses_file)

   local valid_batch_begin_index = 1
   local cumulative_loss = 0
   local cumulative_accuracy = 0
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
      batch_valid_input_representations_table,batch_valid_gold_index_tensor=
	 create_input_structures_from_table(input_table,
					    gold_index_list,
					    torch.range(valid_batch_begin_index,valid_batch_begin_index+valid_batch_size-1),
					    valid_batch_size,
					    t_input_size,
					    v_input_size,
					    opt.input_sequence_cardinality,
					    opt.candidate_cardinality,
					    opt.use_cuda)

      -- passing current test samples through the trained network
      local model_prediction=model:forward(batch_valid_input_representations_table)

      -- accumulate loss
      -- NB: according to documentation, the criterion function already normalizes loss!
      cumulative_loss = cumulative_loss + criterion:forward(model_prediction,batch_valid_gold_index_tensor)

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
      if debug_file_prefix then -- debug_file_prefix will be nil if debug mode is not on

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
end -- test function

-- ****** loading models, data handling functions ******

-- gbt: not necessary, right?
-- print('reading the models file')
-- dofile('model-new.lua')

print('reading the data processing file')
dofile('data-less-RAM.lua')

-- ****** input data reading ******

-- reading word embeddings
word_embeddings,t_input_size=load_embeddings(opt.word_embedding_file,opt.normalize_embeddings, opt.use_cuda)
--reading image embeddings
image_embeddings,v_input_size=load_embeddings(opt.image_embedding_file,opt.normalize_embeddings, opt.use_cuda)
-- reading in the test data
input_table,gold_index_list=
   create_data_tables_from_file(
      opt.test_file,
      opt.test_set_size,
      opt.input_sequence_cardinality,
      opt.candidate_cardinality)

-- *** computing model predictions and accuracy

print('computing model predictions and accuracy on test data')

local _,acc=
   test(input_table,gold_index_list,opt.test_set_size,1,opt.test_set_size,0,output_debug_prefix,output_guesses_file)

print('test set accuracy is ' .. acc)
