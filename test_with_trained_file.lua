-- preambles

require('nngraph')
require('LinearNB') -- for linear mappings without bias

cmd = torch.CmdLine()
-- model file to be read
cmd:option('--model_file','', 'name of file storing trained model generated by main.lua')
-- data files
cmd:option('--word_embedding_file','','word embedding file (with word vectors; first field word, rest of the fields vector values)... must be coherent with representations used at model training!')
cmd:option('--image_embedding_file','','image embedding file (with visual vectors; first field word and image, rest of the fields vector values)... must be coherent with representations used at model training!')
cmd:option('--image_set_size', 0, 'max number of images in a set (must be coherent with max size used at model training!')
cmd:option('--normalize_embeddings',0, 'whether to normalize word and image representations, set to 1 to normalize: must be coherent with choice at model training')
cmd:option('--test_file','','format: one stimulus set per line: first field linguistic referring expression (RE), second field index of the right image for the RE in the image set, rest of the fields image set (n indices of the images in the image dataset')
cmd:option('--test_set_size',10, 'test set size')
-- we need to know the model that was used for training, since this affects
-- oov handling
local mst = {ff_ref=true, max_margin_bl=true, ff_ref_with_summary=true, ff_ref_deviance=true, ff_ref_sim_sum=true}
local msg='model, to choose from: '
for k, _ in pairs(mst) do msg = msg .. k .. ', ' end
cmd:option('--model','ff_ref', msg)
-- output file
cmd:option('--output_guesses_file','','if this file is defined, we print to it, as separated space-delimited columns, the index the model returned as its guess for each test item, and the corresponding log probability')
-- other options
cmd:option('--debug',0,'set to 1 to go through code flagged as for debugging')
opt = cmd:parse(arg or {})
print(opt)

local output_guesses_file=nil
if opt.output_guesses_file~='' then
   output_guesses_file=opt.output_guesses_file
end

-- other general parameters
-- chunks to read files into
BUFSIZE = 2^23 -- 1MB

-- here, list models that can handle deviance, for appropriate data
-- reading
model_can_handle_deviance=0
if ((opt.model=="ff_ref_with_summary") or 
   (opt.model=="ff_ref_deviance") or
   (opt.model=="ff_ref_sim_sum")) then
   model_can_handle_deviance=1
end


print('reading the data processing file')
dofile('data.lua')

print('preparing data')

-- readng word embeddings
word_embeddings,t_input_size=
   load_embeddings(opt.word_embedding_file,opt.normalize_embeddings)
--reading image embeddings
image_embeddings,v_input_size=
   load_embeddings(opt.image_embedding_file,opt.normalize_embeddings)
-- lodading test data
test_word_query_list,
test_image_set_list,
test_index_list= create_input_structures_from_file(opt.test_file,
						   opt.test_set_size,
						   t_input_size,
						   v_input_size,
						   opt.image_set_size)
print('reading in the model')
model = torch.load(opt.model_file)

print('computing model prediction on test data')

-- passing all test samples through the trained network
local model_prediction=model:forward({test_word_query_list,unpack(test_image_set_list)})

-- to compute accuracy, we first retrieve list of indices of image
-- vectors that were preferred by the model
local model_max_log_probs,model_guesses=torch.max(model_prediction,2)
local model_max_probs=torch.exp(model_max_log_probs)
-- we then count how often this guesses are the same as the gold
-- (and thus the difference is 0) (note conversions to long because
-- model_guesses is long tensor)
local hit_count = torch.sum(torch.eq(test_index_list:long(),model_guesses))
-- normalizing accuracy by test set size
local accuracy=hit_count/opt.test_set_size

print('test set accuracy is ' .. accuracy)

--if requested, print guesses, their probs and the overall prob distribution 
--to file
if output_guesses_file then
   local all_probs=torch.exp(model_prediction)
   local f = io.open(output_guesses_file,"w")
   for i=1,model_max_probs:size(1) do
      f:write(model_guesses[i][1]," ",model_max_probs[i][1])
      for j=1,all_probs:size(2) do
	 f:write(" ",all_probs[i][j])
      end
      f:write("\n")
   end
   f:flush()
   f.close()
end


