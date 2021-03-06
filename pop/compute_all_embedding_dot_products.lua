-- preambles

require('nngraph')
require('../LinearNB') -- for linear mappings without bias

cmd = torch.CmdLine()
-- model file to be read
cmd:option('--model_file','', 'name of file storing trained model generated by main.lua')
-- data files
cmd:option('--word_embedding_file','','word embedding file (with word vectors; first field word, rest of the fields vector values)... must be coherent with representations used at model training!')
cmd:option('--image_embedding_file','','image embedding file (with visual vectors; first field word and image, rest of the fields vector values)... must be coherent with representations used at model training!')
cmd:option('--test_file','','format: one stimulus set per line: first field linguistic referring expression (RE), second field index of the right image for the RE in the image set, rest of the fields image set (n indices of the images in the image dataset')
cmd:option('--test_set_size',10, 'test set size')
cmd:option('--output_word_dot_products_file','','to print word embedding dot product matrix')
cmd:option('--output_image_dot_products_file','','to print image embedding dot product matrix')
opt = cmd:parse(arg or {})
print(opt)

-- other general parameters
-- chunks to read files into
BUFSIZE = 2^23 -- 1MB
model_needs_real_image_count=1

print('reading the data processing file')
dofile('data.lua')

print('preparing data')

-- reading word embeddings
word_embeddings,t_input_size=
   load_embeddings(opt.word_embedding_file,1)
--reading image embeddings
image_embeddings,v_input_size=
   load_embeddings(opt.image_embedding_file,1)
test_input_table, test_index_list=
   create_input_structures_from_file(
      opt.test_file,
      opt.test_set_size,
      t_input_size,
      v_input_size,
      5)

print('reading in the model')
model = torch.load(opt.model_file)

print('computing model prediction on test data')
-- passing all test samples through the trained network
local model_prediction=model:forward(test_input_table)

local nodes = model:listModules()[1]['forwardnodes']

if (opt.output_word_dot_products_file ~='') then
   print('computing word embedding similarities')
      
   local raw_embeddings = nil

   for _,node in ipairs(nodes) do
      if node.data.annotations.name=='query' then
	 raw_embeddings = node.data.module.output
      end
   end

   local embeddings=torch.Tensor(raw_embeddings:size()) 

   for i=1,raw_embeddings:size(1) do
      embeddings[i]=raw_embeddings[i]/torch.norm(raw_embeddings[i])
   end

   local similarity_matrix = embeddings*embeddings:t()  

   embeddings=nil

   local f = io.open(opt.output_word_dot_products_file,"w")

   for i=1,similarity_matrix:size(1) do
      for j=1,similarity_matrix:size(2) do
	 if j==1 then
	    f:write(similarity_matrix[i][j])
	 else
	    f:write(" ",similarity_matrix[i][j])
	 end
      end
      f:write("\n")
   end

   f:flush()
   f:close()

   similarity_matrix=nil
end


if (opt.output_image_dot_products_file ~='') then
   print('computing image embedding similarities')
      
   local raw_embeddings = nil

   for _,node in ipairs(nodes) do
      if node.data.annotations.name=='reference_vector_1' then
	 raw_embeddings = node.data.module.output
      end
   end

   local embeddings=torch.Tensor(raw_embeddings:size()) 

   for i=1,raw_embeddings:size(1) do
      embeddings[i]=raw_embeddings[i]/torch.norm(raw_embeddings[i])
   end

   local similarity_matrix = embeddings*embeddings:t()  

   embeddings=nil

   local f = io.open(opt.output_image_dot_products_file,"w")

   for i=1,similarity_matrix:size(1) do
      for j=1,similarity_matrix:size(2) do
	 if j==1 then
	    f:write(similarity_matrix[i][j])
	 else
	    f:write(" ",similarity_matrix[i][j])
	 end
      end
      f:write("\n")
   end

   f:flush()
   f:close()

   similarity_matrix=nil
end


