--[[
******* the following code is to prepare toy training *******
******* and validation data                           *******
--]]

-- TOY DATA FROM HERE

-- this function, given an index integer and a target dimensionality,
-- generates a dimensionality-dimensional vector with all zeroes except
-- for a 1 at the index position  
function generate_one_hot_vector(one_index,dimensionality)
   if one_index>dimensionality then
      print('index cannot be above vector dimensionality, setting it to max allowed value')
      one_index=dimensionality
   end
   local vector=torch.zeros(dimensionality)
   vector[one_index]=1
   return vector
end

function generate_easy_one_hot_toy_data(data_set_size,vocabulary_size,image_ones,image_set_size,min_filled_image_set_size)
   -- image dimensionality equals the number of elements in the
   -- image_ones vector
   local image_dimensionality = image_ones:size(1)

   -- initializing the data structures to hold the data

   -- word_query_list is a data_set_sizexvocabulary_size tensor holding
   -- one-hot representations of the words
   local word_query_list = torch.Tensor(data_set_size,vocabulary_size)

   -- image_set_list is an image_set_size table of data_set_sizex
   -- image_dimensionality tensors: the ith position of each of this
   -- tensors will contain an image representation supposed to be in
   -- the ith set (sets ordered as the corresponding words in the same
   -- positions of word_query_list)
   local image_set_list={}
   -- initializing the tensors with zeroes
   for i=1,image_set_size do
      table.insert(image_set_list,torch.Tensor(data_set_size,image_dimensionality):zero())
   end

   -- index_list contains, for each sample, the index of the correct
   -- image (the one corresponding to the word) into the corresponding
   -- ordered set of tensors in image_set_list
   local index_list = torch.Tensor(data_set_size)

   for i=1,data_set_size do
      -- pick a random word
      local current_word_index=math.random(vocabulary_size)
      word_query_list[i]=generate_one_hot_vector(current_word_index,vocabulary_size)
      -- decide how many items in the image set to actually fill with
      -- "images" (the rest will be padded with zeroes)
      local fill_up_to=math.random(min_filled_image_set_size,image_set_size)
      for j=1,fill_up_to do
	 local confounder_image_index=math.random(image_dimensionality)
	 -- avoid inserting the right image for now
	 if (confounder_image_index==current_word_index) then
	    confounder_image_index=confounder_image_index+1 -- note that we don't need to worry
	                                                    -- about going overboard, since there
	                                                    -- are more image indices than word indices
	                                                
	 end
	 local confounder_image_one=image_ones[confounder_image_index]
	 image_set_list[j][i]=generate_one_hot_vector(confounder_image_one,image_dimensionality)
      end
      -- now we replace a random image with the right one
      local right_image_position=math.random(fill_up_to)
      local right_image_one=image_ones[current_word_index]
      image_set_list[right_image_position][i]=generate_one_hot_vector(right_image_one,image_dimensionality)
      -- finally, we add the position of the right image to the index_list
      index_list[i]=right_image_position
   end
   return word_query_list,image_set_list,index_list
end

-- this function is currently hard-coded to work only by calling
-- generate_easy_one_hot_toy_data, which it calls twice, once to
-- generate training data, and once to generate validation data
function generate_toy_data(training_set_size,validation_set_size,t_input_size,image_set_size,min_filled_image_set_size)

   if (min_filled_image_set_size>image_set_size) then
      print('min_filled_image_set_size larger than image_set_size, resetting it')
      min_filled_image_set_size=image_set_size
   end

   -- images should have extra dimensionality of one with respect to
   -- words (because of the way we generate the image sets)
   local image_dimensionality=t_input_size+1

   -- generating a tensor of indices supposed to be the non-zero
   -- entries in the image vectors (note that we must do this here, so
   -- that it's shared across training and validation data)
   local image_ones = torch.randperm(image_dimensionality)

   print('generating toy training data')
   local training_word_query_list,training_image_set_list,training_index_list = 
     generate_easy_one_hot_toy_data(training_set_size,t_input_size,image_ones,image_set_size,min_filled_image_set_size)
   
   print('generating toy validation data')
   local validation_word_query_list,validation_image_set_list,validation_index_list = 
      generate_easy_one_hot_toy_data(validation_set_size,t_input_size,image_ones,image_set_size,min_filled_image_set_size)
 
   --NB routine also returns image dimensionality, which is expected to overwrite v_input_size
   return 
      training_word_query_list,training_image_set_list,training_index_list,
      validation_word_query_list,validation_image_set_list,validation_index_list,
      image_dimensionality

end

-- TOY DATA TO HERE

-- REAL DATA FROM HERE
function load_embeddings(i_file,normalize_embeddings)

   print('reading embeddings file ' .. i_file)
   local embeddings={}
   local current_data={}
   local f = io.input(i_file)
   while true do
      local lines, rest = f:read(BUFSIZE, "*line")
      if not lines then break end
      if rest then lines = lines .. rest .. '\n' end
      -- traversing current chunk line by line
      for current_line in lines:gmatch("[^\n]+") do
	 -- the following somewhat cumbersome expression will remove
	 -- leading and trailing space, and load all data onto a table
	 current_data = current_line:gsub("^%s*(.-)%s*$", "%1"):split("[ \t]+")
	 -- first field is id, other fields are embedding vector
	 embeddings[current_data[1]]=
	    torch.Tensor({unpack(current_data,2,#current_data)})
	 -- normalize if we are asked to
	 if (normalize_embeddings>0) then
	    local embedding_norm = torch.norm(embeddings[current_data[1]])
	    -- avoid dividing by 0
	    if (embedding_norm~=0) then
	       embeddings[current_data[1]]=
		  embeddings[current_data[1]]/embedding_norm
	    end
	 end
      end
   end
   f.close()
   -- recording word embedding dimensionality
   local input_size = #current_data-1
   return embeddings,input_size
end

function create_input_structures_from_file(i_file,data_set_size,t_in_size,v_in_size,image_set_size)
   local output_table = {}
   local index_list = nil
   local nconfounders_list = nil
   local tuples_start_at_list = nil
   local tuples_end_at_list = nil
   if opt.model=='max_margin_bl' then
      output_table, index_list,
      nconfounders_list, tuples_start_at_list, tuples_end_at_list=
	 create_input_structures_from_file_for_max_margin(i_file,data_set_size,t_in_size,v_in_size)
   else
      output_table, index_list=
	 create_input_structures_from_file_for_other_models(i_file,data_set_size,t_in_size,v_in_size,image_set_size)
   end
   return output_table, index_list, nconfounders_list, tuples_start_at_list, tuples_end_at_list
end

function create_input_structures_from_file_for_other_models(i_file,data_set_size,t_in_size,v_in_size,image_set_size)
   print('reading protocol file ' .. i_file)

   -- initializing the data structures to hold the data
   local output_table = {} -- to put data tensors in (will be model input)

   -- word_query_list is a data_set_size x t_in_size tensor holding 
   -- query word representations
   local word_query_list = nil
   -- if we are in the experiment with the modifiers, we will
   -- concatenate modifier and head, so we must double the size
   if (opt.modifier_mode==1) then
      word_query_list=torch.Tensor(data_set_size,t_in_size*2)
   else
      word_query_list=torch.Tensor(data_set_size,t_in_size)
   end

   -- image_set_list is an image_set_size table of data_set_sizex
   -- v_in_size tensors: the ith position of each of this
   -- tensors will contain an image representation supposed to be in
   -- the ith set (sets ordered as the corresponding words in the same
   -- positions of word_query_list)
   -- if we have modifiers, size will be that of word+image embedding
   local image_set_list={}
   -- initializing the tensors with zeroes
   for i=1,image_set_size do
      if (opt.modifier_mode==1) then
	 table.insert(image_set_list,torch.Tensor(data_set_size,t_in_size+v_in_size):zero())
      else
	 table.insert(image_set_list,torch.Tensor(data_set_size,v_in_size):zero())
      end
   end

   -- non0_slots_count_list contains, for each sample, the number of real
   -- images that were passed in input (this is only needed by certain models)
   local non0_slots_count_list = nil
   if (model_needs_real_image_count==1) then
      non0_slots_count_list = torch.Tensor(data_set_size)
   end

   -- index_list contains, for each sample, the index of the correct
   -- image (the one corresponding to the word) into the corresponding
   -- ordered set of tensors in image_set_list 
   -- if a sample is deviant (with index 0 or -1), the corresponding
   -- index will be image_set_size+1, ONLY FOR MODELS THAT CAN HANDLE
   -- DEVIANT CASES!!!!

   local index_list = torch.Tensor(data_set_size)

   local f = io.input(i_file)
   local i=1
   while true do
      local lines, rest = f:read(BUFSIZE, "*line")
      if not lines then break end
      if rest then lines = lines .. rest .. '\n' end
      -- traversing current chunk line by line
      for current_line in lines:gmatch("[^\n]+") do
	 -- the following somewhat cumbersome expression will remove
	 -- leading and trailing space, and load all data onto a table
	 local current_data = current_line:gsub("^%s*(.-)%s*$", "%1"):split("[ \t]+")
	 -- first field is word id (or modifier:word id if in modifier mode), second field gold index, other
	 -- fields image ids (or modifier:image ids if in modifier mode)
	 if (opt.modifier_mode==1) then
	    local modifier_head=current_data[1]:split(":")
	    word_query_list[i]=torch.cat(word_embeddings[modifier_head[1]],word_embeddings[modifier_head[2]],1)
	 else
	    word_query_list[i]=word_embeddings[current_data[1]]
	 end
	 index_list[i]=current_data[2]
	 -- handling deviant cases: we don't distinguish btw 0 and -1, and we assign image_set_size+1
	 if index_list[i]<1 then
	    if (model_can_handle_deviance==1) then
	       index_list[i]=  image_set_size+1
	    else
	       error('ERROR: chosen model does not support deviance: ' .. tostring(opt.model))
	    end
	 end
	 -- because there might be less images in current trial than
	 -- the maximum (determined by image size) we only replace the
	 -- 0s in the first n image_set_size tensors, where n is the number
	 -- of image indices in the current input row
	 -- again, we must consider possibility that image has modifier
	 local current_images_count = #current_data-2
	 for j=1,current_images_count do
	    local id_position=j+2
	    if (opt.modifier_mode==1) then
	       local modifier_head=current_data[id_position]:split(":")
	       image_set_list[j][i]=torch.cat(word_embeddings[modifier_head[1]],image_embeddings[modifier_head[2]],1)
	    else
	       image_set_list[j][i]=image_embeddings[current_data[id_position]]
	    end
	 end
	 -- we also keep track of real image count in non0_slots_count_list, if needed
	 if (model_needs_real_image_count==1) then
	    non0_slots_count_list[i] = current_images_count
	 end
	 i=i+1
      end
   end
   f.close()

   -- only if model is using this info, we also pass number of real images
   if model_needs_real_image_count==1 then
      table.insert(output_table,
		   non0_slots_count_list:resize(index_list:size(1),1))
   end
   table.insert(output_table,word_query_list)
   for j=1,opt.image_set_size do
      table.insert(output_table,image_set_list[j])
   end
   return output_table, index_list
end

function unpack_for_max_margin(indices,nconfounders,data,start_at_list,end_at_list) -- 
   -- go from the indices to the corresponding slices in the data tuples
   -- returns {query, target, confounder}-tensor table for input to max margin bl, and 

   local ori_set_size=indices:size()[1] -- set size in number of sequences
   local nconf_list_out=nconfounders:index(1,indices) -- how many confounders there are in each of the output sequences (e.g. batch sequences)
   local new_set_size=nconf_list_out:sum() -- set size in number of tuples (model input)
   local start_at_list_out=start_at_list:index(1,indices) -- start of relevant tuples in data
   local end_at_list_out=end_at_list:index(1,indices) -- end of relevant tuples in data
   
   local output_indices=torch.Tensor(new_set_size):long() -- the indices of the tuples we want (long for compatibility with index function)
   local old=1; local new=0
   for i=1,ori_set_size do
      new=old+nconf_list_out[i]
      output_indices[{{old,new-1}}]=torch.range(start_at_list_out[i],end_at_list_out[i])  -- -1 cause we stop just before the new starting point
      old=new -- next time we start where we left off
      i=i+1
   end

   -- print('output_indices:')
   -- print(output_indices)
   local output_tuples={} -- {q_sequence,t_sequence,c_sequence}
   for j=1,3 do
      output_tuples[j]=data[j]:index(1,output_indices)
      -- print('output sizes ' .. tostring(j))
      -- print(output_tuples[j]:size())
   end
   -- print(output_tuples)
   return output_tuples,new_set_size

end
   -- -- old, to debug function unpack...
   -- print('ori_set_size:')
   -- print(ori_set_size)
   -- print('new_set_size:')
   -- print(new_set_size)
      -- print(tostring('---'))
      -- print(tostring('i:'))
      -- print(tostring(i))
      -- print('start_at:')
      -- print(start_at_list_out[i])
      -- print('end_at:')
      -- print(end_at_list_out[i])
      -- print('nelem:')
      -- print(nconf_list_out[i])
      -- print('new:')
      -- print(new)
      -- print(tostring('---'))
      -- print(output_tuples[j][{{},{1,5}}])

function create_input_structures_from_file_for_max_margin(i_file,data_set_size,t_input_size,v_input_size)
   print('reading protocol file ' .. i_file)

   -- the data will be structured as follows: a table of three
   -- tensors, one for all the queries, one for all the targets, one
   -- for all the confounders. For each sequence with n images, n-1
   -- tensor elements will be created (for sequence 'bun 3 hovel raft
   -- bun', we'll build {[bun_word_vector, bun_word_vector],
   -- [bun_image_vector, bun_image_vector], [hovel_image_vector,
   -- raft_image_vector]})
   
   -- nconfounders_list contains, for each sample, how many
   -- confounders there are (= number of images -1 for target;
   -- corresponds to ntuples created)
   local nconfounders_list = torch.Tensor(data_set_size)
   local tuples_start_at_list = torch.Tensor(data_set_size) -- where in the model input (tuples) the tuples for the current sequence start
   local tuples_end_at_list = torch.Tensor(data_set_size) -- where in the model input (tuples) the tuples for the current sequence end

   -- idx_list contains, for each sample, the index of the correct
   -- image (the one corresponding to the word) into the corresponding
   -- ordered set of tensors in image_set_list
   -- if a sample is deviant (with index 0 or -1), the corresponding
   -- index will be image_set_size+1, ONLY FOR MODELS THAT CAN HANDLE
   -- DEVIANT CASES!!!!
   local idx_list = torch.Tensor(data_set_size)
   
   local word_query_t = {}
   local target_image_t = {}
   local confounder_t = {}
   -- need to create a 'virtual file' so I can set up the dimensionality of the tensors
   local f = io.input(i_file)
   local i=1 -- line (datapoint) counter
   local ntuples=1 -- number of {query, target, confounder} tuples to be created
   while true do
      local lines, rest = f:read(BUFSIZE, "*line")
      if not lines then break end
      if rest then lines = lines .. rest .. '\n' end
      -- traversing current chunk line by line
      for current_line in lines:gmatch("[^\n]+") do
	 -- the following somewhat cumbersome expression will remove
	 -- leading and trailing space, and load all data onto a table
	 local current_data = current_line:gsub("^%s*(.-)%s*$", "%1"):split("[ \t]+")
	 -- first field is word id, second field gold index, other
	 -- fields image ids
	 -- print(tostring('---'))
	 -- io.write(tostring(current_line .. "\n"))
	 -- example: accumulator	2	granddaughter_246169	accumulator_445171	gilt_439010	dairy_278492
	 tuples_start_at_list[i]=ntuples
	 local query=current_data[1]
	 local current_images_count = #current_data-2
	 nconfounders_list[i]=current_images_count-1 -- recording number of confounders in sequence; = number of images -1 (target)
	 idx_list[i]=current_data[2] -- recording index of the right image
	 local target_position=current_data[2]+2 -- vector of the image in position gold index + 2
	 local target_image=current_data[target_position]
	 for j=1,current_images_count do
	    local id_position=j+2
	    if id_position~=target_position then
	       word_query_t[ntuples]=query
	       target_image_t[ntuples]=target_image
	       confounder_t[ntuples]=current_data[id_position]
	       -- io.write(word_query_t[ntuples] .. ", " .. target_image_t[ntuples] .. ", " .. confounder_t[ntuples] .. "\n")
	       ntuples=ntuples+1
	    end
	 end
	 tuples_end_at_list[i]=ntuples-1 -- because we add one after processing all images
	 -- io.write(nconfounders_list[i] .. ", " .. tuples_start_at_list[i] .. ", " .. tuples_end_at_list[i] .. "\n")
	 i=i+1
      end
   end
   f.close()
   ntuples=ntuples-1 -- -1 because we start by 1 / add one
   print("     total number of datapoints (corresponding to total images in " .. tostring(i-1) .. " sequences): " .. tostring(ntuples))

   -- initializing the data structures to hold the data
   
   -- word_query_list is a ntuples x t_input_size tensor holding query
   -- word representations
   local word_query_list = torch.Tensor(ntuples,t_input_size)
   -- target_image_query_list is a ntuples x v_input_size tensor
   -- holding target image representations
   local target_image_list=torch.Tensor(ntuples,v_input_size)
   -- confounder_query_list is a ntuples x v_input_size tensor holding
   -- confounder representations
   local confounder_list=torch.Tensor(ntuples,v_input_size)
   for j=1,ntuples do
      -- io.write(word_query_t[j] .. ", " .. target_image_t[j] .. ", " .. confounder_t[j] .. "\n")
      word_query_list[j]=word_embeddings[word_query_t[j]]
      target_image_list[j]=image_embeddings[target_image_t[j]]
      confounder_list[j]=image_embeddings[confounder_t[j]]
      -- print(tostring('---'))
      -- print(word_query_list[{{j},{1,5}}])
      -- print(target_image_list[{{j},{1,5}}])
      -- print(confounder_list[{{j},{1,5}}])
      -- print(tostring('---'))
   end
   data_table={word_query_list,target_image_list,confounder_list}
   return data_table, idx_list, nconfounders_list, tuples_start_at_list, tuples_end_at_list
end

-- REAL DATA TO HERE

function dummy()
   if opt.model=='max_margin_bl' then
   --   set_size=nconfounders:size(1)
      print('piiip')
   end
end
