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

-- remove this backup once we have checked that function below works
function create_input_structures_from_file_old(i_file,data_set_size,t_input_size,v_input_size,image_set_size)
   print('reading protocol file ' .. i_file)

   -- initializing the data structures to hold the data

   -- word_query_list is a data_set_sizext_input_size tensor holding 
   -- query word representations
   local word_query_list = torch.Tensor(data_set_size,t_input_size)

   -- image_set_list is an image_set_size table of data_set_sizex
   -- v_input_size tensors: the ith position of each of this
   -- tensors will contain an image representation supposed to be in
   -- the ith set (sets ordered as the corresponding words in the same
   -- positions of word_query_list)
   -- gbt: please explain better (set? sets ordered? do you mean image sequence position?)
   local image_set_list={}
   -- initializing the tensors with zeroes
   for i=1,image_set_size do
      table.insert(image_set_list,torch.Tensor(data_set_size,v_input_size):zero())
   end

   -- index_list contains, for each sample, the index of the correct
   -- image (the one corresponding to the word) into the corresponding
   -- ordered set of tensors in image_set_list 
   -- if a sample is deviant (with index 0 or -1), the corresponding
   -- index will be image_set_size+1, ONLY FOR MODELS THAT CAN HANDLE
   -- DEVIANT CASES!!!!

   local index_list = torch.Tensor(data_set_size)

   local f = io.input(i_file)
   local i=1 -- line counter
   while true do
      local lines, rest = f:read(BUFSIZE, "*line")
      if not lines then break end
      if rest then lines = lines .. rest .. '\n' end
      -- traversing current chunk line by line
      -- local i=1 -- line counter ; gbt: bug -- 
      for current_line in lines:gmatch("[^\n]+") do
	 -- the following somewhat cumbersome expression will remove
	 -- leading and trailing space, and load all data onto a table
	 current_data = current_line:gsub("^%s*(.-)%s*$", "%1"):split("[ \t]+") -- should be local ***
	 -- first field is word id, second field gold index, other
	 -- fields image ids
	 word_query_list[i]=word_embeddings[current_data[1]]
	 index_list[i]=current_data[2]
	 -- handling deviant cases
	 if index_list[i]<1 then
	    if (model_can_handle_deviance==1) then
--	    if ((opt.model=="ff_ref_with_summary") or 
--	       (opt.model=="ff_ref_deviance")) then
	       index_list[i]=  image_set_size+1
	    else
	       error('ERROR: chosen model does not support deviance: ' .. tostring(opt.model))
	    end
	 end
	 -- because there might be less images in current trial than
	 -- the maximum (determined by image size) we only replace the
	 -- 0s in the first n image_set_size tensors, where n is the number
	 -- of image indices in the current input row
	 local current_images_count = #current_data-2
	 for j=1,current_images_count do
	    local id_position=j+2
	    image_set_list[j][i]=image_embeddings[current_data[id_position]]
	 end
	 i=i+1
      end-- end for current_line
   end
   f.close()
   return word_query_list,image_set_list,index_list
end

-- DEV VERSION FROM HERE
function create_input_structures_from_file(i_file,data_set_size,t_input_size,v_input_size,image_set_size)
   print('reading protocol file ' .. i_file)

   -- initializing the data structures to hold the data

   -- word_query_list is a data_set_sizext_input_size tensor holding 
   -- query word representations
   local word_query_list = nil
   if (opt.modifier_mode==1) then
      word_query_list=torch.Tensor(data_set_size,t_input_size*2)
   else
   -- if we are in the experiment with the modifiers, we will
   -- concatenate modifier and head, so we must double the size
      word_query_list=torch.Tensor(data_set_size,t_input_size)
   end

   -- image_set_list is an image_set_size table of data_set_sizex
   -- v_input_size tensors: the ith position of each of this
   -- tensors will contain an image representation supposed to be in
   -- the ith set (sets ordered as the corresponding words in the same
   -- positions of word_query_list)
   -- if we have modifiers, size will be that of word+image embedding
   local image_set_list={}
   -- initializing the tensors with zeroes
   for i=1,image_set_size do
      if (opt.modifier_mode==1) then
	 table.insert(image_set_list,torch.Tensor(data_set_size,t_input_size+v_input_size):zero())
      else
	 table.insert(image_set_list,torch.Tensor(data_set_size,v_input_size):zero())
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
--      local i=1 -- line counter
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
   return word_query_list,image_set_list,non0_slots_count_list,index_list
end
-- DEV VERSION TO HERE

function create_input_structures_from_file_for_max_margin(i_file,data_set_size,t_input_size,v_input_size)
   print('reading protocol file ' .. i_file)

   -- the data will be structured as follows: a table of three
   -- tensors, one for all the queries, one for all the targets, one
   -- for all the confounders. For each sequence with n images, n-1
   -- tensor elements will be created (for sequence 'bun 3 hovel raft
   -- bun', we'll build {[bun_word_vector, bun_word_vector],
   -- [bun_image_vector, bun_image_vector], [hovel_image_vector,
   -- raft_image_vector]})
   
   local word_query_t = {}
   local target_image_t = {}
   local confounder_t = {}

   -- nimgs_list contains, for each sample, how many images there are
   local nimgs_list = torch.Tensor(data_set_size)

   -- index_list contains, for each sample, the index of the correct
   -- image (the one corresponding to the word) into the corresponding
   -- ordered set of tensors in image_set_list 
   -- if a sample is deviant (with index 0 or -1), the corresponding
   -- index will be image_set_size+1, ONLY FOR MODELS THAT CAN HANDLE
   -- DEVIANT CASES!!!!
   local index_list = torch.Tensor(data_set_size)
   
   -- need to create a 'virtual file' so I can set up the dimensionality of the tensors
   local f = io.input(i_file)
   local i=1 -- line (datapoint) counter
   local npairs=0 -- number of {target, confounder} pairs to be created
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
	 -- io.write(tostring(current_line .. "\n"))
	 -- example: accumulator	2	granddaughter_246169	accumulator_445171	gilt_439010	dairy_278492
	 local query=current_data[1]
	 local current_images_count = #current_data-2
	 nimgs_list[i]=current_images_count -- recording number of images in sequence
	 index_list[i]=current_data[2] -- recording index of the right image
	 local target_position=current_data[2]+2 -- vector of the image in position gold index + 2
	 local target_image=current_data[target_position]
	 for j=1,current_images_count do
	    local id_position=j+2
	    if id_position~=target_position then
	       npairs=npairs+1
	       word_query_t[npairs]=query
	       target_image_t[npairs]=target_image
	       confounder_t[npairs]=current_data[id_position]
	       -- io.write(word_query_t[npairs] .. ", " .. target_image_t[npairs] .. ", " .. confounder_t[npairs] .. "\n")
	    end
	 end
	 i=i+1
      end
   end
   f.close()

   
   -- initializing the data structures to hold the data

   local nseq = i-1 -- because we start by 1
   print("     total number of datapoints (corresponding to total images in " .. tostring(nseq) .. " sequences): " .. tostring(npairs))
   -- word_query_list is a npairs x t_input_size tensor holding query
   -- word representations
   local word_query_list = torch.Tensor(npairs,t_input_size)
   -- target_image_query_list is a npairs x v_input_size tensor
   -- holding target image representations
   local target_image_list=torch.Tensor(npairs,v_input_size)
   -- confounder_query_list is a npairs x v_input_size tensor holding
   -- confounder representations
   local confounder_list=torch.Tensor(npairs,v_input_size)

   for j=1,npairs do
      word_query_list[j]=word_embeddings[word_query_t[j]]
      target_image_list[j]=image_embeddings[target_image_t[j]]
      confounder_list[j]=image_embeddings[confounder_t[j]]
   end
   data_table={word_query_list,target_image_list,confounder_list}
   gold_predictions=torch.Tensor(npairs):zero()+1 -- tensor for gold predictions is just a long list of 1s (see models file)
   return data_table, gold_predictions, nimgs_list, index_list -- nimgs_list is only needed for testing mode, but we return it always

end

-- REAL DATA TO HERE
