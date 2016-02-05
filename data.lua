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
   -- tensors will contain an image representatio supposed to be in
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
function load_embeddings(i_file)

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
      end
   end
   f.close()
   -- recording word embedding dimensionality
   local input_size = #current_data-1
   return embeddings,input_size
end

function create_input_structures_from_file(i_file,data_set_size,t_input_size,v_input_size,image_set_size)
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
   local image_set_list={}
   -- initializing the tensors with zeroes
   for i=1,image_set_size do
      table.insert(image_set_list,torch.Tensor(data_set_size,v_input_size):zero())
   end

   -- index_list contains, for each sample, the index of the correct
   -- image (the one corresponding to the word) into the corresponding
   -- ordered set of tensors in image_set_list
   local index_list = torch.Tensor(data_set_size)

   local f = io.input(i_file)
   local i=1
   while true do
      local lines, rest = f:read(BUFSIZE, "*line")
      if not lines then break end
      if rest then lines = lines .. rest .. '\n' end
      -- traversing current chunk line by line
      local i=1 -- line counter
      for current_line in lines:gmatch("[^\n]+") do
	 -- the following somewhat cumbersome expression will remove
	 -- leading and trailing space, and load all data onto a table
	 current_data = current_line:gsub("^%s*(.-)%s*$", "%1"):split("[ \t]+")
	 -- first field is word id, second field gold index, other
	 -- fields image ids
	 word_query_list[i]=word_embeddings[current_data[1]]
	 index_list[i]=current_data[2]
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
      end
   end
   f.close()
   return word_query_list,image_set_list,index_list
end

-- REAL DATA TO HERE

--[===[
OLD
function load_data(word_lexicon_file,image_dataset_file,protocol_prefix)
   --
   print("Hi there!")
--   print(word_lexicon_file)
   -- -- tensor of size 
   -- local training_word_query_list=torch.randn(training_set_size,t_input_size)
   -- local validation_word_query_list=torch.randn(validation_set_size,t_input_size)
   
   -- return training_word_query_list,validation_word_query_list,training_image_set_list,validation_image_set_list,training_index_list,validation_index_list,training_set_size,validation_set_size,image_set_size,t_input_size,v_input_size
      return true
end
--]===]

