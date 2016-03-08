function create_input_structures_from_file_with_deviants(i_file,data_set_size,t_input_size,v_input_size,image_set_size)
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
   -- if a sample is deviant (with index 0 or -1), the corresponding index
   -- will be image_set_size+1
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
	 current_data = current_line:gsub("^%s*(.-)%s*$", "%1"):split("[ \t]+") -- should be local *************
	 -- first field is word id, second field gold index, other
	 -- fields image ids
	 word_query_list[i]=word_embeddings[current_data[1]]
	 index_list[i]=current_data[2]
	 -- handling deviant cases
	 if index_list[i]<1 then
	    index_list[i]=  image_set_size+1
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
      end
   end
   f.close()
   return word_query_list,image_set_list,index_list
end
