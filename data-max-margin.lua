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
	 local target_position=current_data[2]+2 -- vector of the image in position gold index + 2
	 local target_image=current_data[target_position]
	 local current_images_count = #current_data-2
	 nimgs_list[i]=current_images_count -- recording number of images in sequence
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
   return data_table, gold_predictions, nimgs_list -- nimgs_list is only needed for testing mode, but we return it always
end
