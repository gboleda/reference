-- returns an embeddings table and the dimensionality of the
-- embeddings
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

-- returns output_table, containing a set of n x embeddings_dim tensors
-- each of which has the data for one input trial per row (n is the
-- number of trials, embeddings_dim changes depending on the nature of
-- the corresponding input data: words vs images), as well as a nx1
-- tensor with the gold indices
function create_input_structures_from_file(i_file,data_set_size,t_in_size,v_in_size,input_sequence_cardinality,candidate_cardinality)
   print('reading protocol file ' .. i_file)

   -- initializing the data structures to hold the data
   local output_table = {} -- to put data tensors in (will be model input)

   -- for the query, we need 3 tensors, two for the attributes, one
   -- for the object name: each of them is data_set_size x t_in_size
   -- here and below, we initialize the tensors to 0, so if an item is
   -- missing, we will simply use the 0 vector as its representation
   local query_att1_list = torch.Tensor(data_set_size,t_in_size):fill(0)
   local query_att2_list = torch.Tensor(data_set_size,t_in_size):fill(0)
   local query_object_list = torch.Tensor(data_set_size,t_in_size):fill(0)

   -- input_sequence_list is an 2*input_sequence_cardinality table of
   -- alternating data_set_size x t_in_size and data_set_size x
   -- v_in_size tensors: the odd tensors will contain attribute
   -- representations, the even slots will contain image
   -- representations, with each pair of consecutive tensors standing
   -- for an object token (pairs ordered as the object tokens in the
   -- corresponding stimulus); the entries in the input_sequence_list
   -- table are aligned with the order of the queries in the query
   -- lists
   local input_sequence_list={}
   for i=1,input_sequence_cardinality do
      table.insert(input_sequence_list,torch.Tensor(data_set_size,t_in_size):fill(0))
      table.insert(input_sequence_list,torch.Tensor(data_set_size,v_in_size):fill(0))
   end

   -- output_sequence_list is a candidate_cardinality table of
   -- data_set_size x v_in_size tensors containing the set of images the
   -- model will have to choose from
   local output_sequence_list={}
   for i=1,candidate_cardinality do
      table.insert(output_sequence_list,torch.Tensor(data_set_size,v_in_size):fill(0))
   end

   
   -- gold_index_list contains, for each sample, the index of correct
   -- candidate image (the index of the image in the output set
   -- corresponding to the composite meaning denoted by the linguistic
   -- query) in the corresponding sequence of tensors in
   -- output_sequence_list
   local gold_index_list = torch.Tensor(data_set_size)

   -- now we traverse the trial file, expected to be in format:
   --
   -- query_object:query_att1:query_att2 gold_index || cand_img_1 ... cand_img_L || seq_att1:seq_obj1
   -- ... seq_attM:seq:objM
   -- where L is candidate_cardinality and M is input_sequence_cardinality
   local f = io.input(i_file)
   local i=1
   local missing_embedding_flag=0
   while true do
      local lines, rest = f:read(BUFSIZE, "*line")
      if not lines then break end
      if rest then lines = lines .. rest .. '\n' end
      -- traversing current chunk line by line
      for current_line in lines:gmatch("[^\n]+") do
	 -- the following somewhat cumbersome expression will remove
	 -- leading and trailing space, and load all data onto a table
	 local current_data = current_line:gsub("^%s*(.-)%s*$", "%1"):split("[ \t]+")
	 -- query will contain object, att1, att2, lets retrieve their
	 -- word embeddings, if any
	 local query=current_data[1]:split(":")
	 if (word_embeddings[query[1]]~=nil) then
	    query_object_list[i]=word_embeddings[query[1]]
	 else
	    missing_embedding_flag=1
	    -- debug
--	    print('missing embedding for item query 1 ' .. query[1])
	    -- debug
	 end
	 if (word_embeddings[query[2]]~=nil) then
	    query_att1_list[i]=word_embeddings[query[2]]
	 else
	    missing_embedding_flag=1
	    -- debug
--	    print('missing embedding for item query 2 ' .. query[2])
	    -- debug
	 end
	 if (word_embeddings[query[3]]~=nil) then
	    query_att2_list[i]=word_embeddings[query[3]]
	 else
	    missing_embedding_flag=1
	    -- debug
--	    print('missing embedding for item query 3 ' .. query[3])
	    -- debug
	 end
	 -- append gold index to corresponding list
	 gold_index_list[i]=current_data[2]

	 -- we ignore current_data[3] as it's a pingie-pingie and move on to the following
	 -- sequence of candidate images
	 local tensor_counter = 0
	 for j=4,candidate_cardinality+3 do
	    local candidate_image=current_data[j]
	    tensor_counter = tensor_counter + 1
	    if (image_embeddings[candidate_image]~=nil) then
	       output_sequence_list[tensor_counter][i]=image_embeddings[candidate_image]
	    else
	       missing_embedding_flag=1
	       -- debug
--	       print('missing embedding for item candidate_image' .. candidate_image)
	       -- debug
	    end
	 end

	 -- skipping another pingie pingie in position candidate_cardinality+4
	 
	 -- finally, we update each tensor in input_sequence_list with
	 -- alternating word att and image object embeddings
	 tensor_counter = 0
	 local start_at = candidate_cardinality+5
	 local end_at = start_at+input_sequence_cardinality-1
	 for j=start_at,end_at do
	    -- object token contains attr, object
	    local object_token=current_data[j]:split(":")
	    tensor_counter = tensor_counter + 1
	    if (word_embeddings[object_token[1]]~=nil) then
	       input_sequence_list[tensor_counter][i]=word_embeddings[object_token[1]]
	    else
	       missing_embedding_flag=1
	       -- debug
--	       print('missing embedding for item object_token 1 ' .. object_token[1])
	       -- debug
	    end
	    tensor_counter = tensor_counter + 1
	    if (image_embeddings[object_token[2]]~=nil) then
	       input_sequence_list[tensor_counter][i]=image_embeddings[object_token[2]]
	    else
	       missing_embedding_flag=1
	       -- debug
--	       print('missing embedding for item object_token 2 ' .. object_token[2])
	       -- debug
	    end
	 end
	 i=i+1
      end
   end
   f.close()

   if (missing_embedding_flag==1) then
      print('not all items had an entry in the embeddings tables')
   end

   table.insert(output_table,query_att1_list)
   table.insert(output_table,query_att2_list)
   table.insert(output_table,query_object_list)
   for j=1,(input_sequence_cardinality*2) do
      table.insert(output_table,input_sequence_list[j])
   end
   for j=1,candidate_cardinality do
      table.insert(output_table,output_sequence_list[j])
   end
   return output_table, gold_index_list
end


-- DEPRECATED FROM HERE TO END OF FILE

-- debug 1: create_input_structures_from_file_1att
-- version that expects only one attribute
-- returns output_table, containing a set of n x embeddings_dim tensors
-- each of which has the data for one input trial per row (n is the
-- number of trials, embeddings_dim changes depending on the nature of
-- the corresponding input data: words vs images), as well as a nx1
-- tensor with the gold indices
function create_input_structures_from_file_1att(i_file,data_set_size,t_in_size,v_in_size,input_sequence_cardinality,candidate_cardinality)
   print('reading protocol file ' .. i_file)

   -- initializing the data structures to hold the data
   local output_table = {} -- to put data tensors in (will be model input)

   -- for the query, we need 3 tensors, two for the attributes, one
   -- for the object name: each of them is data_set_size x t_in_size
   local query_att1_list = torch.Tensor(data_set_size,t_in_size)
   -- since we're not using the second attribute, we fill it in with zeroes
   local query_att2_list = torch.Tensor(data_set_size,t_in_size):fill(0)
   local query_object_list = torch.Tensor(data_set_size,t_in_size)

   -- input_sequence_list is an 2*input_sequence_cardinality table of
   -- alternating data_set_size x t_in_size and data_set_size x
   -- v_in_size tensors: the odd tensors will contain attribute
   -- representations, the even slots will contain image
   -- representations, with each pair of consecutive tensors standing
   -- for an object token (pairs ordered as the object tokens in the
   -- corresponding stimulus); the entries in the input_sequence_list
   -- table are aligned with the order of the queries in the query
   -- lists
   local input_sequence_list={}
   for i=1,input_sequence_cardinality do
      table.insert(input_sequence_list,torch.Tensor(data_set_size,t_in_size))
      table.insert(input_sequence_list,torch.Tensor(data_set_size,v_in_size))
   end

   -- output_sequence_list is a candidate_cardinality table of
   -- data_set_size x v_in_size tensors containing the set of images the
   -- model will have to choose from
   local output_sequence_list={}
   for i=1,candidate_cardinality do
       table.insert(output_sequence_list,torch.Tensor(data_set_size,v_in_size))
   end

   
   -- gold_index_list contains, for each sample, the index of correct
   -- candidate image (the index of the image in the output set
   -- corresponding to the composite meaning denoted by the linguistic
   -- query) in the corresponding sequence of tensors in
   -- output_sequence_list
   local gold_index_list = torch.Tensor(data_set_size)

   -- now we traverse the trial file, expected to be in format:
   --
   -- query_object:query_att1 gold_index || cand_img_1 ... cand_img_L || seq_att1:seq_obj1
   -- ... seq_attM:seq:objM
   -- where L is candidate_cardinality and M is input_sequence_cardinality
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
	 -- query will contain object, att1 lets retrieve their
	 -- word embeddings
	 local query=current_data[1]:split(":")
	 query_object_list[i]=word_embeddings[query[1]]
	 query_att1_list[i]=word_embeddings[query[2]]

	 -- append gold index to corresponding list
	 gold_index_list[i]=current_data[2]

	 -- we ignore current_data[3] as it's a pingie-pingie and move on to the following
	 -- sequence of candidate images
	 local tensor_counter = 0
	 for j=4,candidate_cardinality+3 do
	    local candidate_image=current_data[j]
	    tensor_counter = tensor_counter + 1
	    output_sequence_list[tensor_counter][i]=image_embeddings[candidate_image]
	 end


	 -- skipping another pingie pingie in position candidate_cardinality+4
	 
	 -- finally, we update each tensor in input_sequence_list with
	 -- alternating word att and image object embeddings
	 tensor_counter = 0
	 local start_at = candidate_cardinality+5
	 local end_at = start_at+input_sequence_cardinality-1
	 for j=start_at,end_at do
	    -- object token contains attr, object
	    local object_token=current_data[j]:split(":")
	    tensor_counter = tensor_counter + 1
	    input_sequence_list[tensor_counter][i]=word_embeddings[object_token[1]]
	    tensor_counter = tensor_counter + 1
	    input_sequence_list[tensor_counter][i]=image_embeddings[object_token[2]]
	 end
	 i=i+1
      end
   end
   f.close()

   table.insert(output_table,query_att1_list)
   table.insert(output_table,query_att2_list)
   table.insert(output_table,query_object_list)
   for j=1,(input_sequence_cardinality*2) do
      table.insert(output_table,input_sequence_list[j])
   end
   for j=1,candidate_cardinality do
      table.insert(output_table,output_sequence_list[j])
   end
   return output_table, gold_index_list
end


-- debug 2: create_input_structures_from_file_no_atts
function create_input_structures_from_file_no_atts(i_file,data_set_size,t_in_size,v_in_size,input_sequence_cardinality,candidate_cardinality)
   print('reading protocol file ' .. i_file)

   -- initializing the data structures to hold the data
   local output_table = {} -- to put data tensors in (will be model input)

   -- for the query, we need 3 tensors, two for the attributes, one
   -- for the object name: each of them is data_set_size x t_in_size
   -- since we're not using the attributes, we fill them with zeroes
   local query_att1_list = torch.Tensor(data_set_size,t_in_size):fill(0)
   local query_att2_list = torch.Tensor(data_set_size,t_in_size):fill(0)
   local query_object_list = torch.Tensor(data_set_size,t_in_size)

   -- input_sequence_list is an 2*input_sequence_cardinality table of
   -- alternating data_set_size x t_in_size and data_set_size x
   -- v_in_size tensors: the odd tensors will contain attribute
   -- representations, THAT HERE WE JUST INITIALIZE TO 0 AND WON'T
   -- TOUCH, the even slots will contain image representations, with
   -- each pair of consecutive tensors standing for an object token
   -- (pairs ordered as the object tokens in the corresponding
   -- stimulus); the entries in the input_sequence_list table are
   -- aligned with the order of the queries in the query lists
   local input_sequence_list={}
   for i=1,input_sequence_cardinality do
      table.insert(input_sequence_list,torch.Tensor(data_set_size,t_in_size):fill(0))
      table.insert(input_sequence_list,torch.Tensor(data_set_size,v_in_size))
   end

   -- output_sequence_list is a candidate_cardinality table of
   -- data_set_size x v_in_size tensors containing the set of images the
   -- model will have to choose from
   local output_sequence_list={}
   for i=1,candidate_cardinality do
       table.insert(output_sequence_list,torch.Tensor(data_set_size,v_in_size))
   end

   
   -- gold_index_list contains, for each sample, the index of correct
   -- candidate image (the index of the image in the output set
   -- corresponding to the composite meaning denoted by the linguistic
   -- query) in the corresponding sequence of tensors in
   -- output_sequence_list
   local gold_index_list = torch.Tensor(data_set_size)

   -- now we traverse the trial file, expected to be in format:
   --
   -- query_object gold_index || cand_img_1 ... cand_img_L || seq_obj1
   -- ... seq_objM
   -- where L is candidate_cardinality and M is input_sequence_cardinality
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
	 -- query will contain object lets retrieve the
	 -- word embedding for it
	 query_object_list[i]=word_embeddings[current_data[1]]

	 -- append gold index to corresponding list
	 gold_index_list[i]=current_data[2]

	 -- we ignore current_data[3] as it's a pingie-pingie and move on to the following
	 -- sequence of candidate images
	 local tensor_counter = 0
	 for j=4,candidate_cardinality+3 do
	    local candidate_image=current_data[j]
	    tensor_counter = tensor_counter + 1
	    output_sequence_list[tensor_counter][i]=image_embeddings[candidate_image]
	 end


	 -- skipping another pingie pingie in position candidate_cardinality+4
	 
	 -- finally, we update each tensor in input_sequence_list with
	 -- alternating word att (all 0s) and image object embeddings
	 tensor_counter = 0
	 local start_at = candidate_cardinality+5
	 local end_at = start_at+input_sequence_cardinality-1
	 for j=start_at,end_at do
	    -- current_data will contain objects
	    tensor_counter = tensor_counter + 2
	    input_sequence_list[tensor_counter][i]=image_embeddings[current_data[j]]
	 end
	 i=i+1
      end
   end
   f.close()

   table.insert(output_table,query_att1_list)
   table.insert(output_table,query_att2_list)
   table.insert(output_table,query_object_list)
   for j=1,(input_sequence_cardinality*2) do
      table.insert(output_table,input_sequence_list[j])
   end
   for j=1,candidate_cardinality do
      table.insert(output_table,output_sequence_list[j])
   end
   return output_table, gold_index_list
end



