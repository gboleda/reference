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
function create_input_structures_from_file(i_file,data_set_size,t_in_size,v_in_size,input_sequence_cardinality)
   print('reading protocol file ' .. i_file)

   -- initializing the data structures to hold the data
   local output_table = {} -- to put data tensors in (will be model input)

   -- for the query, we need 3 tensors, two for the attributes, one
   -- for the object name: each of them is data_set_size x t_in_size
   local query_att1_list = torch.Tensor(data_set_size,t_in_size)
   local query_att2_list = torch.Tensor(data_set_size,t_in_size)
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

   -- gold_index_list contains, for each sample, the index of correct
   -- object token (the index of the first object token corresponding
   -- to the composite meaning denoted by the word query) in the
   -- corresponding sequence of tensors in input_sequence_list
   local gold_index_list = torch.Tensor(data_set_size)

   -- now we traverse the trial file, expected to be in format:
   --
   -- query_object:query_att1:query_att2 gold_index seq_att1:seq_obj1
   -- ... seq_attM:seq:objM
   -- where M is input_sequence_cardinality
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
	 -- query will contain att1, att2, object, lets retrieve their
	 -- word embeddings
	 local query=current_data[1]:split(":")
	 query_object_list[i]=word_embeddings[query[1]]
	 query_att1_list[i]=word_embeddings[query[2]]
	 query_att2_list[i]=word_embeddings[query[3]]

	 -- append gold index to corresponding list
	 gold_index_list[i]=current_data[2]

	 -- finally, we update each tensor in input_sequence_list with
	 -- alternating word att and image object embeddings
	 local tensor_counter = 0
	 for j=1,input_sequence_cardinality do
	    local id_position=j+2
	    -- object token contains attr, object
	    local object_token=current_data[id_position]:split(":")
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
   return output_table, gold_index_list
end


---- DEPRECATED FROM HERE
-- returns output_table, containing a set of n x embeddings_dim tensors
-- each of which has the data for one input trial per row (n is the
-- number of trials, embeddings_dim changes depending on the nature of
-- the corresponding input data: words vs images), as well as a nx1
-- tensor with the gold indices
function create_input_structures_from_file_DEPRECATED(i_file,data_set_size,t_in_size,v_in_size,input_sequence_cardinality)
   print('reading protocol file ' .. i_file)

   -- initializing the data structures to hold the data
   local output_table = {} -- to put data tensors in (will be model input)

   -- for the query, we need 3 tensors, two for the attributes, one
   -- for the object name: each of them is data_set_size x t_in_size
   local query_att1_list = torch.Tensor(data_set_size,t_in_size)
   local query_att2_list = torch.Tensor(data_set_size,t_in_size)
   local query_object_list = torch.Tensor(data_set_size,t_in_size)

   -- input_sequence_list is an input_sequence_cardinality table of
   -- data_set_size x (t_in_size+v_in_size) tensors: the ith position
   -- of each of these tensors will contain and "object token", given
   -- by the concatenation of an attribute and an image
   -- representations, supposed to be in the the ith sequence
   -- (sequences ordered as the corresponding queries in the same
   -- positions of the query lists); the tensors in the
   -- input_sequence_list table are ordered to reflect the order of
   -- the object tokens in the sequence
   local input_sequence_list={}
   for i=1,input_sequence_cardinality do
      table.insert(input_sequence_list,torch.Tensor(data_set_size,t_in_size+v_in_size))
   end

   -- gold_index_list contains, for each sample, the index of correct
   -- object token (the index of the first object token corresponding
   -- to the composite meaning denoted by the word query) in the
   -- corresponding sequence of tensors in input_sequence_list
   local gold_index_list = torch.Tensor(data_set_size)

   -- now we traverse the trial file, expected to be in format:
   --
   -- query_object:query_att1:query_att2 gold_index seq_att1:seq_obj1
   -- ... seq_attM:seq:objM
   -- where M is input_sequence_cardinality
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
	 -- query will contain att1, att2, object, lets retrieve their
	 -- word embeddings
	 local query=current_data[1]:split(":")
	 query_object_list[i]=word_embeddings[query[1]]
	 query_att1_list[i]=word_embeddings[query[2]]
	 query_att2_list[i]=word_embeddings[query[3]]

	 -- append gold index to corresponding list
	 gold_index_list[i]=current_data[2]

	 -- finally, we update each tensor in input_sequence_list with
	 -- concatenated word att and image object embeddings
	 for j=1,input_sequence_cardinality do
	    local id_position=j+2
	    -- object token will contain attr, object
	    local object_token=current_data[id_position]:split(":")
	    input_sequence_list[j][i]=torch.cat(word_embeddings[object_token[1]],image_embeddings[object_token[2]],1)
	 end
	 i=i+1
      end
   end
   f.close()

   table.insert(output_table,query_att1_list)
   table.insert(output_table,query_att2_list)
   table.insert(output_table,query_object_list)
   for j=1,input_sequence_cardinality do
      table.insert(output_table,input_sequence_list[j])
   end
   return output_table, gold_index_list
end
