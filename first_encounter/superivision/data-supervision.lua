
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

function create_onehots(size, prefix)
  local identity = torch.eye(size)
  local embeddings = {}
  for i=1,size do
    embeddings[prefix..i] = identity[i]
  end
  return embeddings
end


-- returns output_table, containing a set of n x embeddings_dim tensors
-- each of which has the data for one input trial per row (n is the
-- number of trials, embeddings_dim changes depending on the nature of
-- the corresponding input data: words vs images), as well as a nx1
-- tensor with the gold indices (gold_index_list)

function create_data_tables_from_file(i_file,data_set_size,input_sequence_cardinality)
   print('reading protocol file ' .. i_file)

   -- initializing the data structures to hold the data
   local all_input_table = {} -- to put data information in (will be
         -- transformed to model input when we
         -- create the minibatches in the training
         -- file)
         
   local all_output_table = {}

   -- for the query, we need 2 tables the two attributes
   local query_att1_list = {}
   local query_att2_list = {}
   
   local entity_creating_list = {}

   -- input_sequence_list is an 2*input_sequence_cardinality table of
   -- alternating tables: the odd table will contain attributes, the
   -- even slots will contain image names, with each pair of
   -- consecutive tables standing for an object token (pairs ordered
   -- as the object tokens in the corresponding stimulus); the entries
   -- in the input_sequence_list table are aligned with the order of
   -- the queries in the query lists
   local input_sequence_list={}
   for i=1,input_sequence_cardinality do
      table.insert(input_sequence_list,{})
      table.insert(input_sequence_list,{})
      if i < input_sequence_cardinality then 
         table.insert(entity_creating_list,{})
      end
   end

   -- gold_index_list contains, for each sample, the index of correct
   -- candidate image (the index of the image in the output set
   -- corresponding to the composite meaning denoted by the linguistic
   -- query) in the corresponding sequence of tensors in
   -- output_sequence_list
   local gold_index_list = torch.Tensor(data_set_size)

   -- now we traverse the trial file, expected to be in format:
   -- query_att1:query_att2 gold_index seq_att1:seq_obj1
   -- ... seq_attM:seq:objM
   
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
         -- query will contain object, att1, att2, let's store them
         local query=current_data[1]:split(":")
         table.insert(query_att1_list,query[1])
         table.insert(query_att2_list,query[2])
         -- append gold index to corresponding list
         gold_index_list[i]=current_data[2]
      
         local table_counter = 0
         local start_at = 3
         local end_at = start_at+input_sequence_cardinality-1
         local entity_index = 1 -- for computation convenience, entity_index here is the exposure index - 1
         for j=start_at,end_at do
            -- object token contains attr, object
            local object_token=current_data[j]:split(":")
            table_counter = table_counter + 1
            input_sequence_list[table_counter][i]=object_token[1]
            table_counter = table_counter + 1
            input_sequence_list[table_counter][i]=object_token[2]
            if (j > start_at) then
               local repetition_index = 0
               for prob_index=1,entity_index do
                  if input_sequence_list[prob_index * 2][i] == input_sequence_list[table_counter][i] then
                     repetition_index = prob_index
                  end
               end
               if repetition_index > 0 then
                  entity_creating_list[entity_index][i] = 0
               else
                  entity_creating_list[entity_index][i] = 1
               end
               entity_index = entity_index + 1
            end
         end
         i=i+1
      end
   end
   f.close()

   table.insert(all_input_table,query_att1_list)
   table.insert(all_input_table,query_att2_list)
   for j=1,(input_sequence_cardinality*2) do
      table.insert(all_input_table,input_sequence_list[j])
   end
   for j=1,(input_sequence_cardinality - 1) do
      table.insert(all_output_table,entity_creating_list[j])
   end
   table.insert(all_output_table, gold_index_list)
   return all_input_table, all_output_table
end


-- returns output_table, containing a table of n x embeddings_dim
-- tensors each of which has the data for one input trial per row (n
-- is the number of trials, embeddings_dim changes depending on the
-- nature of the corresponding input data: words vs images), as well
-- as a nx1 tensor with the gold indices
function create_input_structures_from_table(data_tables,output_table,target_indices,data_set_size,t_in_size,v_in_size,input_sequence_cardinality,use_cuda)
   -- initializing the data structures to hold the data
   local output_tensor_table = {} -- to put data tensors in (will be model input)

   -- for the query, we need 2 tensors, two for the attributes 
   -- each of them is data_set_size x t_in_size
   -- here and below, we initialize the tensors to 0, so if an item is
   -- missing, we will simply use the 0 vector as its representation
   local query_att1_list = torch.Tensor(data_set_size,t_in_size):fill(0)
   local query_att2_list = torch.Tensor(data_set_size,t_in_size):fill(0)
   local full_gold_index_tensor = output_table[#output_table]

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

   -- gold_index_list contains, for each sample, the index of correct
   -- candidate image (the index of the image in the output set
   -- corresponding to the composite meaning denoted by the linguistic
   -- query) in the corresponding sequence of tensors in
   -- output_sequence_list
   local gold_index_list = torch.Tensor(data_set_size)
   
   -- now we traverse the indices populating the various tables with
   -- the corresponding contents
   
   

   for i=1,target_indices:size()[1] do
      local current_index=target_indices[i]
      -- first we get the query att 1/2 and object tables
      if (word_embeddings[data_tables[1][current_index]]~=nil) then
         query_att1_list[i]=word_embeddings[data_tables[1][current_index]]
      end
      if (word_embeddings[data_tables[2][current_index]]~=nil) then
         query_att2_list[i]=word_embeddings[data_tables[2][current_index]]
      end
      -- now we read the input tokens, that will be in and alternating word attribute
      -- and image object sequence
      local j=3 -- counter to go through the input table of tables
      while (j<=(input_sequence_cardinality*2)+2) do
         if (word_embeddings[data_tables[j][current_index]]~=nil) then
            input_sequence_list[j-2][i]=
               word_embeddings[data_tables[j][current_index]]
         end
         j=j+1
         if (image_embeddings[data_tables[j][current_index]]~=nil) then
            input_sequence_list[j-2][i]=
               image_embeddings[data_tables[j][current_index]]
         end
         j=j+1
      end
      gold_index_list[i]=full_gold_index_tensor[current_index]
   end

   local entity_weight_list = {}
   for i = 1,input_sequence_cardinality - 1 do
      local weight_i = {}
      local expo_index = i + 1
      for j=1,target_indices:size()[1] do
         local current_index=target_indices[j]
         table.insert(weight_i, output_table[i][current_index])
      end
      table.insert(entity_weight_list, torch.Tensor(weight_i))
   end
   
   if (use_cuda ~=0) then
      table.insert(output_tensor_table,query_att1_list:cuda())
      table.insert(output_tensor_table,query_att2_list:cuda())
      for j=1,(input_sequence_cardinality*2) do
         table.insert(output_tensor_table,input_sequence_list[j]:cuda())
      end
      for j=1,(input_sequence_cardinality - 1) do
         entity_weight_list[j] = entity_weight_list[j]:cuda()
      end
      gold_index_list=gold_index_list:cuda()
   else
      table.insert(output_tensor_table,query_att1_list)
      table.insert(output_tensor_table,query_att2_list)
      for j=1,(input_sequence_cardinality*2) do
         table.insert(output_tensor_table,input_sequence_list[j])
      end
   end
   
   return output_tensor_table, entity_weight_list, gold_index_list
end
