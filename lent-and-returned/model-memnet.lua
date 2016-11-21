-- our reimplementation of memory networks, with variants

function soft_retrieval_memn(query,entity_matrix,output_entity_matrix,temperature)
   -- we take the dot product of each row vector in the
   -- memory with the linguistic query vector, to obtain a
   -- memory-to-query similarity profile, that we softmax normalize
   -- (note Views needed to get right shapes, and rescaling by
   -- temperature)
   -- query needs to be a column vector
   local query_as_column = nn.View(-1,1):setNumInputDims(1)(query)
   local raw_query_entity_similarity_profile = nn.View(-1):setNumInputDims(2)(nn.MM(false,false)({entity_matrix,query_as_column}))
   local rescaled_query_entity_similarity_profile = nn.MulConstant(temperature)(raw_query_entity_similarity_profile)
   local query_entity_similarity_profile = nn.View(1,-1):setNumInputDims(1)(nn.SoftMax()(rescaled_query_entity_similarity_profile)):annotate{name='query_entity_similarity_profile'}
      
   -- we now do "soft retrieval" of the memory vector(s) that match
   -- the query: we obtain a vector that is a weighted sum of all the
   -- memory vectors in the memory (weights = similarity profile, such
   -- that we will return the most similar vector(s) to the query) (we
   -- get a matrix of such vectors because of mini-batches) (the output
   -- is a minibatch x 1 x multi_modal_size matrix, to match with the
   -- rest of the code)
   local retrieved_entity_vector = nil
   if output_entity_matrix == nil then -- in this case it's the one matrix version, or our model
      retrieved_entity_vector = nn.View(-1):setNumInputDims(1)(nn.MM(false,false)({query_entity_similarity_profile,entity_matrix}))
   else -- this is the standard MemNN; the soft retrieval is on the *output* memory matrix
      retrieved_entity_vector = nn.View(-1):setNumInputDims(1)(nn.MM(false,false)({query_entity_similarity_profile,output_entity_matrix}))
   end
   return retrieved_entity_vector
end

-- our reimplementation of memory networks
function mm_standard(t_inp_size,v_inp_size,mm_size,inp_seq_cardinality,candidate_cardinality,hops_count,temperature,dropout_p,use_cuda)

   local inputs = {}

   -- a table to store tables of connections that must share parameters
   local shareList = {}

   -- table to collect query attribute mappings, to be shared
   local query_attribute_mappings = {}

   -- first, we process the query, mapping it onto multimodal space

   -- the first attribute in the query
   local curr_input = nn.Identity()()
   table.insert(inputs,curr_input)
   local query_attribute_1_do = nn.Dropout(dropout_p)(curr_input)
   local query_attribute_1 = nn.LinearNB(t_inp_size, mm_size)(query_attribute_1_do):annotate{name='query_att1'}
   table.insert(query_attribute_mappings,query_attribute_1)

   -- the second attribute in the query
   local curr_input = nn.Identity()()
   table.insert(inputs,curr_input)
   local query_attribute_2_do = nn.Dropout(dropout_p)(curr_input)
   local query_attribute_2 = nn.LinearNB(t_inp_size, mm_size)(query_attribute_2_do):annotate{name='query_att2'}
   table.insert(query_attribute_mappings,query_attribute_2)
   
   -- the object name in the query
   local curr_input = nn.Identity()()
   table.insert(inputs,curr_input)
   local query_object_do = nn.Dropout(dropout_p)(curr_input)
   local query_object = nn.LinearNB(t_inp_size, mm_size)(query_object_do):annotate{name='query_object'}

   -- putting together the multimodal query vector by summing the
   -- output of the previous linear transformations
   local query = nn.CAddTable()({query_attribute_1,query_attribute_2,query_object}):annotate{name='query'}

   -- adding the query attribute mappings to the table of sets sharing weights
   table.insert(shareList,query_attribute_mappings)

   -- now we process the object tokens

   -- two tables to store the entity matrix as it evolves through
   -- time, one storing output representations of the entities
   local entity_matrix_table = {}
   local output_entity_matrix_table = {}

   -- tables where to store the connections that must share weights:
   ---- token attribute mappings
   local token_attribute_mappings = {}
   local output_token_attribute_mappings = {}
   ---- token object mappings
   local token_object_mappings = {}
   local output_token_object_mappings = {}

   -- now we process all the object tokens in a loop
   for i=1,inp_seq_cardinality do
      -- processing the attribute
      local curr_input = nn.Identity()()
      table.insert(inputs,curr_input)
      local token_attribute_do = nn.Dropout(dropout_p)(curr_input)
      local token_attribute = nn.LinearNB(t_inp_size,mm_size)(token_attribute_do)
      table.insert(token_attribute_mappings,token_attribute)
      local output_token_attribute_do = nn.Dropout(dropout_p)(curr_input)
      local output_token_attribute = nn.LinearNB(t_inp_size,mm_size)(output_token_attribute_do)
      table.insert(output_token_attribute_mappings,output_token_attribute)
      -- processing the object image
      local curr_input = nn.Identity()()
      table.insert(inputs,curr_input)
      local token_object_do = nn.Dropout(dropout_p)(curr_input)
      local token_object = nn.LinearNB(v_inp_size,mm_size)(token_object_do)
      table.insert(token_object_mappings,token_object)
      local output_token_object_do = nn.Dropout(dropout_p)(curr_input)
      local output_token_object = nn.LinearNB(v_inp_size,mm_size)(output_token_object_do)
      table.insert(output_token_object_mappings,output_token_object)
      -- putting together attribute and object 
      local object_token_vector_flat = nn.CAddTable()({token_attribute,token_object}):annotate{name='object_token_' .. i}
      local output_object_token_vector_flat = nn.CAddTable()({output_token_attribute,output_token_object}):annotate{name='output_object_token_' .. i}

      -- turning the vector into a 1xmm_size matrix, adding the latter
      -- to the memory tables
      table.insert(entity_matrix_table,nn.View(1,-1):setNumInputDims(1)(object_token_vector_flat))
      table.insert(output_entity_matrix_table,nn.View(1,-1):setNumInputDims(1)(output_object_token_vector_flat))
   end
   -- end of processing input objects

   -- at this point we create the memory by concatenating
   -- all vectors row-wise
   local entity_matrix = nn.JoinTable(1,2)(entity_matrix_table)
   -- and the output memory matrix, also by concatenating row-wise
   local output_entity_matrix = nn.JoinTable(1,2)(output_entity_matrix_table)

   -- putting all sets of mappings that must share weights in the shareList
   table.insert(shareList,token_attribute_mappings)
   table.insert(shareList,token_object_mappings)
   table.insert(shareList,output_token_attribute_mappings)
   table.insert(shareList,output_token_object_mappings)

   -- now we perform retrieval multiple times, updating the query
   -- representation with the retrieved vector at each hop

   -- this is a 1-dimensional vector, so it can be added to the query
   local retrieved_entity_vector = soft_retrieval_memn(query,entity_matrix,output_entity_matrix,temperature)
   if hops_count == 1 then
      query = nn.CAddTable()({query,retrieved_entity_vector})
   else
      local query_to_query_mappings = {} -- to share parameters of linear
      -- transformation of query from
      -- one hop to the other
      for current_hop_stage=1,hops_count do
	 -- update query by summing current retrieved vector to it, after a linear mapping of the current query
	 local mapped_query = nn.LinearNB(mm_size, mm_size)(query)
	 table.insert(query_to_query_mappings,mapped_query)       -- parameters of this mapping are shared across hops
	 query = nn.CAddTable()({mapped_query,retrieved_entity_vector})
      end -- end of hopping
      -- adding query to query mapping parameters to table of parameter sets sharing weights
      table.insert(shareList,query_to_query_mappings)
   end

   -- to generate the final prediction vector, we pass the last
   -- version of the query through a linear mapping
   local final_prediction = nn.LinearNB(mm_size,mm_size)(query)
   -- reshaping final_prediction so it is of the right form to match the function return_entity_image
   final_prediction_column = nn.View(-1,1,mm_size):setNumInputDims(2)(final_prediction)

   -- now we call the return_entity_image function to obtain a softmax
   -- over candidate images
   local output_distribution=return_entity_image(v_inp_size,mm_size,candidate_cardinality,dropout_p,inputs,shareList,final_prediction_column)
   
   -- wrapping up the model
   local model = nn.gModule(inputs,{output_distribution})
   
   -- following code is adapted from MeMNN 
   if (use_cuda ~= 0) then
      model:cuda()
   end
   -- IMPORTANT! do weight sharing after model is in cuda
   for i = 1,#shareList do
      local m1 = shareList[i][1].data.module
      for j = 2,#shareList[i] do
	 local m2 = shareList[i][j].data.module
	 m2:share(m1,'weight','bias','gradWeight','gradBias')
      end
   end
   return model

end

-- our reimplementation of memory networks, but with just one matrix
function mm_one_matrix(t_inp_size,v_inp_size,mm_size,inp_seq_cardinality,candidate_cardinality,hops_count,temperature,dropout_p,use_cuda)

   local inputs = {}

   -- a table to store tables of connections that must share parameters
   local shareList = {}

   -- table to collect query attribute mappings, to be shared
   local query_attribute_mappings = {}

   -- first, we process the query, mapping it onto multimodal space

   -- the first attribute in the query
   local curr_input = nn.Identity()()
   table.insert(inputs,curr_input)
   local query_attribute_1_do = nn.Dropout(dropout_p)(curr_input)
   local query_attribute_1 = nn.LinearNB(t_inp_size, mm_size)(query_attribute_1_do):annotate{name='query_att1'}
   table.insert(query_attribute_mappings,query_attribute_1)

   -- the second attribute in the query
   local curr_input = nn.Identity()()
   table.insert(inputs,curr_input)
   local query_attribute_2_do = nn.Dropout(dropout_p)(curr_input)
   local query_attribute_2 = nn.LinearNB(t_inp_size, mm_size)(query_attribute_2_do):annotate{name='query_att2'}
   table.insert(query_attribute_mappings,query_attribute_2)
   
   -- the object name in the query
   local curr_input = nn.Identity()()
   table.insert(inputs,curr_input)
   local query_object_do = nn.Dropout(dropout_p)(curr_input)
   local query_object = nn.LinearNB(t_inp_size, mm_size)(query_object_do):annotate{name='query_object'}

   -- putting together the multimodal query vector by summing the
   -- output of the previous linear transformations
   local query = nn.CAddTable()({query_attribute_1,query_attribute_2,query_object}):annotate{name='query'}

   -- adding the query attribute mappings to the table of sets sharing weights
   table.insert(shareList,query_attribute_mappings)

   -- now we process the object tokens

   -- a table to store the entity matrix
   local entity_matrix_table = {}

   -- tables where to store the connections that must share weights:
   ---- token attribute mappings
   local token_attribute_mappings = {}
   ---- token object mappings
   local token_object_mappings = {}

   -- now we process all the object tokens in a loop
   for i=1,inp_seq_cardinality do
      -- processing the attribute
      local curr_input = nn.Identity()()
      table.insert(inputs,curr_input)
      local token_attribute_do = nn.Dropout(dropout_p)(curr_input)
      local token_attribute = nn.LinearNB(t_inp_size,mm_size)(token_attribute_do)
      table.insert(token_attribute_mappings,token_attribute)
      -- processing the object image
      local curr_input = nn.Identity()()
      table.insert(inputs,curr_input)
      local token_object_do = nn.Dropout(dropout_p)(curr_input)
      local token_object = nn.LinearNB(v_inp_size,mm_size)(token_object_do)
      table.insert(token_object_mappings,token_object)
      -- putting together attribute and object 
      local object_token_vector_flat = nn.CAddTable()({token_attribute,token_object}):annotate{name='object_token_' .. i}
      -- reshaping
      local object_token_vector = nn.View(1,-1):setNumInputDims(1)(object_token_vector_flat)
      table.insert(entity_matrix_table,object_token_vector)
   end
   -- end of processing input objects

   -- at this point we create the entity matrix by concatenating
   -- all vectors row-wise
   local entity_matrix = nn.JoinTable(1,2)(entity_matrix_table)

   -- putting all sets of mappings that must share weights in the shareList
   table.insert(shareList,token_attribute_mappings)
   table.insert(shareList,token_object_mappings)

   -- now we perform retrieval multiple times, updating the query
   -- representation with the retrieved vector at each hop

   -- this is a 1-dimensional vector, so it can be added to the query
   local retrieved_entity_vector = soft_retrieval_memn(query,entity_matrix,nil,temperature)
   if hops_count == 1 then
      query = nn.CAddTable()({query,retrieved_entity_vector})
   else
      local query_to_query_mappings = {} -- to share parameters of linear
      -- transformation of query from
      -- one hop to the other
      for current_hop_stage=1,hops_count do
	 -- update query by summing current retrieved vector to it, after a linear mapping of the current query
	 local mapped_query = nn.LinearNB(mm_size, mm_size)(query)
	 table.insert(query_to_query_mappings,mapped_query)       -- parameters of this mapping are shared across hops
	 query = nn.CAddTable()({mapped_query,retrieved_entity_vector})
      end -- end of hopping
      -- adding query to query mapping parameters to table of parameter sets sharing weights
      table.insert(shareList,query_to_query_mappings)
   end

   -- to generate the final prediction vector, we pass the last
   -- version of the query through a linear mapping
   local final_prediction = nn.LinearNB(mm_size,mm_size)(query)
   -- reshaping final_prediction so it is of the right form to match the function return_entity_image
   final_prediction_column = nn.View(-1,1,mm_size):setNumInputDims(2)(final_prediction)

   -- now we call the return_entity_image function to obtain a softmax
   -- over candidate images
   local output_distribution=return_entity_image(v_inp_size,mm_size,candidate_cardinality,dropout_p,inputs,shareList,final_prediction_column)
   
   -- wrapping up the model
   local model = nn.gModule(inputs,{output_distribution})
   
   -- following code is adapted from MeMNN 
   if (use_cuda ~= 0) then
      model:cuda()
   end
   -- IMPORTANT! do weight sharing after model is in cuda
   for i = 1,#shareList do
      local m1 = shareList[i][1].data.module
      for j = 2,#shareList[i] do
	 local m2 = shareList[i][j].data.module
	 m2:share(m1,'weight','bias','gradWeight','gradBias')
      end
   end
   return model

end

