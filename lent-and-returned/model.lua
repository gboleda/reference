-- to pad with zeroes (y would be an extra row with 0s)
-- y = torch.Tensor(1,2):zero()
-- nn.JoinTable(1,2):forward({x,y})

-- summing tensors
-- nn.CAddTable():forward({nah,bah})

function entity_prediction(t_inp_size,o_inp_size,mm_size,inp_seq_cardinality)

   local inputs = {}

   -- first, we process the query, mapping it onto multimodal space

   -- the first attribute in the query
   local curr_input = nn.Identity()()
   table.insert(inputs,curr_input)
   local query_attribute_1 = nn.LinearNB(t_inp_size, mm_size)(curr_input):annotate{name='query_att1'}

   -- the second attribute in the query
   local curr_input = nn.Identity()()
   table.insert(inputs,curr_input)
   local query_attribute_2 = nn.LinearNB(t_inp_size, mm_size)(curr_input):annotate{name='query_att2'}
   -- sharing matrix with first attribute (no bias/gradBias sharing
   -- since we're not using the bias term)
   query_attribute_2.data.module:share(query_attribute_1.data.module,'weight','gradWeight')
   
   -- the object name in the query
   local curr_input = nn.Identity()()
   table.insert(inputs,curr_input)
   local query_object = nn.LinearNB(t_inp_size, mm_size)(curr_input):annotate{name='query_object'}

   -- putting together the multimodal query vector by summing the output of the previous
   -- linear transformations
   local query = nn.CAddTable()({query_attribute_1,query_attribute_2,query_object}):annotate{name='query'}

   -- now we process the object tokens

   -- a table to store the entity matrix as it evolves through time
   local entity_matrix_table = {}

   -- a table to store the first cell recording the "new entity" mass
   -- which will serve as the template for weight sharing of the other
   -- cells (this seems like the most elegant way, in terms of
   -- minimizing repeated call)
   local raw_new_entity_mass_template_table = {}

   -- the first object token is a special case, as it will always be
   -- directly mapped to the first row of the entity matrix
   local curr_input = nn.Identity()()
   table.insert(inputs,curr_input)
   local first_object_token_vector = nn.LinearNB(o_inp_size,mm_size)(curr_input):annotate{name='object_token_1'}
   -- turning the vector into a 1xmm_size matrix, adding the latter as
   -- first state of the entity matrix table
   table.insert(entity_matrix_table,nn.View(1,-1):setNumInputDims(1)(first_object_token_vector))


   -- debug
   local raw_new_entity_mass = nil
   -- now we process all the other object tokens in a loop
   for i=2,inp_seq_cardinality do
      local curr_input = nn.Identity()()
      table.insert(inputs,curr_input)
      local object_token_vector_name='object_token_' .. i
      local object_token_vector = nn.LinearNB(o_inp_size,mm_size)(curr_input):annotate{name=object_token_vector_name}  
      -- parameters to be shared with first mapped vector
      object_token_vector.data.module:share(first_object_token_vector.data.module,'weight','gradWeight')
      -- measuring the similarity of the current vector to the ones in
      -- the previous state of the entity matrix
      local raw_similarity_profile_to_entity_matrix = nn.MM(false,false)
      ({entity_matrix_table[i-1],nn.View(1,-1):setNumInputDims(1)(object_token_vector)})
      
      -- computing the new-entity cell value
      -- Mean or Sum??? this could be a parameter...
      -- debug
      raw_new_entity_mass = nn.Linear(1,1)(nn.Mean(1,-1)(raw_similarity_profile_to_entity_matrix))
      --local raw_new_entity_mass = nn.Linear(1,1)(nn.Mean(1,-1)(raw_similarity_profile_to_entity_matrix))
      if i==2 then -- this is the first cell, let's store it as a template
	 table.insert(raw_new_entity_mass_template_table,raw_new_entity_mass)
	 else -- share parameters
	    raw_new_entity_mass.data.module:share(raw_new_entity_mass_template_table[1].data.module,'weight','bias','gradWeight','gradBias')
      end
   end

   -- wrapping up the model
   return nn.gModule(inputs,{query,raw_similarity_profile_to_entity_matrix,raw_new_entity_mass})

end
