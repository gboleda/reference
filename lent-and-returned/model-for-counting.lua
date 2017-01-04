function build_entity_matrix(t_inp_size, v_inp_size, mm_size, inp_seq_cardinality, dropout_p, in_table, obj_mappings_table, att_mappings_table, share_table)
   -- this function builds an entity matrix with as many rows as input
   -- tokens, although some row might be 0 vectors because the
   -- corresponding input information is mapped to existing entity
   -- vectors

   -- note that we assume that the object- and attribute-parameter
   -- tables (obj_mappings_table, att_mappings_table) have already
   -- been created, because we want to share their parameters with
   -- other components of the model: if you don't want to share their
   -- parameters with other components, you should make sure you're
   -- passing special tables
   
   -- a table to store the entity matrix as it evolves through time
   local entity_matrix_table = {}

   -- table to store the connections that must share weights:
   --- mappings to raw new entity mass
   local raw_new_entity_mass_mappings = {}

   -- the first object token is a special case, as it will always be
   -- directly mapped to the first row of the entity matrix
   -- first processing the attribute
   local curr_input = nn.Identity()()
   table.insert(in_table,curr_input)
   local first_token_attribute_do = nn.Dropout(dropout_p)(curr_input)
   local first_token_attribute = nn.LinearNB(t_inp_size,mm_size)(first_token_attribute_do)
   table.insert(att_mappings_table,first_token_attribute)
   -- then processing the object image
   local curr_input = nn.Identity()()
   table.insert(in_table,curr_input)
   local first_token_object_do = nn.Dropout(dropout_p)(curr_input)
   local first_token_object = nn.LinearNB(v_inp_size,mm_size)(first_token_object_do)
   table.insert(obj_mappings_table,first_token_object)
   -- putting together attribute and object 
   local first_object_token_vector = nn.CAddTable()({first_token_attribute,first_token_object})
   -- turning the vector into a 1xmm_size matrix, adding the latter as
   -- first state of the entity matrix table
   table.insert(entity_matrix_table,nn.View(1,-1):setNumInputDims(1)(first_object_token_vector))

   -- now we process all the other object tokens in a loop
   for i=2,inp_seq_cardinality do
      -- processing the attribute
      local curr_input = nn.Identity()()
      table.insert(in_table,curr_input)
      local token_attribute_do = nn.Dropout(dropout_p)(curr_input)
      local token_attribute = nn.LinearNB(t_inp_size,mm_size)(token_attribute_do)
      table.insert(att_mappings_table,token_attribute)
      -- processing the object image
      local curr_input = nn.Identity()()
      table.insert(in_table,curr_input)
      local token_object_do = nn.Dropout(dropout_p)(curr_input)
      local token_object = nn.LinearNB(v_inp_size,mm_size)(token_object_do)
      table.insert(obj_mappings_table,token_object)
      -- putting together attribute and object 
      local object_token_vector_flat = nn.CAddTable()({token_attribute,token_object}):annotate{name='object_token_' .. i}
      -- reshaping
      local object_token_vector = nn.View(-1,1):setNumInputDims(1)(object_token_vector_flat)

      -- measuring the similarity of the current vector to the ones in
      -- the previous state of the entity matrix
      local raw_similarity_profile_to_entity_matrix = nn.MM(false,false)
      ({entity_matrix_table[i-1],object_token_vector})

      -- computing the new-entity cell value
      -- average or max or sum by default of input vector cells...
      local raw_cumulative_similarity=nil
      if (opt.new_mass_aggregation_method=='mean') then
        raw_cumulative_similarity=nn.Mean(1,2)(raw_similarity_profile_to_entity_matrix)
      elseif (opt.new_mass_aggregation_method=='max') then
         raw_cumulative_similarity=nn.Max(1,2)(raw_similarity_profile_to_entity_matrix)
      else -- sum by default
         raw_cumulative_similarity = nn.Sum(1,2)(raw_similarity_profile_to_entity_matrix)
      end
      raw_cumulative_similarity:annotate{name='raw_cumulative_similarity_' .. i}
      local raw_new_entity_mass = nn.Linear(1,1)(raw_cumulative_similarity):annotate{name='raw_new_entity_mass_' .. i}
      table.insert(raw_new_entity_mass_mappings,raw_new_entity_mass)

      -- passing through nonlinearity if requested
      local transformed_new_entity_mass=nil
      if (nonlinearity=='none') then
          transformed_new_entity_mass=raw_new_entity_mass
      else
          --local nonlinear_hidden_layer = nil
          if (nonlinearity == 'relu') then
             transformed_new_entity_mass = nn.ReLU()(raw_new_entity_mass)
          elseif (nonlinearity == 'tanh') then
             transformed_new_entity_mass = nn.Tanh()(raw_new_entity_mass)
          else -- sigmoid is leftover option: if (nonlinearity == 'sigmoid') then
             print("Really using sigmoid")
             transformed_new_entity_mass = nn.Sigmoid()(raw_new_entity_mass)
          end
      end

      -- now, we concatenate the similarity profile with this new
      -- cell, and normalize
      -- NB: the output of the following very messy line of code is a
      -- matrix with the profile of each item in a minibatch as
      -- a ROW vector
      --transformed_new_entity_mass = nn.Peek("create new", true)(transformed_new_entity_mass)
      local minus_transform_new_entity_mass = nn.AddConstant(1,false)(nn.MulConstant(-1,false)(transformed_new_entity_mass))
      local normalized_similarity_profile = nn.SoftMax()(nn.View(-1):setNumInputDims(2)(raw_similarity_profile_to_entity_matrix)):annotate{name='normalized_similarity_profile_' .. i}
      --normalized_similarity_profile = nn.Peek("softmax", true)(normalized_similarity_profile)
      normalized_similarity_profile = nn.MM(false, false){nn.View(-1,i - 1, 1)(normalized_similarity_profile),nn.View(-1,1, 1)(minus_transform_new_entity_mass)}
      normalized_similarity_profile = (nn.JoinTable(2,2)({nn.View(-1,i - 1)(normalized_similarity_profile),transformed_new_entity_mass}))
      --normalized_similarity_profile = nn.Peek("final weight", true)(normalized_similarity_profile)
      -- we now create a matrix that has, on each ROW, the current
      -- token vector, multiplied by the corresponding entry on the
      -- normalized similarity profile (including, in the final row,
      -- weighting by the normalized new mass cell): 
      local weighted_object_token_vector_matrix = nn.MM(false,true){nn.View(-1,1):setNumInputDims(1)(normalized_similarity_profile),object_token_vector}

      -- at this point we update the entity matrix by adding the
      -- weighted versions of the current object token vector to each
      -- row of it (we pad the bottom of the entity matrix with a zero
      -- row, so that we can add it to the version of the current
      -- object token vector that was weighted by the new mass cell
      entity_matrix_table[i]= nn.CAddTable(){
         nn.Padding(1,1,2)(entity_matrix_table[i-1]),weighted_object_token_vector_matrix}:annotate{'entity_matrix_table' .. i}
   end
   -- end of processing input objects

   -- adding to shareList the entity mass mappings
   table.insert(share_table,raw_new_entity_mass_mappings)

   -- and finally returning the last state of the entity matrix
   return entity_matrix_table[inp_seq_cardinality]
end

-- adaptation of the models to counting
-- to do:
-- for the entity counting task: normalize every entity vector and sum. # gbt: Note that we are actually exposing the entity library in this task. # gbt: How are we going to adapt the other models?
-- but... if we do this and we have identical images for the entities... isn't the entity counting task trivial for our model?
-- for the entity-of-a-given-category counting task: normalize every entity vector, do dot product to query of every entity vector, and sum. # for the moment we do it without normalizing to see if the model learns to scale. If not, revert to normalization.
function entity_prediction_image_att_shared_neprob_counting(t_inp_size,v_inp_size,mm_size,inp_seq_cardinality,candidate_cardinality,nonlinearity,temperature,dropout_p,use_cuda)

   local inputs = {}

   -- a table to store tables of connections that must share parameters
   local shareList = {}

   -- table to collect all attribute mappings, to be shared
   local attribute_mappings = {}

   -- first, we process the query, mapping it onto multimodal space

   -- the first attribute in the query
   local curr_input = nn.Identity()()
   table.insert(inputs,curr_input)
   local query_attribute_1_do = nn.Dropout(dropout_p)(curr_input)
   local query_attribute_1 = nn.LinearNB(t_inp_size, mm_size)(query_attribute_1_do):annotate{name='query_att1'}
   table.insert(attribute_mappings,query_attribute_1)

   -- the second attribute in the query
   local curr_input = nn.Identity()()
   table.insert(inputs,curr_input)
   local query_attribute_2_do = nn.Dropout(dropout_p)(curr_input)
   local query_attribute_2 = nn.LinearNB(t_inp_size, mm_size)(query_attribute_2_do):annotate{name='query_att2'}
   table.insert(attribute_mappings,query_attribute_2)
   
   -- the object name in the query
   local curr_input = nn.Identity()()
   table.insert(inputs,curr_input)
   local query_object_do = nn.Dropout(dropout_p)(curr_input)
   local query_object = nn.LinearNB(t_inp_size, mm_size)(query_object_do):annotate{name='query_object'}

   -- putting together the multimodal query vector by summing the
   -- output of the previous linear transformations, and ensuring it
   -- will be a column vector
   query = nn.View(-1,1):setNumInputDims(1)(nn.CAddTable()({query_attribute_1,query_attribute_2,query_object}):annotate{name='query'})

   -- we initalize the table to store the object mappings here, so we can share first when constructing the entity matrix, and then
   -- later when processing the candidate images
   local token_object_mappings = {}

   -- now we call a function to process the object tokens and return an entity matrix
   local stable_entity_matrix = build_entity_matrix(t_inp_size,v_inp_size,mm_size,inp_seq_cardinality,dropout_p,inputs,token_object_mappings,attribute_mappings,shareList)

   -- we are done processing input attributes, so we add their parameter list to the list of parameters to be shared
   table.insert(shareList,attribute_mappings)
   
   -- at this point, we take the dot product of each row (entity)
   -- vector in the entity matrix with the linguistic query vector, to
   -- obtain an entity-to-query similarity profile, which we pass through a sigmoid to encourage the values to be binary
   local raw_matrix_query_entity_similarity_profile = nn.MM(false,false)({stable_entity_matrix,query})
   local matrix_query_entity_similarity_profile = nn.Sigmoid()(raw_matrix_query_entity_similarity_profile)
   -- and we sum them to obtain the predicted number of entities
   local output_number_of_entities = nn.Sum(1,2)(matrix_query_entity_similarity_profile)
   -- adding token_object_mappings to shareList only now, after we also added to it the candidate image mappings
   table.insert(shareList,token_object_mappings)

   -- wrapping up the model
   local model = nn.gModule(inputs,{output_number_of_entities})
   
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


-- BACKUP FROM HERE
-- adaptation of the models to counting
-- to do:
-- for the entity counting task: normalize every entity vector and sum. # gbt: Note that we are actually exposing the entity library in this task. # gbt: How are we going to adapt the other models?
-- but... if we do this and we have identical images for the entities... isn't the entity counting task trivial for our model?
-- for the entity-of-a-given-category counting task: normalize every entity vector, do dot product to query of every entity vector, and sum. # gbt: did I get it right?

function entity_prediction_image_att_shared_neprob_counting_backup(t_inp_size,v_inp_size,mm_size,inp_seq_cardinality,candidate_cardinality,nonlinearity,temperature,dropout_p,use_cuda)

   local inputs = {}

   -- a table to store tables of connections that must share parameters
   local shareList = {}

   -- table to collect all attribute mappings, to be shared
   local attribute_mappings = {}

   -- first, we process the query, mapping it onto multimodal space

   -- the first attribute in the query
   local curr_input = nn.Identity()()
   table.insert(inputs,curr_input)
   local query_attribute_1_do = nn.Dropout(dropout_p)(curr_input)
   local query_attribute_1 = nn.LinearNB(t_inp_size, mm_size)(query_attribute_1_do):annotate{name='query_att1'}
   table.insert(attribute_mappings,query_attribute_1)

   -- the second attribute in the query
   local curr_input = nn.Identity()()
   table.insert(inputs,curr_input)
   local query_attribute_2_do = nn.Dropout(dropout_p)(curr_input)
   local query_attribute_2 = nn.LinearNB(t_inp_size, mm_size)(query_attribute_2_do):annotate{name='query_att2'}
   table.insert(attribute_mappings,query_attribute_2)
   
   -- the object name in the query
   local curr_input = nn.Identity()()
   table.insert(inputs,curr_input)
   local query_object_do = nn.Dropout(dropout_p)(curr_input)
   local query_object = nn.LinearNB(t_inp_size, mm_size)(query_object_do):annotate{name='query_object'}

   -- putting together the multimodal query vector by summing the
   -- output of the previous linear transformations, and ensuring it
   -- will be a column vector
   local query = nn.View(-1,1):setNumInputDims(1)(nn.CAddTable()({query_attribute_1,query_attribute_2,query_object}):annotate{name='query'})

   -- now we process the object tokens

   -- a table to store the entity matrix as it evolves through time
   local entity_matrix_table = {}

   -- tables where to store the connections that must share weights:
   ---- token object mappings
   local token_object_mappings = {}
   --- mappings to raw new entity mass
   local raw_new_entity_mass_mappings = {}

   -- the first object token is a special case, as it will always be
   -- directly mapped to the first row of the entity matrix
   -- first processing the attribute
   local curr_input = nn.Identity()()
   table.insert(inputs,curr_input)
   local first_token_attribute_do = nn.Dropout(dropout_p)(curr_input)
   local first_token_attribute = nn.LinearNB(t_inp_size,mm_size)(first_token_attribute_do)
   table.insert(attribute_mappings,first_token_attribute)
   -- then processing the object image
   local curr_input = nn.Identity()()
   table.insert(inputs,curr_input)
   local first_token_object_do = nn.Dropout(dropout_p)(curr_input)
   local first_token_object = nn.LinearNB(v_inp_size,mm_size)(first_token_object_do)
   table.insert(token_object_mappings,first_token_object)
   -- putting together attribute and object 
   local first_object_token_vector = nn.CAddTable()({first_token_attribute,first_token_object})
   -- turning the vector into a 1xmm_size matrix, adding the latter as
   -- first state of the entity matrix table
   table.insert(entity_matrix_table,nn.View(1,-1):setNumInputDims(1)(first_object_token_vector))

   -- now we process all the other object tokens in a loop
   for i=2,inp_seq_cardinality do
      -- processing the attribute
      local curr_input = nn.Identity()()
      table.insert(inputs,curr_input)
      local token_attribute_do = nn.Dropout(dropout_p)(curr_input)
      local token_attribute = nn.LinearNB(t_inp_size,mm_size)(token_attribute_do)
      table.insert(attribute_mappings,token_attribute)
      -- processing the object image
      local curr_input = nn.Identity()()
      table.insert(inputs,curr_input)
      local token_object_do = nn.Dropout(dropout_p)(curr_input)
      local token_object = nn.LinearNB(v_inp_size,mm_size)(token_object_do)
      table.insert(token_object_mappings,token_object)
      -- putting together attribute and object 
      local object_token_vector_flat = nn.CAddTable()({token_attribute,token_object}):annotate{name='object_token_' .. i}
      -- reshaping
      local object_token_vector = nn.View(-1,1):setNumInputDims(1)(object_token_vector_flat)

      -- measuring the similarity of the current vector to the ones in
      -- the previous state of the entity matrix
      local raw_similarity_profile_to_entity_matrix = nn.MM(false,false)
      ({entity_matrix_table[i-1],object_token_vector})

      -- computing the new-entity cell value
      -- average or max or sum by default of input vector cells...
      local raw_cumulative_similarity=nil
      if (opt.new_mass_aggregation_method=='mean') then
        raw_cumulative_similarity=nn.Mean(1,2)(raw_similarity_profile_to_entity_matrix)
      elseif (opt.new_mass_aggregation_method=='max') then
         raw_cumulative_similarity=nn.Max(1,2)(raw_similarity_profile_to_entity_matrix)
      else -- sum by default
         raw_cumulative_similarity = nn.Sum(1,2)(raw_similarity_profile_to_entity_matrix)
      end
      raw_cumulative_similarity:annotate{name='raw_cumulative_similarity_' .. i}
      local raw_new_entity_mass = nn.Linear(1,1)(raw_cumulative_similarity):annotate{name='raw_new_entity_mass_' .. i}
      table.insert(raw_new_entity_mass_mappings,raw_new_entity_mass)

      -- passing through nonlinearity if requested
      local transformed_new_entity_mass=nil
      if (nonlinearity=='none') then
          transformed_new_entity_mass=raw_new_entity_mass
      else
          --local nonlinear_hidden_layer = nil
          if (nonlinearity == 'relu') then
             transformed_new_entity_mass = nn.ReLU()(raw_new_entity_mass)
          elseif (nonlinearity == 'tanh') then
             transformed_new_entity_mass = nn.Tanh()(raw_new_entity_mass)
          else -- sigmoid is leftover option: if (nonlinearity == 'sigmoid') then
             transformed_new_entity_mass = nn.Sigmoid()(raw_new_entity_mass)
          end
      end

      -- now, we concatenate the similarity profile with this new
      -- cell, and normalize
      -- NB: the output of the following very messy line of code is a
      -- matrix with the profile of each item in a minibatch as
      -- a ROW vector
      --transformed_new_entity_mass = nn.Peek("create new", true)(transformed_new_entity_mass)
      local minus_transform_new_entity_mass = nn.AddConstant(1,false)(nn.MulConstant(-1,false)(transformed_new_entity_mass))
      local normalized_similarity_profile = nn.SoftMax()(nn.View(-1):setNumInputDims(2)(raw_similarity_profile_to_entity_matrix)):annotate{name='normalized_similarity_profile_' .. i}
      --normalized_similarity_profile = nn.Peek("softmax", true)(normalized_similarity_profile)
      normalized_similarity_profile = nn.MM(false, false){nn.View(-1,i - 1, 1)(normalized_similarity_profile),nn.View(-1,1, 1)(minus_transform_new_entity_mass)}
      normalized_similarity_profile = (nn.JoinTable(2,2)({nn.View(-1,i - 1)(normalized_similarity_profile),transformed_new_entity_mass}))
      --normalized_similarity_profile = nn.Peek("final weight", true)(normalized_similarity_profile)
      -- we now create a matrix that has, on each ROW, the current
      -- token vector, multiplied by the corresponding entry on the
      -- normalized similarity profile (including, in the final row,
      -- weighting by the normalized new mass cell): 
      local weighted_object_token_vector_matrix = nn.MM(false,true){nn.View(-1,1):setNumInputDims(1)(normalized_similarity_profile),object_token_vector}

      -- at this point we update the entity matrix by adding the
      -- weighted versions of the current object token vector to each
      -- row of it (we pad the bottom of the entity matrix with a zero
      -- row, so that we can add it to the version of the current
      -- object token vector that was weighted by the new mass cell
      entity_matrix_table[i]= nn.CAddTable(){
         nn.Padding(1,1,2)(entity_matrix_table[i-1]),weighted_object_token_vector_matrix}:annotate{'entity_matrix_table' .. i}
   end
   -- end of processing input objects

   -- adding to shareList here the attribute and entity mass mappings
   table.insert(shareList,attribute_mappings)
   table.insert(shareList,raw_new_entity_mass_mappings)


   -- at this point, we take the dot product of each row (entity)
   -- vector in the entity matrix with the linguistic query vector, to
   -- obtain an entity-to-query similarity profile, that we softmax
   -- normalize (note Views needed to get right shapes, and rescaling
   -- by temperature)
   local raw_query_entity_similarity_profile = nn.View(-1):setNumInputDims(2)(nn.MM(false,false)({entity_matrix_table[inp_seq_cardinality],query}))
   local rescaled_query_entity_similarity_profile = nn.MulConstant(temperature)(raw_query_entity_similarity_profile)
   local query_entity_similarity_profile = nn.View(1,-1):setNumInputDims(1)(nn.SoftMax()(rescaled_query_entity_similarity_profile)):annotate{name='query_entity_similarity_profile'}

   -- we now do "soft retrieval" of the entity that matches the query:
   -- we obtain a vector that is a weighted sum of all the entity
   -- vectors in the entity library (weights= similarity profile, such
   -- that we will return the entity that is most similar to the
   -- query) (we get a matrix of such vectors because of mini-batches)
   local retrieved_entity_matrix = nn.MM(false,false)({query_entity_similarity_profile,entity_matrix_table[inp_seq_cardinality]})
   
   -- now we call the return_entity_image_shared function to obtain a softmax
   -- over candidate images
   local output_distribution=return_entity_image_shared(v_inp_size,mm_size,candidate_cardinality,dropout_p,inputs,token_object_mappings,retrieved_entity_matrix)
   
   -- adding token_object_mappings to shareList only now, after we also added to it the candidate image mappings
   table.insert(shareList,token_object_mappings)


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
