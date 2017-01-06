---- BACKUPS OF MODELS AND VARIATIONS WE'VE TRIED -----

function return_entity_image_no_parameters(v_inp_size,mm_size,candidate_cardinality,dropout_p,in_table,share_table,retrieved_entity_matrix)
   local image_candidate_vectors={}
   -- image candidates vectors
   for i=1,candidate_cardinality do
      local image_candidate_vector = nn.Identity()()
      table.insert(in_table,image_candidate_vector)
      table.insert(image_candidate_vectors, image_candidate_vector)
   end
   
   -- reshaping the table into a matrix with a candidate
   -- vector per row
   -- ==> second argument to JoinTable tells it that 
   -- the expected inputs in the table are one-dimensional (the
   -- candidate vectors), necessary not to confuse 
   -- it when batches are passed
   local all_candidate_values=nn.JoinTable(1,1)(image_candidate_vectors)
   -- again, note setNumInputDims for
   -- taking the dot product of each candidate vector
   -- with the retrieved_entity vector
   local candidate_matrix=nn.View(#image_candidate_vectors,-1):setNumInputDims(1)(all_candidate_values)
   local dot_vector_split=nn.MM(false,true)({retrieved_entity_matrix,candidate_matrix})
   local dot_vector=nn.View(-1):setNumInputDims(2)(dot_vector_split) -- reshaping into batch-by-nref matrix for minibatch
                                                                           -- processing
   return nn.LogSoftMax()(dot_vector)
end

function return_entity_image_bias(v_inp_size,mm_size,candidate_cardinality,dropout_p,in_table,share_table,retrieved_entity_matrix)
   local image_candidate_vectors={}
   -- image candidates vectors
   for i=1,candidate_cardinality do
      local curr_input = nn.Identity()()
      table.insert(in_table,curr_input)
      local image_candidate_vector_do = nn.Dropout(dropout_p)(curr_input)
      local image_candidate_vector = nn.Linear(v_inp_size,mm_size)(image_candidate_vector_do)
      table.insert(image_candidate_vectors, image_candidate_vector)
   end

   -- we store image_candidate_vectors within the passed share_table
   -- since they will share weights with each other (weight sharing is
   -- performed right before returning a model, since, in case we're
   -- on gpu's, it has to be done before the model is cudified)
   table.insert(share_table,image_candidate_vectors)
   
   -- reshaping the table into a matrix with a candidate
   -- vector per row
   -- ==> second argument to JoinTable tells it that 
   -- the expected inputs in the table are one-dimensional (the
   -- candidate vectors), necessary not to confuse 
   -- it when batches are passed
   local all_candidate_values=nn.JoinTable(1,1)(image_candidate_vectors)
   -- again, note setNumInputDims for
   -- taking the dot product of each candidate vector
   -- with the retrieved_entity vector
   local candidate_matrix=nn.View(#image_candidate_vectors,-1):setNumInputDims(1)(all_candidate_values)
   local dot_vector_split=nn.MM(false,true)({retrieved_entity_matrix,candidate_matrix})
   local dot_vector=nn.View(-1):setNumInputDims(2)(dot_vector_split) -- reshaping into batch-by-nref matrix for minibatch
                                                                           -- processing
   return nn.LogSoftMax()(dot_vector)
end

-- our main model -- VERY OLD VERSION
function entity_prediction_VERY_OLD(t_inp_size,v_inp_size,mm_size,inp_seq_cardinality,dropout_p)

   local inputs = {}

   -- first, we process the query, mapping it onto multimodal space

   -- the first attribute in the query
   local curr_input = nn.Identity()()
   table.insert(inputs,curr_input)
   local query_attribute_1_do = nn.Dropout(dropout_p)(curr_input)
   local query_attribute_1 = nn.LinearNB(t_inp_size, mm_size)(query_attribute_1_do):annotate{name='query_att1'}

   -- the second attribute in the query
   local curr_input = nn.Identity()()
   table.insert(inputs,curr_input)
   local query_attribute_2_do = nn.Dropout(dropout_p)(curr_input)
   local query_attribute_2 = nn.LinearNB(t_inp_size, mm_size)(query_attribute_2_do):annotate{name='query_att2'}
   -- sharing matrix with first attribute (no bias/gradBias sharing
   -- since we're not using the bias term)
   query_attribute_2.data.module:share(query_attribute_1.data.module,'weight','gradWeight')
   
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

   -- a table to store the first cell recording the "new entity" mass
   -- which will serve as the template for weight sharing of the other
   -- cells (this seems like the most elegant way, in terms of
   -- minimizing repeated call)
   local raw_new_entity_mass_template_table = {}

   -- the first object token is a special case, as it will always be
   -- directly mapped to the first row of the entity matrix
   -- first processing the attribute
   local curr_input = nn.Identity()()
   table.insert(inputs,curr_input)
   local first_token_attribute_do = nn.Dropout(dropout_p)(curr_input)
   local first_token_attribute = nn.LinearNB(t_inp_size,mm_size)(first_token_attribute_do)
   -- then processing the object image
   local curr_input = nn.Identity()()
   table.insert(inputs,curr_input)
   local first_token_object_do = nn.Dropout(dropout_p)(curr_input)
   local first_token_object = nn.LinearNB(v_inp_size,mm_size)(first_token_object_do)
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
      -- sharing the word mapping weights with the first token
      token_attribute.data.module:share(first_token_attribute.data.module,'weight','gradWeight')
      -- processing the object image
      local curr_input = nn.Identity()()
      table.insert(inputs,curr_input)
      local token_object_do = nn.Dropout(dropout_p)(curr_input)
      local token_object = nn.LinearNB(t_inp_size,mm_size)(token_object_do)
      -- parameters to be shared with first token object image
      token_object.data.module:share(first_token_object.data.module,'weight','gradWeight')
      -- putting together attribute and object 
      local object_token_vector_flat = nn.CAddTable()({token_attribute,token_object}):annotate{name='object_token_' .. i}
      -- reshaping
      local object_token_vector = nn.View(-1,1):setNumInputDims(1)(object_token_vector_flat)

      -- measuring the similarity of the current vector to the ones in
      -- the previous state of the entity matrix
      local raw_similarity_profile_to_entity_matrix = nn.MM(false,false)
      ({entity_matrix_table[i-1],object_token_vector})

      -- computing the new-entity cell value
      raw_new_entity_mass = nil
      -- average or sum input vector cells...
      if (opt.new_mass_aggregation_method=='mean') then
	 raw_new_entity_mass = nn.Linear(1,1)(nn.Mean(1,2)(raw_similarity_profile_to_entity_matrix))
      else
	 raw_new_entity_mass = nn.Linear(1,1)(nn.Sum(1,2)(raw_similarity_profile_to_entity_matrix))
      end
      if i==2 then -- this is the first cell, let's store it as a template
	 table.insert(raw_new_entity_mass_template_table,raw_new_entity_mass)
      else -- share parameters
	 raw_new_entity_mass.data.module:share(raw_new_entity_mass_template_table[1].data.module,'weight','bias','gradWeight','gradBias')
      end

      -- now, we concatenate the similarity profile with this new
      -- cell, and normalize
      -- NB: the output of the following very messy line of code is a
      -- matrix with the profile of each item in a minibatch as
      -- a ROW vector
      local normalized_similarity_profile = nn.SoftMax()(nn.View(-1):setNumInputDims(2)(nn.JoinTable(1,2)({raw_similarity_profile_to_entity_matrix,raw_new_entity_mass})))

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

   -- at this point, we take the dot product of each row (entity)
   -- vector in the entity matrix with the linguistic query vector, to
   -- obtain an entity-to-query similarity profile, that we
   -- LOG-softmax normalize (the log is there for compatibility with
   -- ClassNLLCriterion)
   local query_entity_similarity_profile = nn.LogSoftMax()(nn.View(-1):setNumInputDims(2)(nn.MM(false,false)
											  ({entity_matrix_table[inp_seq_cardinality],query})))

   -- wrapping up the model
   return nn.gModule(inputs,{query_entity_similarity_profile})

end


-- a control feed forward network from the concatenation of
-- inputs to a softmax over the output
function ff_OLD(t_inp_size,v_inp_size,h_size,inp_seq_cardinality,h_layer_count,nonlinearity,dropout_p)
   
   local inputs = {}
   local hidden_layers = {}
   
   -- the first attribute in the query
   local curr_input = nn.Identity()()
   table.insert(inputs,curr_input)
   -- the second attribute in the query
   local curr_input = nn.Identity()()
   table.insert(inputs,curr_input)
   -- the object name in the query
   local curr_input = nn.Identity()()
   table.insert(inputs,curr_input)
   
   -- now we process the candidates (object tokens)
   for i=1,inp_seq_cardinality do
      -- first an attribute
      local curr_input = nn.Identity()()
      table.insert(inputs,curr_input)
      -- then an object image
      local curr_input = nn.Identity()()
      table.insert(inputs,curr_input)
   end

   local all_input=nn.JoinTable(2,2)(inputs) -- second argument to JoinTable is for batch mode
   local InDim=t_inp_size*3+(t_inp_size+v_inp_size)*inp_seq_cardinality
   -- all_input=Peek()(all_input_for_peek)
   local first_hidden_layer_do = nn.Dropout(dropout_p)(all_input)
   --   local first_hidden_layer = nn.Linear(InDim, h_size)(nn.Peek()(first_hidden_layer_do))
   local first_hidden_layer = nn.Linear(InDim, h_size)(first_hidden_layer_do)
   
   -- gbt: todo: add check at option reading time that required nonlin is one of none, relu, tanh, sigmoid
   -- go through all layers
   local hidden_layer = first_hidden_layer
   for i=1,h_layer_count do
      if i>1 then
	 hidden_layer_do = nn.Dropout(dropout_p)(hidden_layers[i-1])
	 -- hidden_layer = nn.Linear(h_size,h_size)(nn.Peek()(hidden_layer_do))
	 hidden_layer = nn.Linear(h_size,h_size)(hidden_layer_do)
      end
      if (nonlinearity=='none') then
	 table.insert(hidden_layers,hidden_layer)
      else
	 local nonlinear_hidden_layer = nil
	 if (nonlinearity == 'relu') then
	    nonlinear_hidden_layer = nn.ReLU()(hidden_layer)
	 elseif (nonlinearity == 'tanh') then
	    nonlinear_hidden_layer = nn.Tanh()(hidden_layer)
	 else -- sigmoid is leftover option: if (nonlinearity == 'sigmoid') then
	    nonlinear_hidden_layer = nn.Sigmoid()(hidden_layer)
	 end
	 table.insert(hidden_layers,nonlinear_hidden_layer)
      end
   end

   -- now we predict from the last hidden layer via a linear projection to
   -- the number of output slots passed through the log softmax (for
   -- compatibility with the ClassNLL criterion)
   local output_distribution = nn.LogSoftMax()(nn.Linear(h_size,inp_seq_cardinality)(hidden_layers[h_layer_count]))

   -- wrapping up the model
   return nn.gModule(inputs,{output_distribution})

end
-- our main model -- version before October 2016
function entity_prediction(t_inp_size,v_inp_size,mm_size,inp_seq_cardinality,candidate_cardinality,nonlinearity,temperature,dropout_p,use_cuda)

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
   -- output of the previous linear transformations, and ensuring it
   -- will be a column vector
   local query = nn.View(-1,1):setNumInputDims(1)(nn.CAddTable()({query_attribute_1,query_attribute_2,query_object}):annotate{name='query'})

   -- adding the query attribute mappings to the table of sets sharing weights
   table.insert(shareList,query_attribute_mappings)

   -- now we process the object tokens

   -- a table to store the entity matrix as it evolves through time
   local entity_matrix_table = {}

   -- tables where to store the connections that must share weights:
   ---- token attribute mappings
   local token_attribute_mappings = {}
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
   table.insert(token_attribute_mappings,first_token_attribute)
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
      local object_token_vector = nn.View(-1,1):setNumInputDims(1)(object_token_vector_flat)

      -- measuring the similarity of the current vector to the ones in
      -- the previous state of the entity matrix
      local raw_similarity_profile_to_entity_matrix = nn.MM(false,false)
      ({entity_matrix_table[i-1],object_token_vector})

      -- computing the new-entity cell value
      -- average or sum input vector cells...
      local raw_cumulative_similarity=nil
      if (opt.new_mass_aggregation_method=='mean') then
	 raw_cumulative_similarity=nn.Mean(1,2)(raw_similarity_profile_to_entity_matrix)
      else -- sum by default
	 raw_cumulative_similarity = nn.Sum(1,2)(raw_similarity_profile_to_entity_matrix)
      end
      raw_cumulative_similarity:annotate{name='raw_cumulative_similarity_' .. i}
      -- -- debug from here
      -- -- we hard-code the raw_new_entity model
      -- local raw_new_entity_mass = nn.AddConstant(5)(nn.MulConstant(-1)(raw_cumulative_similarity)):annotate{name='raw_new_entity_mass'}
      -- -- debug to here
      local raw_new_entity_mass = nn.Linear(1,1)(raw_cumulative_similarity):annotate{name='raw_new_entity_mass_' .. i}
      table.insert(raw_new_entity_mass_mappings,raw_new_entity_mass)

      -- passing through nonlinearity if requested
      local transformed_new_entity_mass=nil
      if (nonlinearity=='none') then
      	 transformed_new_entity_mass=raw_new_entity_mass
      else
      	 local nonlinear_hidden_layer = nil
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
      local normalized_similarity_profile = nn.SoftMax()(nn.View(-1):setNumInputDims(2)(nn.JoinTable(1,2)({raw_similarity_profile_to_entity_matrix,transformed_new_entity_mass}))):annotate{name='normalized_similarity_profile_' .. i}

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

   -- putting all sets of mappings that must share weights in the shareList
   table.insert(shareList,token_attribute_mappings)
   table.insert(shareList,token_object_mappings)
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
   
   -- now we call the return_entity_image function to obtain a softmax
   -- over candidate images
   local output_distribution=return_entity_image(v_inp_size,mm_size,candidate_cardinality,dropout_p,inputs,shareList,retrieved_entity_matrix)
   
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

-- our main model in a version enforcing one-to-one projection of 
-- multimodal vectors onto reference vectors
function entity_prediction_one_to_one(t_inp_size,v_inp_size,mm_size,inp_seq_cardinality,candidate_cardinality,nonlinearity,temperature,dropout_p,use_cuda)

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
   -- output of the previous linear transformations, and ensuring it
   -- will be a column vector
   local query = nn.View(-1,1):setNumInputDims(1)(nn.CAddTable()({query_attribute_1,query_attribute_2,query_object}):annotate{name='query'})

   -- adding the query attribute mappings to the table of sets sharing weights
   table.insert(shareList,query_attribute_mappings)

   -- now we process the object tokens

   -- a table to store the entity matrix as it evolves through time
   local entity_matrix_table = {}

   -- tables where to store the connections that must share weights:
   ---- token attribute mappings
   local token_attribute_mappings = {}
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
   table.insert(token_attribute_mappings,first_token_attribute)
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

      -- at this point we update the entity matrix by concatenating
      -- the previous state of the matrix and a new row containing the
      -- new vector
      entity_matrix_table[i] = nn.JoinTable(1,2)({entity_matrix_table[i-1],object_token_vector})
   end
   -- end of processing input objects

   -- putting all sets of mappings that must share weights in the shareList
   table.insert(shareList,token_attribute_mappings)
   table.insert(shareList,token_object_mappings)


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
   
   -- now we call the return_entity_image function to obtain a softmax
   -- over candidate images
   local output_distribution=return_entity_image(v_inp_size,mm_size,candidate_cardinality,dropout_p,inputs,shareList,retrieved_entity_matrix)
   
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


-- our main model in a version enforcing one-to-one projection of 
-- multimodal vectors onto reference vectors, with sharing of all visual vectors
function entity_prediction_one_to_one_shared(t_inp_size,v_inp_size,mm_size,inp_seq_cardinality,candidate_cardinality,nonlinearity,temperature,dropout_p,use_cuda)

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
   -- output of the previous linear transformations, and ensuring it
   -- will be a column vector
   local query = nn.View(-1,1):setNumInputDims(1)(nn.CAddTable()({query_attribute_1,query_attribute_2,query_object}):annotate{name='query'})

   -- adding the query attribute mappings to the table of sets sharing weights
   table.insert(shareList,query_attribute_mappings)

   -- now we process the object tokens

   -- a table to store the entity matrix as it evolves through time
   local entity_matrix_table = {}

   -- tables where to store the connections that must share weights:
   ---- token attribute mappings
   local token_attribute_mappings = {}
   -- a table to share all image mappings
   local visual_mappings = {}

   -- the first object token is a special case, as it will always be
   -- directly mapped to the first row of the entity matrix
   -- first processing the attribute
   local curr_input = nn.Identity()()
   table.insert(inputs,curr_input)
   local first_token_attribute_do = nn.Dropout(dropout_p)(curr_input)
   local first_token_attribute = nn.LinearNB(t_inp_size,mm_size)(first_token_attribute_do)
   table.insert(token_attribute_mappings,first_token_attribute)
   -- then processing the object image
   local curr_input = nn.Identity()()
   table.insert(inputs,curr_input)
   local first_token_object_do = nn.Dropout(dropout_p)(curr_input)
   local first_token_object = nn.LinearNB(v_inp_size,mm_size)(first_token_object_do)
   table.insert(visual_mappings,first_token_object)
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
      table.insert(token_attribute_mappings,token_attribute)
      -- processing the object image
      local curr_input = nn.Identity()()
      table.insert(inputs,curr_input)
      local token_object_do = nn.Dropout(dropout_p)(curr_input)
      local token_object = nn.LinearNB(v_inp_size,mm_size)(token_object_do)
      table.insert(visual_mappings,token_object)
      -- putting together attribute and object 
      local object_token_vector_flat = nn.CAddTable()({token_attribute,token_object}):annotate{name='object_token_' .. i}
      -- reshaping
      local object_token_vector = nn.View(1,-1):setNumInputDims(1)(object_token_vector_flat)

      -- at this point we update the entity matrix by concatenating
      -- the previous state of the matrix and a new row containing the
      -- new vector
      entity_matrix_table[i] = nn.JoinTable(1,2)({entity_matrix_table[i-1],object_token_vector})
   end
   -- end of processing input objects

   -- putting attribute mappings in the shareList
   table.insert(shareList,token_attribute_mappings)

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
   local output_distribution=return_entity_image_shared(v_inp_size,mm_size,candidate_cardinality,dropout_p,inputs,visual_mappings,retrieved_entity_matrix)
   
   -- adding visual mappings to shareList now, after they have assembled all image mappings
   table.insert(shareList,visual_mappings)

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

-- our main model in a version with no backprop through time
-- inputs are projected onto a matrix directly
function entity_prediction_direct_entity_matrix(t_inp_size,v_inp_size,mm_size,inp_seq_cardinality,candidate_cardinality,nonlinearity,temperature,dropout_p,use_cuda)

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
   -- output of the previous linear transformations, and ensuring it
   -- will be a column vector
   local query = nn.View(-1,1):setNumInputDims(1)(nn.CAddTable()({query_attribute_1,query_attribute_2,query_object}):annotate{name='query'})

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


   -- at this point, we take the dot product of each row (entity)
   -- vector in the entity matrix with the linguistic query vector, to
   -- obtain an entity-to-query similarity profile, that we softmax
   -- normalize (note Views needed to get right shapes, and rescaling
   -- by temperature)
   local raw_query_entity_similarity_profile = nn.View(-1):setNumInputDims(2)(nn.MM(false,false)({entity_matrix,query}))
   local rescaled_query_entity_similarity_profile = nn.MulConstant(temperature)(raw_query_entity_similarity_profile)
   local query_entity_similarity_profile = nn.View(1,-1):setNumInputDims(1)(nn.SoftMax()(rescaled_query_entity_similarity_profile)):annotate{name='query_entity_similarity_profile'}

   -- we now do "soft retrieval" of the entity that matches the query:
   -- we obtain a vector that is a weighted sum of all the entity
   -- vectors in the entity library (weights= similarity profile, such
   -- that we will return the entity that is most similar to the
   -- query) (we get a matrix of such vectors because of mini-batches)
   local retrieved_entity_matrix = nn.MM(false,false)({query_entity_similarity_profile,entity_matrix})
   
   -- now we call the return_entity_image function to obtain a softmax
   -- over candidate images
   local output_distribution=return_entity_image(v_inp_size,mm_size,candidate_cardinality,dropout_p,inputs,shareList,retrieved_entity_matrix)
   
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

-- our main model in a version with no backprop through time
-- inputs are projected onto a matrix directly, all image parameters are shared
function entity_prediction_direct_entity_matrix_shared(t_inp_size,v_inp_size,mm_size,inp_seq_cardinality,candidate_cardinality,nonlinearity,temperature,dropout_p,use_cuda)

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
   -- output of the previous linear transformations, and ensuring it
   -- will be a column vector
   local query = nn.View(-1,1):setNumInputDims(1)(nn.CAddTable()({query_attribute_1,query_attribute_2,query_object}):annotate{name='query'})

   -- adding the query attribute mappings to the table of sets sharing weights
   table.insert(shareList,query_attribute_mappings)

   -- now we process the object tokens

   -- a table to store the entity matrix
   local entity_matrix_table = {}

   -- tables where to store the attribute mapping weights
   local token_attribute_mappings = {}

   -- a table to share all image mappings
   local visual_mappings = {}

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
      table.insert(visual_mappings,token_object)
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

   -- putting the token attribute mappings in the shared-weight sets shareList
   table.insert(shareList,token_attribute_mappings)

   -- at this point, we take the dot product of each row (entity)
   -- vector in the entity matrix with the linguistic query vector, to
   -- obtain an entity-to-query similarity profile, that we softmax
   -- normalize (note Views needed to get right shapes, and rescaling
   -- by temperature)
   local raw_query_entity_similarity_profile = nn.View(-1):setNumInputDims(2)(nn.MM(false,false)({entity_matrix,query}))
   local rescaled_query_entity_similarity_profile = nn.MulConstant(temperature)(raw_query_entity_similarity_profile)
   local query_entity_similarity_profile = nn.View(1,-1):setNumInputDims(1)(nn.SoftMax()(rescaled_query_entity_similarity_profile)):annotate{name='query_entity_similarity_profile'}

   -- we now do "soft retrieval" of the entity that matches the query:
   -- we obtain a vector that is a weighted sum of all the entity
   -- vectors in the entity library (weights= similarity profile, such
   -- that we will return the entity that is most similar to the
   -- query) (we get a matrix of such vectors because of mini-batches)
   local retrieved_entity_matrix = nn.MM(false,false)({query_entity_similarity_profile,entity_matrix})
   
   -- now we call the return_entity_image_shared function to obtain a softmax
   -- over candidate images
   local output_distribution=return_entity_image_shared(v_inp_size,mm_size,candidate_cardinality,dropout_p,inputs,visual_mappings,retrieved_entity_matrix)

   -- adding visual mappings to shareList now, after they have assembled all image mappings
      table.insert(shareList,visual_mappings)
   
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

-- our main model in a version with no backprop through time
-- inputs are projected onto a matrix directly, all image parameters are shared (assumes query is actually in image space as well)
function entity_prediction_direct_entity_matrix_shared_hack(t_inp_size,v_inp_size,mm_size,inp_seq_cardinality,candidate_cardinality,nonlinearity,temperature,dropout_p,use_cuda)

   local inputs = {}

   -- a table to store tables of connections that must share parameters
   local shareList = {}

   -- table to collect query attribute mappings, to be shared
   local query_attribute_mappings = {}

   -- a table to share all image mappings
   local visual_mappings = {}

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
   -- NB: here we assume t_inp_size==v_inp_size
   table.insert(visual_mappings,query_object)

   -- putting together the multimodal query vector by summing the
   -- output of the previous linear transformations, and ensuring it
   -- will be a column vector
   local query = nn.View(-1,1):setNumInputDims(1)(nn.CAddTable()({query_attribute_1,query_attribute_2,query_object}):annotate{name='query'})

   -- adding the query attribute mappings to the table of sets sharing weights
   table.insert(shareList,query_attribute_mappings)

   -- now we process the object tokens

   -- a table to store the entity matrix
   local entity_matrix_table = {}

   -- tables where to store the attribute mapping weights
   local token_attribute_mappings = {}

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
      table.insert(visual_mappings,token_object)
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

   -- putting the token attribute mappings in the shared-weight sets shareList
   table.insert(shareList,token_attribute_mappings)

   -- at this point, we take the dot product of each row (entity)
   -- vector in the entity matrix with the linguistic query vector, to
   -- obtain an entity-to-query similarity profile, that we softmax
   -- normalize (note Views needed to get right shapes, and rescaling
   -- by temperature)
   local raw_query_entity_similarity_profile = nn.View(-1):setNumInputDims(2)(nn.MM(false,false)({entity_matrix,query}))
   local rescaled_query_entity_similarity_profile = nn.MulConstant(temperature)(raw_query_entity_similarity_profile)
   local query_entity_similarity_profile = nn.View(1,-1):setNumInputDims(1)(nn.SoftMax()(rescaled_query_entity_similarity_profile)):annotate{name='query_entity_similarity_profile'}

   -- we now do "soft retrieval" of the entity that matches the query:
   -- we obtain a vector that is a weighted sum of all the entity
   -- vectors in the entity library (weights= similarity profile, such
   -- that we will return the entity that is most similar to the
   -- query) (we get a matrix of such vectors because of mini-batches)
   local retrieved_entity_matrix = nn.MM(false,false)({query_entity_similarity_profile,entity_matrix})
   
   -- now we call the return_entity_image function to obtain a softmax
   -- over candidate images
   local output_distribution=return_entity_image_shared(v_inp_size,mm_size,candidate_cardinality,dropout_p,inputs,visual_mappings,retrieved_entity_matrix)

   -- adding visual mappings to shareList now, after they have assembled all image mappings
      table.insert(shareList,visual_mappings)
   
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



-- our main model in a version with no embedding matrices
function entity_prediction_no_parameters(t_inp_size,v_inp_size,mm_size,inp_seq_cardinality,candidate_cardinality,nonlinearity,temperature,dropout_p,use_cuda)

   local inputs = {}


   -- first, we process the query, storing it directly into the entity matrix

   -- the first attribute in the query
   local query_attribute_1 = nn.Identity()()
   table.insert(inputs,query_attribute_1)
   -- the second attribute in the query
   local query_attribute_2 = nn.Identity()()
   table.insert(inputs,query_attribute_2)
   -- the object name in the query
   local query_object = nn.Identity()()
   table.insert(inputs,query_object)

   -- putting together the multimodal query vector by summing the read
   -- vectors, and ensuring the result will be a column vector
   local query = nn.View(-1,1):setNumInputDims(1)(nn.CAddTable()({query_attribute_1,query_attribute_2,query_object}):annotate{name='query'})

   -- now we process the object tokens

   -- a table to store the entity matrix
   local entity_matrix_table = {}

   -- now we process all the object tokens in a loop
   for i=1,inp_seq_cardinality do
      -- processing the attribute
      local token_attribute = nn.Identity()()
      table.insert(inputs,token_attribute)
      -- processing the object image
      local token_object = nn.Identity()()
      table.insert(inputs,token_object)
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

   -- at this point, we take the dot product of each row (entity)
   -- vector in the entity matrix with the linguistic query vector, to
   -- obtain an entity-to-query similarity profile, that we softmax
   -- normalize (note Views needed to get right shapes, and rescaling
   -- by temperature)
   local raw_query_entity_similarity_profile = nn.View(-1):setNumInputDims(2)(nn.MM(false,false)({entity_matrix,query}))
   local rescaled_query_entity_similarity_profile = nn.MulConstant(temperature)(raw_query_entity_similarity_profile)
   local query_entity_similarity_profile = nn.View(1,-1):setNumInputDims(1)(nn.SoftMax()(rescaled_query_entity_similarity_profile)):annotate{name='query_entity_similarity_profile'}

   -- we now do "soft retrieval" of the entity that matches the query:
   -- we obtain a vector that is a weighted sum of all the entity
   -- vectors in the entity library (weights= similarity profile, such
   -- that we will return the entity that is most similar to the
   -- query) (we get a matrix of such vectors because of mini-batches)
   local retrieved_entity_matrix = nn.MM(false,false)({query_entity_similarity_profile,entity_matrix})
   
   -- now we call the return_entity_image function to obtain a softmax
   -- over candidate images
   local output_distribution=return_entity_image_no_parameters(v_inp_size,mm_size,candidate_cardinality,dropout_p,inputs,shareList,retrieved_entity_matrix)
   
   -- wrapping up the model
   local model = nn.gModule(inputs,{output_distribution})
   
   -- following code is adapted from MeMNN 
   if (use_cuda ~= 0) then
      model:cuda()
   end
   return model

end


-- our main model with two entity matrices, one used to track and
-- query entities, the other to produce a probe to query the output
function entity_prediction_two_libraries(t_inp_size,v_inp_size,mm_size,inp_seq_cardinality,candidate_cardinality,nonlinearity,temperature,dropout_p,use_cuda)

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
   -- output of the previous linear transformations, and ensuring it
   -- will be a column vector
   local query = nn.View(-1,1):setNumInputDims(1)(nn.CAddTable()({query_attribute_1,query_attribute_2,query_object}):annotate{name='query'})

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
   --- mappings to raw new entity mass
   local raw_new_entity_mass_mappings = {}

   -- the first object token is a special case, as it will always be
   -- directly mapped to the first row of the entity matrix
   -- first processing the attribute
   local curr_input = nn.Identity()()
   table.insert(inputs,curr_input)
   local first_token_attribute_do = nn.Dropout(dropout_p)(curr_input)
   local first_token_attribute = nn.LinearNB(t_inp_size,mm_size)(first_token_attribute_do)
   table.insert(token_attribute_mappings,first_token_attribute)
   local output_first_token_attribute_do = nn.Dropout(dropout_p)(curr_input)
   local output_first_token_attribute = nn.LinearNB(t_inp_size,mm_size)(output_first_token_attribute_do)
   table.insert(output_token_attribute_mappings,output_first_token_attribute)

   -- then processing the object image
   local curr_input = nn.Identity()()
   table.insert(inputs,curr_input)
   local first_token_object_do = nn.Dropout(dropout_p)(curr_input)
   local first_token_object = nn.LinearNB(v_inp_size,mm_size)(first_token_object_do)
   table.insert(token_object_mappings,first_token_object)
   local output_first_token_object_do = nn.Dropout(dropout_p)(curr_input)
   local output_first_token_object = nn.LinearNB(v_inp_size,mm_size)(output_first_token_object_do)
   table.insert(output_token_object_mappings,output_first_token_object)
   -- putting together attribute and object 
   local first_object_token_vector = nn.CAddTable()({first_token_attribute,first_token_object})
   local output_first_object_token_vector = nn.CAddTable()({output_first_token_attribute,output_first_token_object})
   -- turning the vector into a 1xmm_size matrix, adding the latter as
   -- first state of the entity matrix table
   table.insert(entity_matrix_table,nn.View(1,-1):setNumInputDims(1)(first_object_token_vector))
   table.insert(output_entity_matrix_table,nn.View(1,-1):setNumInputDims(1)(output_first_object_token_vector))
   -- now we process all the other object tokens in a loop
   for i=2,inp_seq_cardinality do
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
      -- reshaping
      local object_token_vector = nn.View(-1,1):setNumInputDims(1)(object_token_vector_flat)
      local output_object_token_vector = nn.View(-1,1):setNumInputDims(1)(output_object_token_vector_flat)

      -- measuring the similarity of the current vector to the ones in
      -- the previous state of the entity matrix
      local raw_similarity_profile_to_entity_matrix = nn.MM(false,false)
      ({entity_matrix_table[i-1],object_token_vector})

      -- computing the new-entity cell value
      -- average or sum input vector cells...
      local raw_cumulative_similarity=nil
      if (opt.new_mass_aggregation_method=='mean') then
	 raw_cumulative_similarity=nn.Mean(1,2)(raw_similarity_profile_to_entity_matrix)
      elseif (opt.new_mass_aggregation_method=='max') then
	 raw_cumulative_similarity=nn.Max(1,2)(raw_similarity_profile_to_entity_matrix)
      else -- sum by default
	 raw_cumulative_similarity = nn.Sum(1,2)(raw_similarity_profile_to_entity_matrix)
      end
      raw_cumulative_similarity:annotate{name='raw_cumulative_similarity_' .. i}
      -- -- debug from here
      -- -- we hard-code the raw_new_entity model
      -- local raw_new_entity_mass = nn.AddConstant(5)(nn.MulConstant(-1)(raw_cumulative_similarity)):annotate{name='raw_new_entity_mass'}
      -- -- debug to here
      local raw_new_entity_mass = nn.Linear(1,1)(raw_cumulative_similarity):annotate{name='raw_new_entity_mass_' .. i}
      table.insert(raw_new_entity_mass_mappings,raw_new_entity_mass)

      -- passing through nonlinearity if requested
      local transformed_new_entity_mass=nil
      if (nonlinearity=='none') then
      	 transformed_new_entity_mass=raw_new_entity_mass
      else
      	 local nonlinear_hidden_layer = nil
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
      local normalized_similarity_profile = nn.SoftMax()(nn.View(-1):setNumInputDims(2)(nn.JoinTable(1,2)({raw_similarity_profile_to_entity_matrix,transformed_new_entity_mass}))):annotate{name='normalized_similarity_profile_' .. i}

      -- we now create a matrix that has, on each ROW, the current
      -- token vector, multiplied by the corresponding entry on the
      -- normalized similarity profile (including, in the final row,
      -- weighting by the normalized new mass cell):
      local reshaped_normalized_similarity_profile = nn.View(-1,1):setNumInputDims(1)(normalized_similarity_profile)
      local weighted_object_token_vector_matrix = nn.MM(false,true){reshaped_normalized_similarity_profile,object_token_vector}
      -- we repeat the same operation but now using the output representation of the object token vector
      local output_weighted_object_token_vector_matrix = nn.MM(false,true){reshaped_normalized_similarity_profile,output_object_token_vector}

      -- at this point we update the entity matrix by adding the
      -- weighted versions of the current object token vector to each
      -- row of it (we pad the bottom of the entity matrix with a zero
      -- row, so that we can add it to the version of the current
      -- object token vector that was weighted by the new mass cell)
      entity_matrix_table[i]= nn.CAddTable(){
	 nn.Padding(1,1,2)(entity_matrix_table[i-1]),weighted_object_token_vector_matrix}:annotate{'entity_matrix_table' .. i}
   -- and similarly for the output entity matrix
      output_entity_matrix_table[i]= nn.CAddTable(){
	 nn.Padding(1,1,2)(output_entity_matrix_table[i-1]),output_weighted_object_token_vector_matrix}:annotate{'output_entity_matrix_table' .. i}
   end
   -- end of processing input objects

   -- putting all sets of mappings that must share weights in the shareList
   table.insert(shareList,token_attribute_mappings)
   table.insert(shareList,token_object_mappings)
   table.insert(shareList,output_token_attribute_mappings)
   table.insert(shareList,output_token_object_mappings)
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
   -- vectors in the OUTPUT entity library (weights= similarity profile, such
   -- that we will return the entity that is most similar to the
   -- query) (we get a matrix of such vectors because of mini-batches)
   local retrieved_entity_matrix = nn.MM(false,false)({query_entity_similarity_profile,output_entity_matrix_table[inp_seq_cardinality]})
   
   -- now we call the return_entity_image function to obtain a softmax
   -- over candidate images
   local output_distribution=return_entity_image(v_inp_size,mm_size,candidate_cardinality,dropout_p,inputs,shareList,retrieved_entity_matrix)
   
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



-- our main model with a special mapping to obtain the probe vector
function entity_prediction_probe(t_inp_size,v_inp_size,mm_size,inp_seq_cardinality,candidate_cardinality,nonlinearity,temperature,dropout_p,use_cuda)

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
   -- output of the previous linear transformations, and ensuring it
   -- will be a column vector
   local query = nn.View(-1,1):setNumInputDims(1)(nn.CAddTable()({query_attribute_1,query_attribute_2,query_object}):annotate{name='query'})

   -- adding the query attribute mappings to the table of sets sharing weights
   table.insert(shareList,query_attribute_mappings)

   -- now we process the object tokens

   -- a table to store the entity matrix as it evolves through time
   local entity_matrix_table = {}

   -- tables where to store the connections that must share weights:
   ---- token attribute mappings
   local token_attribute_mappings = {}
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
   table.insert(token_attribute_mappings,first_token_attribute)
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
      local object_token_vector = nn.View(-1,1):setNumInputDims(1)(object_token_vector_flat)

      -- measuring the similarity of the current vector to the ones in
      -- the previous state of the entity matrix
      local raw_similarity_profile_to_entity_matrix = nn.MM(false,false)
      ({entity_matrix_table[i-1],object_token_vector})

      -- computing the new-entity cell value
      -- average or sum input vector cells...
      local raw_cumulative_similarity=nil
      if (opt.new_mass_aggregation_method=='mean') then
	 raw_cumulative_similarity=nn.Mean(1,2)(raw_similarity_profile_to_entity_matrix)
      else -- sum by default
	 raw_cumulative_similarity = nn.Sum(1,2)(raw_similarity_profile_to_entity_matrix)
      end
      raw_cumulative_similarity:annotate{name='raw_cumulative_similarity_' .. i}
      -- -- debug from here
      -- -- we hard-code the raw_new_entity model
      -- local raw_new_entity_mass = nn.AddConstant(5)(nn.MulConstant(-1)(raw_cumulative_similarity)):annotate{name='raw_new_entity_mass'}
      -- -- debug to here
      local raw_new_entity_mass = nn.Linear(1,1)(raw_cumulative_similarity):annotate{name='raw_new_entity_mass_' .. i}
      table.insert(raw_new_entity_mass_mappings,raw_new_entity_mass)

      -- passing through nonlinearity if requested
      local transformed_new_entity_mass=nil
      if (nonlinearity=='none') then
      	 transformed_new_entity_mass=raw_new_entity_mass
      else
      	 local nonlinear_hidden_layer = nil
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
      local normalized_similarity_profile = nn.SoftMax()(nn.View(-1):setNumInputDims(2)(nn.JoinTable(1,2)({raw_similarity_profile_to_entity_matrix,transformed_new_entity_mass}))):annotate{name='normalized_similarity_profile_' .. i}

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

   -- putting all sets of mappings that must share weights in the shareList
   table.insert(shareList,token_attribute_mappings)
   table.insert(shareList,token_object_mappings)
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

   -- now, for this version of the model, we will map this retrieved
   -- entity vector onto another "probe" vector to be compared to the
   -- output (note usual View sillyness)
   local probe_vector_matrix = nn.View(1,-1):setNumInputDims(1)(nn.LinearNB(mm_size,mm_size)(nn.View(-1):setNumInputDims(2)(retrieved_entity_matrix)))


   -- now we call the return_entity_image function to obtain a softmax
   -- over candidate images
   local output_distribution=return_entity_image(v_inp_size,mm_size,candidate_cardinality,dropout_p,inputs,shareList,probe_vector_matrix)
   
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

-- our main model sharing all image embeddings
function entity_prediction_image_shared(t_inp_size,v_inp_size,mm_size,inp_seq_cardinality,candidate_cardinality,nonlinearity,temperature,dropout_p,use_cuda)

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
   -- output of the previous linear transformations, and ensuring it
   -- will be a column vector
   local query = nn.View(-1,1):setNumInputDims(1)(nn.CAddTable()({query_attribute_1,query_attribute_2,query_object}):annotate{name='query'})

   -- adding the query attribute mappings to the table of sets sharing weights
   table.insert(shareList,query_attribute_mappings)

   -- now we process the object tokens

   -- a table to store the entity matrix as it evolves through time
   local entity_matrix_table = {}

   -- tables where to store the connections that must share weights:
   ---- token attribute mappings
   local token_attribute_mappings = {}
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
   table.insert(token_attribute_mappings,first_token_attribute)
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
      local object_token_vector = nn.View(-1,1):setNumInputDims(1)(object_token_vector_flat)

      -- measuring the similarity of the current vector to the ones in
      -- the previous state of the entity matrix
      local raw_similarity_profile_to_entity_matrix = nn.MM(false,false)
      ({entity_matrix_table[i-1],object_token_vector})

      -- computing the new-entity cell value
      -- average or sum input vector cells...
      local raw_cumulative_similarity=nil
      if (opt.new_mass_aggregation_method=='mean') then
	 raw_cumulative_similarity=nn.Mean(1,2)(raw_similarity_profile_to_entity_matrix)
      elseif (opt.new_mass_aggregation_method=='max') then
	 raw_cumulative_similarity=nn.Max(1,2)(raw_similarity_profile_to_entity_matrix)
      else -- sum by default
	 raw_cumulative_similarity = nn.Sum(1,2)(raw_similarity_profile_to_entity_matrix)
      end
      raw_cumulative_similarity:annotate{name='raw_cumulative_similarity_' .. i}
      -- -- debug from here
      -- -- we hard-code the raw_new_entity model
      -- local raw_new_entity_mass = nn.AddConstant(5)(nn.MulConstant(-1)(raw_cumulative_similarity)):annotate{name='raw_new_entity_mass'}
      -- -- debug to here
      local raw_new_entity_mass = nn.Linear(1,1)(raw_cumulative_similarity):annotate{name='raw_new_entity_mass_' .. i}
      table.insert(raw_new_entity_mass_mappings,raw_new_entity_mass)

      -- passing through nonlinearity if requested
      local transformed_new_entity_mass=nil
      if (nonlinearity=='none') then
      	 transformed_new_entity_mass=raw_new_entity_mass
      else
      	 local nonlinear_hidden_layer = nil
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
      local normalized_similarity_profile = nn.SoftMax()(nn.View(-1):setNumInputDims(2)(nn.JoinTable(1,2)({raw_similarity_profile_to_entity_matrix,transformed_new_entity_mass}))):annotate{name='normalized_similarity_profile_' .. i}

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

   -- putting all sets of verbal mappings that must share weights in the shareList
   table.insert(shareList,token_attribute_mappings)
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

-- our main model sharing all image embeddings and all attribute
-- embeddings but accepting a different dropout rate for the
-- attributes, to make their matching less perfect
function entity_prediction_image_att_do_shared(t_inp_size,v_inp_size,mm_size,inp_seq_cardinality,candidate_cardinality,nonlinearity,temperature,dropout_p,att_dropout_p,use_cuda)

   local inputs = {}

   -- a table to store tables of connections that must share parameters
   local shareList = {}

   -- table to collect all attribute mappings, to be shared
   local attribute_mappings = {}

   -- first, we process the query, mapping it onto multimodal space

   -- the first attribute in the query
   local curr_input = nn.Identity()()
   table.insert(inputs,curr_input)
   local query_attribute_1_do = nn.Dropout(att_dropout_p)(curr_input)
   local query_attribute_1 = nn.LinearNB(t_inp_size, mm_size)(query_attribute_1_do):annotate{name='query_att1'}
   table.insert(attribute_mappings,query_attribute_1)

   -- the second attribute in the query
   local curr_input = nn.Identity()()
   table.insert(inputs,curr_input)
   local query_attribute_2_do = nn.Dropout(att_dropout_p)(curr_input)
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
   local first_token_attribute_do = nn.Dropout(att_dropout_p)(curr_input)
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
      local token_attribute_do = nn.Dropout(att_dropout_p)(curr_input)
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
      -- -- debug from here
      -- -- we hard-code the raw_new_entity model
      -- local raw_new_entity_mass = nn.AddConstant(5)(nn.MulConstant(-1)(raw_cumulative_similarity)):annotate{name='raw_new_entity_mass'}
      -- -- debug to here
      local raw_new_entity_mass = nn.Linear(1,1)(raw_cumulative_similarity):annotate{name='raw_new_entity_mass_' .. i}
      table.insert(raw_new_entity_mass_mappings,raw_new_entity_mass)

      -- passing through nonlinearity if requested
      local transformed_new_entity_mass=nil
      if (nonlinearity=='none') then
      	 transformed_new_entity_mass=raw_new_entity_mass
      else
      	 local nonlinear_hidden_layer = nil
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
      local normalized_similarity_profile = nn.SoftMax()(nn.View(-1):setNumInputDims(2)(nn.JoinTable(1,2)({raw_similarity_profile_to_entity_matrix,transformed_new_entity_mass}))):annotate{name='normalized_similarity_profile_' .. i}

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

-- our main model with bias
function entity_prediction_bias(t_inp_size,v_inp_size,mm_size,inp_seq_cardinality,candidate_cardinality,nonlinearity,temperature,dropout_p,use_cuda)

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
   local query_attribute_1 = nn.Linear(t_inp_size, mm_size)(query_attribute_1_do):annotate{name='query_att1'}
   table.insert(query_attribute_mappings,query_attribute_1)

   -- the second attribute in the query
   local curr_input = nn.Identity()()
   table.insert(inputs,curr_input)
   local query_attribute_2_do = nn.Dropout(dropout_p)(curr_input)
   local query_attribute_2 = nn.Linear(t_inp_size, mm_size)(query_attribute_2_do):annotate{name='query_att2'}
   table.insert(query_attribute_mappings,query_attribute_2)
   
   -- the object name in the query
   local curr_input = nn.Identity()()
   table.insert(inputs,curr_input)
   local query_object_do = nn.Dropout(dropout_p)(curr_input)
   local query_object = nn.Linear(t_inp_size, mm_size)(query_object_do):annotate{name='query_object'}

   -- putting together the multimodal query vector by summing the
   -- output of the previous linear transformations, and ensuring it
   -- will be a column vector
   local query = nn.View(-1,1):setNumInputDims(1)(nn.CAddTable()({query_attribute_1,query_attribute_2,query_object}):annotate{name='query'})

   -- adding the query attribute mappings to the table of sets sharing weights
   table.insert(shareList,query_attribute_mappings)

   -- now we process the object tokens

   -- a table to store the entity matrix as it evolves through time
   local entity_matrix_table = {}

   -- tables where to store the connections that must share weights:
   ---- token attribute mappings
   local token_attribute_mappings = {}
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
   local first_token_attribute = nn.Linear(t_inp_size,mm_size)(first_token_attribute_do)
   table.insert(token_attribute_mappings,first_token_attribute)
   -- then processing the object image
   local curr_input = nn.Identity()()
   table.insert(inputs,curr_input)
   local first_token_object_do = nn.Dropout(dropout_p)(curr_input)
   local first_token_object = nn.Linear(v_inp_size,mm_size)(first_token_object_do)
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
      local token_attribute = nn.Linear(t_inp_size,mm_size)(token_attribute_do)
      table.insert(token_attribute_mappings,token_attribute)
      -- processing the object image
      local curr_input = nn.Identity()()
      table.insert(inputs,curr_input)
      local token_object_do = nn.Dropout(dropout_p)(curr_input)
      local token_object = nn.Linear(v_inp_size,mm_size)(token_object_do)
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
      -- average or sum input vector cells...
      local raw_cumulative_similarity=nil
      if (opt.new_mass_aggregation_method=='mean') then
	 raw_cumulative_similarity=nn.Mean(1,2)(raw_similarity_profile_to_entity_matrix)
      else -- sum by default
	 raw_cumulative_similarity = nn.Sum(1,2)(raw_similarity_profile_to_entity_matrix)
      end
      raw_cumulative_similarity:annotate{name='raw_cumulative_similarity_' .. i}
      -- -- debug from here
      -- -- we hard-code the raw_new_entity model
      -- local raw_new_entity_mass = nn.AddConstant(5)(nn.MulConstant(-1)(raw_cumulative_similarity)):annotate{name='raw_new_entity_mass'}
      -- -- debug to here
      local raw_new_entity_mass = nn.Linear(1,1)(raw_cumulative_similarity):annotate{name='raw_new_entity_mass_' .. i}
      table.insert(raw_new_entity_mass_mappings,raw_new_entity_mass)


      -- passing through nonlinearity if requested
      local transformed_new_entity_mass=nil
      if (nonlinearity=='none') then
      	 transformed_new_entity_mass=raw_new_entity_mass
      else
      	 local nonlinear_hidden_layer = nil
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
      local normalized_similarity_profile = nn.SoftMax()(nn.View(-1):setNumInputDims(2)(nn.JoinTable(1,2)({raw_similarity_profile_to_entity_matrix,transformed_new_entity_mass}))):annotate{name='normalized_similarity_profile_' .. i}

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

   -- putting all sets of mappings that must share weights in the shareList
   table.insert(shareList,token_attribute_mappings)
   table.insert(shareList,token_object_mappings)
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
   
   -- now we call the return_entity_image function to obtain a softmax
   -- over candidate images
   local output_distribution=return_entity_image_bias(v_inp_size,mm_size,candidate_cardinality,dropout_p,inputs,shareList,retrieved_entity_matrix)
   
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
