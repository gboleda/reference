function return_entity_image_OLD(v_inp_size,mm_size,candidate_cardinality,dropout_p,in_table,retrieved_entity_matrix)
   local image_candidate_vectors={}
   -- image candidates vectors
   for i=1,candidate_cardinality do
      local curr_input = nn.Identity()()
      table.insert(in_table,curr_input)
      local image_candidate_vector_do = nn.Dropout(dropout_p)(curr_input)
      local image_candidate_vector = nn.LinearNB(v_inp_size,mm_size)(image_candidate_vector_do)
      if i>1 then -- share parameters of each img cand vector
	 image_candidate_vector.data.module:share(image_candidate_vectors[1].data.module,'weight','gradWeight')
      end
      table.insert(image_candidate_vectors, image_candidate_vector)
   end
   
   -- reshaping the table into a matrix with a candidate
   -- vector per row
   -- ==> second argument to JoinTable tells it that 
   -- the expected inputs in the table are one-dimensional (the
   -- candidate vectors), necessary not to confuse 
   -- it when batches are passed
   -- local all_candidate_values=nn.JoinTable(1,1)(nn.Peek()(image_candidate_vectors))
   local all_candidate_values=nn.JoinTable(1,1)(image_candidate_vectors)

   -- again, note setNumInputDims for
   -- taking the dot product of each candidate vector
   -- with the retrieved_entity vector
   local candidate_matrix=nn.View(#image_candidate_vectors,-1):setNumInputDims(1)(all_candidate_values)
   -- local dot_vector_split=nn.MM(false,true)(nn.Peek()({retrieved_entity_matrix,candidate_matrix}))
   local dot_vector_split=nn.MM(false,true)({retrieved_entity_matrix,candidate_matrix})
   local dot_vector=nn.View(-1):setNumInputDims(2)(dot_vector_split) -- reshaping into batch-by-nref matrix for minibatch
                                                                           -- processing
   return nn.LogSoftMax()(dot_vector)
end

function return_entity_image(v_inp_size,mm_size,candidate_cardinality,dropout_p,in_table,share_table,retrieved_entity_matrix)
   local image_candidate_vectors={}
   -- image candidates vectors
   for i=1,candidate_cardinality do
      local curr_input = nn.Identity()()
      table.insert(in_table,curr_input)
      local image_candidate_vector_do = nn.Dropout(dropout_p)(curr_input)
      local image_candidate_vector = nn.LinearNB(v_inp_size,mm_size)(image_candidate_vector_do)
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


-- our main model
function entity_prediction(t_inp_size,v_inp_size,mm_size,inp_seq_cardinality,candidate_cardinality,dropout_p,use_cuda)

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
      raw_new_entity_mass = nil
      -- average or sum input vector cells...
      if (opt.new_mass_aggregation_method=='mean') then
	 raw_new_entity_mass = nn.Linear(1,1)(nn.Mean(1,2)(raw_similarity_profile_to_entity_matrix))
      else
	 raw_new_entity_mass = nn.Linear(1,1)(nn.Sum(1,2)(raw_similarity_profile_to_entity_matrix))
      end
      table.insert(raw_new_entity_mass_mappings,raw_new_entity_mass)

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
   -- end of processing input objects

   -- putting all sets of mappings that must share weights in the shareList
   table.insert(shareList,token_attribute_mappings)
   table.insert(shareList,token_object_mappings)
   table.insert(shareList,raw_new_entity_mass_mappings)


   -- at this point, we take the dot product of each row (entity)
   -- vector in the entity matrix with the linguistic query vector, to
   -- obtain an entity-to-query similarity profile, that we
   -- softmax normalize (note Views needed to get right shapes)
   local raw_query_entity_similarity_profile = nn.View(-1):setNumInputDims(2)(nn.MM(false,false)({entity_matrix_table[inp_seq_cardinality],query}))
   local query_entity_similarity_profile = nn.View(1,-1):setNumInputDims(1)(nn.SoftMax()(raw_query_entity_similarity_profile))
   
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


----- BACKUP FROM HERE
-- our main model
function entity_prediction_BACKUP(t_inp_size,v_inp_size,mm_size,inp_seq_cardinality,candidate_cardinality,dropout_p,use_cuda)

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
      local token_object = nn.LinearNB(v_inp_size,mm_size)(token_object_do) -- gbt: why not v_inp_size? why doesn't it break down?
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
   -- end of processing input objects


   -- at this point, we take the dot product of each row (entity)
   -- vector in the entity matrix with the linguistic query vector, to
   -- obtain an entity-to-query similarity profile, that we
   -- softmax normalize (note Views needed to get right shapes)
   local raw_query_entity_similarity_profile = nn.View(-1):setNumInputDims(2)(nn.MM(false,false)({entity_matrix_table[inp_seq_cardinality],query}))
   local query_entity_similarity_profile = nn.View(1,-1):setNumInputDims(1)(nn.SoftMax()(raw_query_entity_similarity_profile))
   
   -- we now do "soft retrieval" of the entity that matches the query:
   -- we obtain a vector that is a weighted sum of all the entity
   -- vectors in the entity library (weights= similarity profile, such
   -- that we will return the entity that is most similar to the
   -- query) (we get a matrix of such vectors because of mini-batches)
   local retrieved_entity_matrix = nn.MM(false,false)({query_entity_similarity_profile,entity_matrix_table[inp_seq_cardinality]})
   
   -- now we call the return_entity_image function to obtain a softmax
   -- over candidate images
   local output_distribution=return_entity_image_OLD(v_inp_size,mm_size,candidate_cardinality,dropout_p,inputs,retrieved_entity_matrix)

   -- wrapping up the model
   return nn.gModule(inputs,{output_distribution})
end

----- BACKUP TO HERE


-- a control feed forward network from the concatenation of
-- inputs to a softmax over the output
function ff(t_inp_size,v_inp_size,mm_size,h_size,inp_seq_cardinality,candidate_cardinality,h_layer_count,nonlinearity,dropout_p,use_cuda)
   
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
   local first_hidden_layer_do = nn.Dropout(dropout_p)(all_input)
   local first_hidden_layer = nn.Linear(InDim, h_size)(first_hidden_layer_do)
   
   -- gbt: todo: add check at option reading time that required nonlin is one of none, relu, tanh, sigmoid
   -- go through all layers
   local hidden_layer = first_hidden_layer
   for i=1,h_layer_count do
      if i>1 then
	 hidden_layer_do = nn.Dropout(dropout_p)(hidden_layers[i-1])
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

   -- now we map from the last hidden layer to a vector that will represent our
   -- retrieved entity vector (we get a matrix of such vectors because of mini-batches)
   local retrieved_entity_matrix_2D_do = nn.Dropout(dropout_p)(hidden_layers[h_layer_count])
   local retrieved_entity_matrix_2D = nn.Linear(h_size,mm_size)(retrieved_entity_matrix_2D_do)
   -- local retrieved_entity_matrix=nn.View(-1):setNumInputDims(2)(retrieved_entity_matrix_2D)
   local retrieved_entity_matrix=nn.Reshape(1,mm_size,true)(retrieved_entity_matrix_2D) -- reshaping to minibatch x 1 x mm_size for dot product with candidate image vectors in return_entity_image function

   -- the function return_entity_image assumes a share
   -- table to be updated with modules sharing their parameters, so we
   -- must initialize shareList
   local shareList = {}
   
   -- now we call the return_entity_image function to obtain a softmax
   -- over candidate images
   local output_distribution=return_entity_image(v_inp_size,mm_size,candidate_cardinality,dropout_p,inputs,shareList,retrieved_entity_matrix)

   -- wrapping up the model
   local model= nn.gModule(inputs,{output_distribution})

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

-- a control rnn from the inputs to a softmax over the outputs
function rnn(t_inp_size,v_inp_size,mm_size,summary_size,h_size,inp_seq_cardinality,candidate_cardinality,h_layer_count,nonlinearity,dropout_p,use_cuda)

   local inputs = {}

   -- this is a table to collect tables of layers that must
   -- share parameters
   local shareList = {}

   -- first, we process the query, mapping its components in parts
   -- onto a first hidden layer to which we will later also map the final
   -- state of recurrent processing of the object tokens

   -- table to share the attribute weights
   local query_attribute_vectors_table = {}

   -- the first attribute in the query
   local curr_input = nn.Identity()()
   table.insert(inputs,curr_input)
   local query_attribute_1_do = nn.Dropout(dropout_p)(curr_input)
   local query_attribute_1 = nn.Linear(t_inp_size, h_size)(query_attribute_1_do)
   table.insert(query_attribute_vectors_table,query_attribute_1)

   -- the second attribute in the query
   local curr_input = nn.Identity()()
   table.insert(inputs,curr_input)
   local query_attribute_2_do = nn.Dropout(dropout_p)(curr_input)
   local query_attribute_2 = nn.Linear(t_inp_size, h_size)(query_attribute_2_do)
   table.insert(query_attribute_vectors_table,query_attribute_2)
   table.insert(shareList,query_attribute_vectors_table)
   
   -- the object name in the query
   local curr_input = nn.Identity()()
   table.insert(inputs,curr_input)
   local query_object_do = nn.Dropout(dropout_p)(curr_input)
   local query_object = nn.Linear(t_inp_size, h_size)(query_object_do)

   -- note that here we pass the query representation through a
   -- nonlinearity (if requested in general) to make it consistent
   -- with the summary vector representation, where the nonlinearity
   -- is typcially useful to properly handle the recurrence
   local linear_query_full_vector = nn.CAddTable()({query_attribute_1,query_attribute_2,query_object})
   local query_full_vector = nil
   if (nonlinearity == 'none') then
      query_full_vector=linear_query_full_vector
   else
      -- if requested, passing through a nonlinear transform: relu,
      -- tanh sigmoid
      if (nonlinearity == 'relu') then
	 query_full_vector = nn.ReLU()(linear_query_full_vector)
      elseif (nonlinearity == 'tanh') then
	 query_full_vector = nn.Tanh()(linear_query_full_vector)
      else -- sigmoid is leftover option: if (nonlinearity == 'sigmoid') then
	 query_full_vector = nn.Sigmoid()(linear_query_full_vector)
      end
   end

   -- now we process the object tokens

   -- a table to store the summary vector as it evolves through time
   local summary_vector_table = {}

   -- tables to store token attribute/object layers for weight sharing
   local token_attribute_vectors_table = {}
   local token_object_vectors_table = {}

   -- a table to store recurrent layers as they evolve, for weight
   -- sharing
   local recurrent_vectors_table = {}

   -- the first object token is a special case, as it has no history

   -- processing the attribute
   local curr_input = nn.Identity()()
   table.insert(inputs,curr_input)
   local first_token_attribute_do = nn.Dropout(dropout_p)(curr_input)
   local first_token_attribute = nn.Linear(t_inp_size,summary_size)(first_token_attribute_do)
   table.insert(token_attribute_vectors_table,first_token_attribute)
   -- then processing the object image
   local curr_input = nn.Identity()()
   table.insert(inputs,curr_input)
   local first_token_object_do = nn.Dropout(dropout_p)(curr_input)
   local first_token_object = nn.Linear(v_inp_size,summary_size)(first_token_object_do)
   table.insert(token_object_vectors_table,first_token_object)
   -- putting together attribute and object 
   local first_object_token_vector = nn.CAddTable()({first_token_attribute,first_token_object})

   -- passing through a nonlinearity if requested, and adding to the table of summary vectors
   if (nonlinearity == 'none') then
      table.insert(summary_vector_table,first_object_token_vector)
   else
      local nonlinear_summary_vector = nil
      -- if requested, passing through a nonlinear transform: relu,
      -- tanh sigmoid
      if (nonlinearity == 'relu') then
	 nonlinear_summary_vector = nn.ReLU()(first_object_token_vector)
      elseif (nonlinearity == 'tanh') then
	 nonlinear_summary_vector = nn.Tanh()(first_object_token_vector)
      else -- sigmoid is leftover option: if (nonlinearity == 'sigmoid') then
	 nonlinear_summary_vector = nn.Sigmoid()(first_object_token_vector)
      end
      table.insert(summary_vector_table,nonlinear_summary_vector)
   end


   -- now we process all the other object tokens in a loop
   for i=2,inp_seq_cardinality do
      -- processing the attribute
      local curr_input = nn.Identity()()
      table.insert(inputs,curr_input)
      local token_attribute_do = nn.Dropout(dropout_p)(curr_input)
      local token_attribute = nn.Linear(t_inp_size,summary_size)(token_attribute_do)
      table.insert(token_attribute_vectors_table,token_attribute)
      -- processing the object image
      local curr_input = nn.Identity()()
      table.insert(inputs,curr_input)
      local token_object_do = nn.Dropout(dropout_p)(curr_input)
      local token_object = nn.Linear(v_inp_size,summary_size)(token_object_do)
      table.insert(token_object_vectors_table,token_object)

      -- also mapping the previous state of the summary vector
      -- NB: NO dropout on the recurrent connection
      local recurrent_vector = nn.Linear(summary_size,summary_size)(summary_vector_table[i-1])
      table.insert(recurrent_vectors_table,recurrent_vector)

      -- putting together attribute, object and recurrence, possibly
      -- passing through a non-linearity, then recording in the
      -- summary_vector_table the current state of the summary vector
      local summary_vector = nn.CAddTable()({token_attribute,token_object,recurrent_vector})
      if (nonlinearity == 'none') then
	 table.insert(summary_vector_table,summary_vector)
      else
	 local nonlinear_summary_vector = nil
	 -- if requested, passing through a nonlinear transform: relu,
	 -- tanh sigmoid
	 if (nonlinearity == 'relu') then
	    nonlinear_summary_vector = nn.ReLU()(summary_vector)
	 elseif (nonlinearity == 'tanh') then
	    nonlinear_summary_vector = nn.Tanh()(summary_vector)
	 else -- sigmoid is leftover option  (nonlinearity == 'sigmoid') then
	    nonlinear_summary_vector = nn.Sigmoid()(summary_vector)
	 end
	 table.insert(summary_vector_table,nonlinear_summary_vector)
      end
   end -- done processing the objects
   -- time to add the tables of connections that must share layers to shareList:
   table.insert(shareList,token_attribute_vectors_table)
   table.insert(shareList,token_object_vectors_table)
   table.insert(shareList,recurrent_vectors_table)

   -- now we use the query vector and the end state of the summary
   -- vector as input to a feed forward neural network that will
   -- predict the entity to probe the candidate images with
   local hidden_layers = {}

   -- we have already computed the first-hidden layer representation
   -- of the query, now we map the summary vector into the hidden
   -- space, combine it with the query representation, and then for
   -- each layer pass through non-linearity if requested and add to
   -- hidden layer table
   local mapped_summary_do = nn.Dropout(dropout_p)(summary_vector_table[inp_seq_cardinality])
   local mapped_summary = nn.Linear(summary_size, h_size)(mapped_summary_do)
   local hidden_layer =  nn.CAddTable()({query_full_vector,mapped_summary})
   -- go through all layers (including the already computed first, just for the nonlinearity)
   for i=1,h_layer_count do
      if i>1 then
	 hidden_layer_do = nn.Dropout(dropout_p)(hidden_layers[i-1])
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

   -- now we map from the last hidden layer to a vector that will represent our
   -- retrieved entity vector (we get a matrix of such vectors because of mini-batches)
   local retrieved_entity_matrix_2D_do = nn.Dropout(dropout_p)(hidden_layers[h_layer_count])
   local retrieved_entity_matrix_2D = nn.Linear(h_size,mm_size)(retrieved_entity_matrix_2D_do)
   -- local retrieved_entity_matrix=nn.View(-1):setNumInputDims(2)(retrieved_entity_matrix_2D)
   local retrieved_entity_matrix=nn.Reshape(1,mm_size,true)(retrieved_entity_matrix_2D) -- reshaping to minibatch x 1 x mm_size for dot product with candidate image vectors in return_entity_image function
   
   -- now we call the return_entity_image function to obtain a softmax
   -- over candidate images
   local output_distribution=return_entity_image(v_inp_size,mm_size,candidate_cardinality,dropout_p,inputs,shareList,retrieved_entity_matrix)

   -- wrapping up the model
   local model= nn.gModule(inputs,{output_distribution})

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


