function return_entity_image(v_inp_size,mm_size,candidate_cardinality,dropout_p,in_table,retrieved_entity_matrix)
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
   local all_candidate_values=nn.JoinTable(1,1)(image_candidate_vectors) -- second argument to JoinTable tells it that 
                                                                   -- the expected inputs in the table are one-dimensional (the
                                                                   --  candidate vectors), necessary not to confuse 
                                                                   -- it when batches are passed
   local candidate_matrix=nn.View(#image_candidate_vectors,-1):setNumInputDims(1)(all_candidate_values) -- again, note 
                                                                                                  -- setNumInputDims for
   -- taking the dot product of each candidate vector
   -- with the retrieved_entity vector
   local dot_vector_split=nn.MM(false,true)({retrieved_entity_matrix,candidate_matrix})
   local dot_vector=nn.View(-1):setNumInputDims(2)(dot_vector_split) -- reshaping into batch-by-nref matrix for minibatch
                                                                           -- processing
   return nn.LogSoftMax()(dot_vector)
end

-- our main model
function entity_prediction(t_inp_size,v_inp_size,mm_size,inp_seq_cardinality,dropout_p)

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
function ff(t_inp_size,v_inp_size,mm_size,h_size,inp_seq_cardinality,candidate_cardinality,h_layer_count,nonlinearity,dropout_p)
   
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

   -- now we map from the last hidden layer to a vector that will represent our
   -- retrieved entity vector (we get a matrix of such vectors because of mini-batches)
   local retrieved_entity_matrix = nn.Linear(h_size,mm_size)(hidden_layers[h_layer_count])
   
   -- now we call the return_entity_image function to obtain a softmax
   -- over candidate images
   local output_distribution=return_entity_image(v_inp_size,mm_size,candidate_cardinality,dropout_p,inputs,retrieved_entity_matrix)

   -- wrapping up the model
   return nn.gModule(inputs,{output_distribution})

end

function OLDff(t_inp_size,v_inp_size,h_size,inp_seq_cardinality,h_layer_count,nonlinearity)

   local inputs = {}
   local hidden_layers = {}
   -- we concatenate all inputs (query and candidates), mapping them all to the hidden layer

   -- the first attribute in the query
   local curr_input = nn.Identity()()
   table.insert(inputs,curr_input)
   local query_attribute_1 = nn.LinearNB(t_inp_size, h_size)(curr_input)

   -- the second attribute in the query
   local curr_input = nn.Identity()()
   table.insert(inputs,curr_input)
   local query_attribute_2 = nn.LinearNB(t_inp_size, h_size)(curr_input)
   
   -- the object name in the query
   local curr_input = nn.Identity()()
   table.insert(inputs,curr_input)
   local query_object = nn.LinearNB(t_inp_size, h_size)(curr_input)

   -- merging the mapped input vectors into a single hidden layer
   -- gbt: why are you summing them?
   local first_hidden_layer = nn.CAddTable()({query_attribute_1,query_attribute_2,query_object})

   -- now we process the candidate (object tokens), mapping them and
   -- adding them to the hidden layer
   for i=1,inp_seq_cardinality do
      -- first an attribute
      local curr_input = nn.Identity()()
      table.insert(inputs,curr_input)
      local token_attribute = nn.LinearNB(t_inp_size,h_size)(curr_input)
      -- then an object image
      local curr_input = nn.Identity()()
      table.insert(inputs,curr_input)
      local token_object = nn.LinearNB(v_inp_size,h_size)(curr_input)
      -- adding to the hidden layer
      first_hidden_layer = nn.CAddTable()({first_hidden_layer,token_attribute,token_object})
   end

   if (nonlinearity == 'none') then
      table.insert(hidden_layers,first_hidden_layer)
   else
      local nonlinear_first_hidden_layer = nil
      -- if requested, passing the hidden layer through a nonlinear
      -- transform: relu, tanh sigmoid
      if (nonlinearity == 'relu') then
	 nonlinear_first_hidden_layer = nn.ReLU()(first_hidden_layer)
      elseif (nonlinearity == 'tanh') then
	 nonlinear_first_hidden_layer = nn.Tanh()(first_hidden_layer)
      else -- sigmoid is leftover option  (nonlinearity == 'sigmoid') then
	 nonlinear_first_hidden_layer = nn.Sigmoid()(first_hidden_layer)
      end
      table.insert(hidden_layers,nonlinear_first_hidden_layer)
   end

   -- go through further layers if so instructed
   for i=2,h_layer_count do
      local hidden_layer = nn.Linear(h_size,h_size)(hidden_layers[i-1])
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

