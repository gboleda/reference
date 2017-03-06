function build_entity_matrix(t_inp_size, v_inp_size, mm_size, inp_seq_cardinality, dropout_p, in_table, obj_mappings_table, att_mappings_table) --, share_table)
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

      -- computing the old-entity mass value as max of input vector
      -- cells followed by a transformation that ensures that it is
      -- between 0 and 1
      local raw_old_entity_mass=nn.Max(1,2)(raw_similarity_profile_to_entity_matrix):annotate{name='raw_old_entity_mass_' .. i}
      local normalized_old_entity_mass = nn.ReLU()(nn.Tanh()(raw_old_entity_mass))
      -- now, we concatenate the similarity profile with a new cell,
      -- given by 1 - normalized_old_entity_mass
      local normalized_new_entity_mass = nn.AddConstant(1,false)(nn.MulConstant(-1,false)(normalized_old_entity_mass))
      local normalized_similarity_profile = nn.SoftMax()(nn.View(-1):setNumInputDims(2)(raw_similarity_profile_to_entity_matrix))
      -- NB: the output of the following very messy line of code is a
      -- matrix with the profile of each item in a minibatch as
      -- a ROW vector
      normalized_similarity_profile = nn.MM(false, false){nn.View(-1,i - 1, 1)(normalized_similarity_profile),nn.View(-1,1, 1)(normalized_old_entity_mass)}
      normalized_similarity_profile = (nn.JoinTable(2,2)({nn.View(-1,i - 1)(normalized_similarity_profile),normalized_new_entity_mass})):annotate{name='normalized_similarity_profile_' .. i}
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

   -- and finally returning the last state of the entity matrix
   return entity_matrix_table[inp_seq_cardinality]
end


--- PASTED FROM HERE
-- our main model sharing all image embeddings and all attribute embeddings
-- WITH GEMMA'S AND MARCO'S CHANGE TO GERMÁN'S CHANGE
function entity_prediction_image_att_shared_neprob_onion(t_inp_size,v_inp_size,mm_size,inp_seq_cardinality,candidate_cardinality,temperature,dropout_p,use_cuda)

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

   -- we initalize the table to store the object mappings here
   local token_object_mappings = {}

   -- now we call a function to process the object tokens and return an entity matrix
   local stable_entity_matrix = build_entity_matrix(t_inp_size,v_inp_size,mm_size,inp_seq_cardinality,dropout_p,inputs,token_object_mappings,attribute_mappings)--,shareList)

   -- we are done processing input attributes, so we add their parameter list to the list of parameters to be shared
   table.insert(shareList,attribute_mappings)

   -- at this point, we take the dot product of each row (entity)
   -- vector in the entity matrix with the linguistic query vector, to
   -- obtain an entity-to-query similarity profile, that we softmax
   -- normalize (note Views needed to get right shapes, and rescaling
   -- by temperature)
   local raw_query_entity_similarity_profile = nn.View(-1):setNumInputDims(2)(nn.MM(false,false)({stable_entity_matrix,query}))
   local rescaled_query_entity_similarity_profile = nn.MulConstant(temperature)(raw_query_entity_similarity_profile)
   local query_entity_similarity_profile = nn.View(1,-1):setNumInputDims(1)(nn.SoftMax()(rescaled_query_entity_similarity_profile)):annotate{name='query_entity_similarity_profile'}

   -- we now do "soft retrieval" of the entity that matches the query:
   -- we obtain a vector that is a weighted sum of all the entity
   -- vectors in the entity library (weights= similarity profile, such
   -- that we will return the entity that is most similar to the
   -- query) (we get a matrix of such vectors because of mini-batches)
   local retrieved_entity_matrix = nn.MM(false,false)({query_entity_similarity_profile,stable_entity_matrix})
   
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
