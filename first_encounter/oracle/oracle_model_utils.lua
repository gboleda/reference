function build_entity_matrix(t_inp_size, v_inp_size, mm_size, inp_seq_cardinality, dropout_p, in_table, obj_mappings_table, 
                             att_mappings_table, temperature)
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
   
   local entity_weight_mats = {}
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

      local normalized_similarity_profile = nn.Identity()()
      table.insert(entity_weight_mats, normalized_similarity_profile)

      local weighted_object_token_vector_matrix = nn.MM(false,true)({nn.View(-1,1):setNumInputDims(1)(normalized_similarity_profile),object_token_vector})

      -- at this point we update the entity matrix by adding the
      -- weighted versions of the current object token vector to each
      -- row of it (we pad the bottom of the entity matrix with a zero
      -- row, so that we can add it to the version of the current
      -- object token vector that was weighted by the new mass cell
      entity_matrix_table[i]= nn.CAddTable()({
         nn.Padding(1,1,2)(entity_matrix_table[i-1]),weighted_object_token_vector_matrix}):annotate{'entity_matrix_table' .. i}
   end
   -- end of processing input objects
   
   -- put the oracle matrix at the end of input
   for i=1,inp_seq_cardinality - 1 do
      table.insert(in_table,entity_weight_mats[i])
   end
   -- and finally returning the last state of the entity matrix
   return entity_matrix_table[inp_seq_cardinality]
end

function entity_prediction_image_oracle(t_inp_size,v_inp_size,mm_size,inp_seq_cardinality, temperature, dropout_p,use_cuda)

   local inputs = {}

   -- a table to store tables of connections that must share parameters
   local shareList = {}

   -- table to collect all mappings, to be shared
   local attribute_mappings = {}
   local token_object_mappings = {}

-- adding token_object_mappings to shareList only now, after we also added to it the candidate image mappings
   table.insert(shareList,token_object_mappings)
   table.insert(shareList,attribute_mappings)
   
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
   
   -- putting together the multimodal query vector by summing the
   -- output of the previous linear transformations, and ensuring it
   -- will be a column vector
   local query = nn.View(-1,1):setNumInputDims(1)(nn.CAddTable()({query_attribute_1,query_attribute_2}):annotate{name='query'})



   -- now we call a function to process the object tokens and return an entity matrix
   local stable_entity_matrix = build_entity_matrix(t_inp_size,v_inp_size,mm_size,inp_seq_cardinality,dropout_p,inputs,
                        token_object_mappings,attribute_mappings, temperature)

   -- at this point, we take the dot product of each row (entity)
   -- vector in the entity matrix with the linguistic query vector, to
   -- obtain an entity-to-query similarity profile, that we softmax
   -- normalize (note Views needed to get right shapes, and rescaling
   -- by temperature)
   local raw_query_entity_similarity_profile = nn.View(-1):setNumInputDims(2)(nn.MM(false,false)({stable_entity_matrix,query}))
   local rescaled_query_entity_similarity_profile = nn.MulConstant(1)(raw_query_entity_similarity_profile)
   local output_distribution = nn.LogSoftMax()(rescaled_query_entity_similarity_profile):annotate{name='query_entity_similarity_profile'}


   -- wrapping up the model
   local model = nn.gModule(inputs,{output_distribution})
   
   -- following code is adapted from MeMNN 
   if (use_cuda ~= 0) then
      model:cuda()
   end
   -- IMPORTANT! do weight sharing after model is in cuda
   for i = 1,#shareList do
      if next(shareList[i]) ~= nil then
         local m1 = shareList[i][1].data.module
         for j = 2,#shareList[i] do
            local m2 = shareList[i][j].data.module
            m2:share(m1,'weight','bias','gradWeight','gradBias')
         end
      end
   end
   return model

end





