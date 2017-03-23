local function add_new_input_and_create_mapping(inputs, input_size, mapping_size, dropout_p, share_mappings)
   local curr_input = nn.Identity()()
   table.insert(inputs,curr_input)
   local mapping_in_do = nn.Dropout(dropout_p)(curr_input)
   local mapping = nn.LinearNB(input_size, mapping_size)(mapping_in_do):annotate{name='query_att1'}
   table.insert(share_mappings,mapping)
   return mapping
end

local function add_new_input_and_create_double_mapping(inputs, input_size, mapping_size, dropout_p, share_mappings_select, share_mappings_compare)
   local curr_input = nn.Identity()()
   table.insert(inputs,curr_input)
   local mapping_in_do = nn.Dropout(dropout_p)(curr_input)
   local mapping_select = nn.LinearNB(input_size, mapping_size)(mapping_in_do):annotate{name='select'}
   local mapping_compare = nn.LinearNB(input_size, mapping_size)(mapping_in_do):annotate{name='compare'}
   table.insert(share_mappings_select,mapping_select)
   table.insert(share_mappings_compare,mapping_compare)
   return mapping_select, mapping_compare
end

local function add_and_compute_token_vector(inputs, t_input_size, v_input_size, mapping_size, dropout_p, attribute_mappings_select, object_mappings_select, 
            attribute_mappings_compare, object_mappings_compare)
    -- first processing the attribute
   local token_attribute_select, token_attribute_compare = add_new_input_and_create_double_mapping(inputs,t_input_size,mapping_size,dropout_p,attribute_mappings_select, attribute_mappings_compare)
   -- processing the object image
   local token_object_select, token_object_compare = add_new_input_and_create_double_mapping(inputs,v_input_size,mapping_size,dropout_p,object_mappings_select, object_mappings_compare)
   -- putting together attribute and object 
   local token_vector_select = nn.CAddTable()({token_attribute_select,token_object_select})
   local token_vector_compare = nn.CAddTable()({token_attribute_compare,token_object_compare})
   return token_vector_select, token_vector_compare
end

local function build_entity_libary_2matrices(t_inp_size, v_inp_size, mm_size, inp_seq_cardinality, dropout_p, inputs, 
                             attribute_mappings_select, attribute_mappings_compare, token_object_mappings_select, 
                             token_object_mappings_compare, raw_new_entity_mass_mappings,
                             weight_distribution_function)
   -- now we process the object tokens
   -- a table to store the entity matrix as it evolves through time
   local entity_matrix_table_select = {}
   local entity_matrix_table_compare = {}
   -- tables where to store the connections that must share weights:
   
   

   -- the first object token is a special case, as it will always be
   -- directly mapped to the first row of the entity matrix
   
   local first_object_token_vector_select, first_object_token_vector_compare = add_and_compute_token_vector(inputs,t_inp_size,
            v_inp_size,mm_size,dropout_p,attribute_mappings_select,token_object_mappings_select, attribute_mappings_compare,token_object_mappings_compare)
   -- turning the vector into a 1xmm_size matrix, adding the latter as
   -- first state of the entity matrix table
   table.insert(entity_matrix_table_select,nn.View(1,-1):setNumInputDims(1)(first_object_token_vector_select))
   table.insert(entity_matrix_table_compare,nn.View(1,-1):setNumInputDims(1)(first_object_token_vector_compare))

   -- now we process all the other object tokens in a loop
   for i=2,inp_seq_cardinality do
      
      local object_token_vector_flat_select, object_token_vector_flat_compare = add_and_compute_token_vector(inputs,t_inp_size,
            v_inp_size,mm_size,dropout_p,attribute_mappings_select,token_object_mappings_select, attribute_mappings_compare, token_object_mappings_compare)
      -- reshaping
      local object_token_vector_select = nn.View(-1,1):setNumInputDims(1)(object_token_vector_flat_select)
      local object_token_vector_compare = nn.View(-1,1):setNumInputDims(1)(object_token_vector_flat_compare)

      -- measuring the similarity of the current vector to the ones in
      -- the previous state of the entity matrix
      local raw_similarity_profile_to_entity_matrix_select = nn.MM(false,false)({entity_matrix_table_select[i-1],object_token_vector_select})

      local normalized_similarity_profile = weight_distribution_function(raw_similarity_profile_to_entity_matrix_select, i, raw_new_entity_mass_mappings)
      
      if (i ~= inp_seq_cardinality) then
         local weighted_object_token_vector_matrix_select = nn.MM(false,true){nn.View(-1,1):setNumInputDims(1)(normalized_similarity_profile),object_token_vector_select}
         entity_matrix_table_select[i]= nn.CAddTable(){
            nn.Padding(1,1,2)(entity_matrix_table_select[i-1]),weighted_object_token_vector_matrix_select}:annotate{'entity_matrix_table' .. i}
      end
      local weighted_object_token_vector_matrix_compare = nn.MM(false,true){nn.View(-1,1):setNumInputDims(1)(normalized_similarity_profile),object_token_vector_compare}
      entity_matrix_table_compare[i]= nn.CAddTable(){
         nn.Padding(1,1,2)(entity_matrix_table_compare[i-1]),weighted_object_token_vector_matrix_compare}:annotate{'entity_matrix_table' .. i}
   end
   return entity_matrix_table_select, entity_matrix_table_compare

end

function build_customize_model_with_2matrices(t_inp_size,v_inp_size,mm_size,inp_seq_cardinality, weight_distribution_function, temperature, dropout_p,use_cuda)

   local inputs = {}

   -- a table to store tables of connections that must share parameters
   local shareList = {}
   
   local attribute_mappings_select= {}
   local attribute_mappings_compare = {}
   ---- token object mappings
   local token_object_mappings_select = {}
   local token_object_mappings_compare = {}
   --- mappings to raw new entity mass
   local raw_new_entity_mass_mappings = {}

   -- adding to shareList here the attribute and entity mass mappings
   table.insert(shareList,attribute_mappings_select)
   table.insert(shareList,attribute_mappings_compare)
   table.insert(shareList,token_object_mappings_select)
   table.insert(shareList,token_object_mappings_compare)
   table.insert(shareList,raw_new_entity_mass_mappings)
   
   
   -- first, we process the query, mapping it onto multimodal space
   -- the query attributes in the query
   local query_attribute_1 = add_new_input_and_create_mapping(inputs,t_inp_size,mm_size,dropout_p,attribute_mappings_compare)
   local query_attribute_2 = add_new_input_and_create_mapping(inputs,t_inp_size,mm_size,dropout_p,attribute_mappings_compare)

   -- putting together the multi-modal query vector by summing the
   -- output of the previous linear transformations, and ensuring it
   -- will be a column vector
   local query = nn.View(-1,1):setNumInputDims(1)(nn.CAddTable()({query_attribute_1,query_attribute_2}):annotate{name='query'})

   local entity_matrix_table_select, entity_matrix_table_compare = build_entity_libary_2matrices(t_inp_size, v_inp_size, mm_size, inp_seq_cardinality, dropout_p, inputs, 
                             attribute_mappings_select, attribute_mappings_compare, token_object_mappings_select, 
                             token_object_mappings_compare, raw_new_entity_mass_mappings,
                             weight_distribution_function)

   local raw_query_entity_similarity_profile = nn.View(-1):setNumInputDims(2)(nn.MM(false,false)({entity_matrix_table_compare[inp_seq_cardinality],query}))
   local rescaled_query_entity_similarity_profile = nn.MulConstant(temperature)(raw_query_entity_similarity_profile)
   local output_distribution = nn.LogSoftMax()(rescaled_query_entity_similarity_profile):annotate{name='query_entity_similarity_profile'}

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