require '../lent_and_returned/model-for-counting'
-- adaptation of the models to counting
-- to do:
-- for the entity counting task: normalize every entity vector and sum. # gbt: Note that we are actually exposing the entity library in this task. # gbt: How are we going to adapt the other models?
-- but... if we do this and we have identical images for the entities... isn't the entity counting task trivial for our model?
-- for the entity-of-a-given-category counting task: normalize every entity vector, do dot product to query of every entity vector, and sum. # for the moment we do it without normalizing to see if the model learns to scale. If not, revert to normalization.
function entity_prediction_image_att_shared_neprob_counting(t_inp_size,v_inp_size,mm_size,inp_seq_cardinality,candidate_cardinality,temperature,dropout_p,use_cuda)

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

   -- we are done processing input attributes and objects, so we add their parameter list to the list of parameters to be shared
   table.insert(shareList,attribute_mappings)
   table.insert(shareList,token_object_mappings)
   
   -- at this point, we take the dot product of each row (entity)
   -- vector in the entity matrix with the linguistic query vector, to
   -- obtain an entity-to-query similarity profile, which we pass through a set of transformations so that non-positive dot products will be
   -- mapped to 0, and positive ones will be in the 0-1 range (making this an approximation to an integer-based "count"
   local raw_matrix_query_entity_similarity_profile_1 = nn.MM(false,false)({stable_entity_matrix,query})
   local raw_matrix_query_entity_similarity_profile_2 = nn.Tanh()(raw_matrix_query_entity_similarity_profile_1)
   local matrix_query_entity_similarity_profile = nn.ReLU()(raw_matrix_query_entity_similarity_profile_2):annotate{name='query_entity_similarity_profile'}


   -- and we sum them to obtain the predicted number of entities
   local output_number_of_entities = nn.Sum(1,2)(matrix_query_entity_similarity_profile)

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