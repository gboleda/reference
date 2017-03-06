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

function return_entity_image_shared(v_inp_size,mm_size,candidate_cardinality,dropout_p,in_table,imageMappings_table,retrieved_entity_matrix)
   local image_candidate_vectors={}
   -- image candidates vectors
   for i=1,candidate_cardinality do
      local curr_input = nn.Identity()()
      table.insert(in_table,curr_input)
      local image_candidate_vector_do = nn.Dropout(dropout_p)(curr_input)
      local image_candidate_vector = nn.LinearNB(v_inp_size,mm_size)(image_candidate_vector_do)
      table.insert(image_candidate_vectors, image_candidate_vector)
      table.insert(imageMappings_table, image_candidate_vector)
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



