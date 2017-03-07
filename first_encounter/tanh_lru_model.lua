function compute_lru_model_weight_distribution(raw_similarity_profile_to_entity_matrix, i, new_entity_mappings_table)
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
      
      return normalized_similarity_profile
end

function entity_prediction_image_att_shared_neprob_onion(t_inp_size,v_inp_size,mm_size,inp_seq_cardinality, temperature, dropout_p,use_cuda)
    return build_customize_model(t_inp_size,v_inp_size,mm_size,inp_seq_cardinality, compute_lru_model_weight_distribution, temperature, dropout_p,use_cuda)
end