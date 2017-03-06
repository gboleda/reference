function compute_dire_model_weight_distribution(raw_similarity_profile_to_entity_matrix, i, shared_raw_new_entity_mapping)
    -- computing the new-entity cell value
    -- average or max or sum by default of input vector cells...
    local raw_cumulative_similarity=nn.Max(1,2)(raw_similarity_profile_to_entity_matrix)
    raw_cumulative_similarity:annotate{name='raw_cumulative_similarity_' .. i}
    local raw_new_entity_mass = nn.Linear(1,1)(raw_cumulative_similarity):annotate{name='raw_new_entity_mass_' .. i}
    table.insert(shared_raw_new_entity_mapping,raw_new_entity_mass)

    -- passing through nonlinearity if requested
    local transformed_new_entity_mass = nn.Sigmoid()(raw_new_entity_mass)
    
    -- now, we concatenate the similarity profile with this new
    -- cell, and normalize
    -- NB: the output of the following very messy line of code is a
    -- matrix with the profile of each item in a minibatch as
    -- a ROW vector
    local minus_transform_new_entity_mass = nn.AddConstant(1,false)(nn.MulConstant(-1,false)(transformed_new_entity_mass))
    local normalized_similarity_profile = nn.SoftMax()(nn.View(-1):setNumInputDims(2)(raw_similarity_profile_to_entity_matrix))
    normalized_similarity_profile = nn.MM(false, false){nn.View(-1,i - 1, 1)(normalized_similarity_profile),nn.View(-1,1, 1)(minus_transform_new_entity_mass)}
    normalized_similarity_profile = (nn.JoinTable(2,2)({nn.View(-1,i - 1)(normalized_similarity_profile),transformed_new_entity_mass})):annotate{name='normalized_similarity_profile_' .. i}
    
    return normalized_similarity_profile
end

function entity_prediction_image_att_shared_neprob(t_inp_size,v_inp_size,mm_size,inp_seq_cardinality,candidate_cardinality, temperature, dropout_p,use_cuda)
    return build_customize_model(t_inp_size,v_inp_size,mm_size,inp_seq_cardinality,candidate_cardinality, compute_dire_model_weight_distribution, temperature, dropout_p,use_cuda)
end