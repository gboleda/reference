local function compute_dire_sebastian_model_weight_distribution(raw_similarity_profile_to_entity_matrix, i, temperature, shared_raw_new_entity_mapping)
    -- computing the new-entity cell value
    -- average or max or sum by default of input vector cells...
    
    local patch_constant = 0.5
    local raw_similarity_profile = nn.Padding(2, 1, 2, patch_constant)(raw_similarity_profile_to_entity_matrix)
    raw_similarity_profile = nn.View(-1, 1)(raw_similarity_profile)
    local scale_similarity_profile = nn.Linear(1, 1)(raw_similarity_profile)
    scale_similarity_profile = nn.Sigmoid()(scale_similarity_profile)
    scale_similarity_profile = nn.View(-1, i)(raw_similarity_profile)
    local normalized_similarity_profile = nn.Normalization()(scale_similarity_profile)
    
    return normalized_similarity_profile
end

function entity_prediction_image_att_shared_neprob_sebastian(t_inp_size,v_inp_size,mm_size,inp_seq_cardinality, temperature, dropout_p,use_cuda)
    return build_customize_model(t_inp_size,v_inp_size,mm_size,inp_seq_cardinality, compute_dire_sebastian_model_weight_distribution, temperature, dropout_p,use_cuda)
end

function entity_prediction_image_att_shared_neprob_sebastian_metric_learning(t_inp_size,v_inp_size,mm_size,inp_seq_cardinality, temperature, dropout_p,use_cuda)
    return build_customize_model_metric_learning(t_inp_size,v_inp_size,mm_size,inp_seq_cardinality, compute_dire_sebastian_model_weight_distribution, temperature, dropout_p,use_cuda)
end

function entity_prediction_image_att_shared_neprob_sebastian_2matrices(t_inp_size,v_inp_size,mm_size,inp_seq_cardinality, temperature, dropout_p,use_cuda)
    return build_customize_model_with_2matrices(t_inp_size,v_inp_size,mm_size,inp_seq_cardinality, compute_dire_sebastian_model_weight_distribution, temperature, dropout_p,use_cuda)
end

function entity_prediction_image_att_shared_neprob_sebastian_2matrices_cosine(t_inp_size,v_inp_size,mm_size,inp_seq_cardinality, temperature, dropout_p,use_cuda)
    return build_customize_model_with_2matrices_cosine(t_inp_size,v_inp_size,mm_size,inp_seq_cardinality, compute_dire_sebastian_model_weight_distribution, temperature, dropout_p,use_cuda)
end