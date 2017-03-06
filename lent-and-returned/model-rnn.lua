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
