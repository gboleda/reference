-- a control feed forward network from the concatenation of
-- inputs to a softmax over the output
function ff(t_inp_size,v_inp_size,mm_size,h_size,inp_seq_cardinality,candidate_cardinality,h_layer_count,nonlinearity,dropout_p,use_cuda)
   
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
   local first_hidden_layer_do = nn.Dropout(dropout_p)(all_input)
   local first_hidden_layer = nn.Linear(InDim, h_size)(first_hidden_layer_do)
   
   -- gbt: todo: add check at option reading time that required nonlin is one of none, relu, tanh, sigmoid
   -- go through all layers
   local hidden_layer = first_hidden_layer
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

   -- the function return_entity_image assumes a share
   -- table to be updated with modules sharing their parameters, so we
   -- must initialize shareList
   local shareList = {}
   
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
