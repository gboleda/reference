-- auxiliary functions

function model_accuracy(model_name,data_size,predictions,gold,nconf)
   -- this function returns only accuracy (for use for validation);
   -- for accuracy AND model predictions, see function
   -- model_accuracy_and_predictions below
   local hit_count=0
   local model_guesses_indices=torch.zeros(data_size)
   local model_guesses_preds=torch.zeros(data_size)
   if model_name=='max_margin_bl' then
      local end_at=0
      for seqn=1,data_size do -- iterating over sequences
	 -- print('---')
	 -- print(seqn)
	 local conf=nconf[seqn] -- how many confounders there are in the sequence
      	 local start_at=end_at+1 -- next time we start after we left off
      	 end_at=start_at+(conf-1)
	 -- first element in predictions contains dot products of query and target. They are the same for all the tuples:
	 local qt=predictions[1][start_at]
      	 local qc_sequence=predictions[2][{{start_at,end_at}}] -- second element in predictions contains dot products of query and the confounder
      	 max_confounder=torch.max(qc_sequence)
      	 if qt > max_confounder then
	    hit_count=hit_count+1
	 end
      	 seqn=seqn+1
      end
   else
      -- to compute accuracy, we first retrieve list of indices of image
      -- vectors that were preferred by the model
      model_max_log_probs,model_guesses_indices=torch.max(predictions,2)
      model_guesses_preds=torch.exp(model_max_log_probs)
      -- we then count how often this guesses are the same as the gold
      -- (and thus the difference is 0) (note conversions to long because
      -- model_guesses_indices is long tensor)
      hit_count=torch.sum(torch.eq(gold:long(),model_guesses_indices))
   end
   -- normalizing accuracy by test/valid set size
   local accuracy=hit_count/data_size
   return accuracy
end

function model_accuracy_and_predictions(model_name,data_size,max_n_imgs_per_seq,predictions,gold_indices,nconf)
   local hit_count=0
   local model_guesses_indices=torch.zeros(data_size)
   local model_guesses_preds=torch.zeros(data_size)
   local all_items_preds=nil
   if model_name=='max_margin_bl' then
      -- data_size x max_n_imgs_per_seq tensor to hold predictions for
      -- all items in sequences (we set values for "non-images" to a
      -- very low number so they don't interfere with accuracy
      -- calculation)
      all_items_preds=torch.Tensor(data_size,max_n_imgs_per_seq)-math.huge
      local end_at=0
      for seqn=1,data_size do -- iterating over sequences
	 local gold=gold_indices[seqn] -- which is the target image of the sequence
	 -- if opt.deviance_mode==1 then
	 if gold < 1 then
	    gold=1 -- deviant sequences are assigned an arbitrary target, just to keep things from breaking down
	 end
	 -- end
	 local conf=nconf[seqn] -- how many confounders there are in the sequence
      	 local start_at=end_at+1 -- next time we start after we left off
      	 end_at=start_at+(conf-1)
	 -- print('---')
	 -- print('seqn ' .. tostring(seqn))
	 -- print('conf ' .. tostring(conf))
	 -- print('start at: ' .. tostring(start_at) .. ', end at: ' .. tostring(end_at))
	 -- first element in predictions contains dot products of query and target. They are the same for all the tuples:
	 local qt=predictions[1][start_at]
      	 local qc_sequence=predictions[2][{{start_at,end_at}}] -- second element in predictions contains dot products of query and the confounder
	 -- print('target: ' .. tostring(gold) .. ' (' .. tostring(qt) .. ')')
	 -- print('confounder: ' .. tostring(qc_sequence))
	 local position_in_confounder_sequence=1
	 for elem=1,conf+1 do
	    -- print('position_in_confounder_sequence: ' .. tostring(position_in_confounder_sequence))
	    if elem==gold then
	       all_items_preds[seqn][elem]=qt
	       -- print('   target ' .. tostring(qt))
	    else
	       all_items_preds[seqn][elem]=qc_sequence[position_in_confounder_sequence]
	       -- print('   confounder ' .. tostring(qc_sequence[position_in_confounder_sequence]))
	       position_in_confounder_sequence=position_in_confounder_sequence+1
	    end
	 end
      	 seqn=seqn+1
      end
      -- print(all_items_preds)
   else -- initialization for all other models
      all_items_preds=torch.exp(predictions)
   end

   -- to compute accuracy, we first retrieve list of indices of image
   -- vectors that were preferred by the model
   model_guesses_preds,model_guesses_indices=torch.max(all_items_preds,2)
   -- we then count how often this guesses are the same as the gold
   -- (and thus the difference is 0) (note conversions to long because
   -- model_guesses_indices is long tensor)
   hit_count=torch.sum(torch.eq(gold_indices:long(),model_guesses_indices))
   if not(opt.model=='max_margin_bl') then
      model_guesses_preds=torch.exp(model_guesses_preds)
   end

   -- normalizing accuracy by test/valid set size
   local accuracy=hit_count/data_size
   return accuracy,model_guesses_indices,model_guesses_preds,all_items_preds
end

-- gbt: deprecated -- erase
function model_predictions_for_deviants(data_size,max_n_imgs_per_seq,predictions,nconf) -- aquí
   local hit_count=0
   local max_dotprods=torch.zeros(data_size)
   local max_diffs=torch.zeros(data_size)
   local all_items_preds=nil
   -- data_size x max_n_imgs_per_seq tensor to hold predictions for
   -- all items in sequences (we set values for "non-images" to a
   -- very low number so they don't interfere with accuracy
   -- calculation)
   all_items_preds=torch.Tensor(data_size,max_n_imgs_per_seq)-math.huge
   local end_at=0
   for seqn=1,data_size do -- iterating over sequences
      local gold=1 -- which is the target image of the sequence; REUSING CODE HERE, BUT THIS WILL ALWAYS BE 1 (see data.lua)
      local conf=nconf[seqn] -- how many confounders there are in the sequence
      local start_at=end_at+1 -- next time we start after we left off
      end_at=start_at+(conf-1)
      -- print('---')
      -- print('seqn ' .. tostring(seqn))
      -- print('conf ' .. tostring(conf))
      -- print('start at: ' .. tostring(start_at) .. ', end at: ' .. tostring(end_at))
      -- first element in predictions contains dot products of query and target. They are the same for all the tuples:
      local qt=predictions[1][start_at]
      local qc_sequence=predictions[2][{{start_at,end_at}}] -- second element in predictions contains dot products of query and the confounder
      -- print('target: ' .. tostring(gold) .. ' (' .. tostring(qt) .. ')')
      -- print('confounder: ' .. tostring(qc_sequence))
      local position_in_confounder_sequence=1
      for elem=1,conf+1 do
	 -- print('position_in_confounder_sequence: ' .. tostring(position_in_confounder_sequence))
	 if elem==gold then
	    all_items_preds[seqn][elem]=qt
	    -- print('   target ' .. tostring(qt))
	 else
	    all_items_preds[seqn][elem]=qc_sequence[position_in_confounder_sequence]
	    -- print('   confounder ' .. tostring(qc_sequence[position_in_confounder_sequence]))
	    position_in_confounder_sequence=position_in_confounder_sequence+1
	 end
      end
      seqn=seqn+1
   end
   -- print(all_items_preds)

   -- we get the max dot products
   max_dotprods,_=torch.max(all_items_preds,2)

   -- normalizing accuracy by test/valid set size
   local accuracy=hit_count/data_size
   return accuracy,all_items_preds
end

function print_model_predictions_to_file(ofile,model_name,guessed_indices,guessed_preds,indiv_model_predictions,extd_dot_vector)
   local f = io.open(ofile,"w")
   -- if model_name=='max_margin_bl' then
   --    print(guessed_indices)
   --    print(guessed_preds)
   --    print(indiv_model_predictions)
   -- else
   for i=1,guessed_preds:size(1) do
      f:write(guessed_indices[i][1]," ",guessed_preds[i][1])
      for j=1,indiv_model_predictions:size(2) do
	 -- here to end if added by gbt instead of
	 -- f:write(" ",indiv_model_predictions[i][j])
	 towrite=indiv_model_predictions[i][j]
	 if towrite > -math.huge then
	    f:write(" ",towrite)
	 end
      end
      if (opt.debug==1) then -- in debug mode, we also return dot vectors and deviance cell (when available)
	 if extd_dot_vector then -- added by gbt
	    for k=1,extd_dot_vector:size(2) do
	       f:write(" ",extd_dot_vector[i][k])
	    end
	 end
      end
      f:write("\n")
   end
   f:flush()
   f.close()
end

-- gbt: deprecated -- erase
function print_model_predictions_to_file_for_deviants(ofile,gold_indices,indiv_model_predictions)
   local f = io.open(ofile,"w")
   -- print(indiv_model_predictions)
   for i=1,indiv_model_predictions:size(1) do
      f:write(gold_indices[i])
      for j=1,indiv_model_predictions:size(2) do
   	 towrite=indiv_model_predictions[i][j]
   	 if towrite > -math.huge then
   	    f:write(" ",towrite)
   	 end
      end
      f:write("\n")
   end
   f:flush()
   f.close()
end
