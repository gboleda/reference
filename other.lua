-- auxiliary functions

function compute_accuracy(model_name,data_size,predictions,gold,nconf)
   local hit_count=0
   if model_name=='max_margin_bl' then
      local end_at=0
      -- to compute accuracy, we first get the answers for each sequence
      for seqn=1,data_size do -- iterating over sequences
      	 -- print('---')
      	 -- print(seqn)
	 conf=nconf[seqn] -- how many confounders there are in the sequence
	 -- print(conf)
      	 local start_at=end_at+1 -- next time we start after we left off
      	 end_at=start_at+(conf-1)
      	 -- print('start at: ' .. tostring(start_at) .. ', end at: ' .. tostring(end_at))
	 local qt=predictions[1][start_at] -- first element in predictionss contains dot products of query and target. They are the same for all the tuples.
      	 local qc_sequence=predictions[2][{{start_at,end_at}}]
      	 max_confounder=torch.max(qc_sequence)
      	 -- print('target:' .. tostring(qt))
      	 -- print('confounder:' .. tostring(qc_sequence))
      	 -- print('target - max confounder: ' .. tostring(qt) .. ';' .. tostring(max_confounder))
      	 if qt > max_confounder then
	    hit_count=hit_count+1
	    -- print('hit! ' .. tostring(hit_count))
	 end
      	 seqn=seqn+1
      end
   else
      -- to compute accuracy, we first retrieve list of indices of image
      -- vectors that were preferred by the model
      local model_max_log_probs,model_guesses=torch.max(predictions,2)
      local model_max_probs=torch.exp(model_max_log_probs)
      -- we then count how often this guesses are the same as the gold
      -- (and thus the difference is 0) (note conversions to long because
      -- model_guesses is long tensor)
      hit_count = torch.sum(torch.eq(gold:long(),model_guesses))
   end
   -- normalizing accuracy by test/valid set size
   local accuracy=hit_count/data_size
   return accuracy
end
