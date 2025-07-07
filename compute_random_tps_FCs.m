function FC_arr = compute_random_tps_FCs(cond, num_subjs, parc, task_len, str)
    
    if strcmp(str,'rand') % Sample time points randomly
        sample_pt = randperm(1190);
        sample_pts = sample_pt(1:task_len);
        FC_arr = zeros(parc,parc, num_subjs);
    elseif strcmp(str,'consec') % Sample time points consecutively 
        sample_pt = randperm(1190-task_len,1);
        sample_pts = sample_pt:(sample_pt+task_len-1);
        FC_arr = zeros(parc,parc, num_subjs);
    else 
        error('Erorr! Invalid sampling choice of time points.')
    end

    % Extract the points from RS TS
    for i = 1:num_subjs
        FC_arr(:,:,i) = corr(cond.ts{i}(sample_pts,:));
    end
end
