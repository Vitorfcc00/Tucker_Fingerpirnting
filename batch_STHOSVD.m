% Batch example ST-HOSVD (Carvalho et al., AppliedSciences 2025)
% Vitor Carvalho, Purdue University
% PLEASE CITE US!
% If you are using this code for your research, please kindly cite us:
% Functional Connectome Fingerprinting Through Tucker Tensor Decomposition.
% Carvalho, V.; Liu, M.; Harezlak, J.; Estrada Gómez, A.M.; Goñi, J. 
% https://www.mdpi.com/2076-3417/15/9/4821
%
% IMPORTANT: TensorToolbox IS NEEDED!
% Please download TensorToolbox
% https://gitlab.com/tensors/tensor_toolbox/-/releases/v3.5

%% Initialize environment
addpath('C:\\Users\\vitor\\OneDrive\\Documents\\MATLAB\\tensor_toolbox-v3.5\\tensor_toolbox-v3.5') % add path to tensor toolbox
addpath('C:\\Users\\vitor\\Box\\Estrada_Goñi_Collaboration\\Vitor_Files\\MATLAB_Package_STHOSVD') %add path to current directory

task_labels = {'EMOTION', 'REST'}; %tasks in ascending order of scanning length
ranks_ses = [50,100,150,200,250,300,350,400,426];
ranks_parc = [100, 200, 300, 400, 414];
numTasks = numel(task_labels);
numParc = numel(ranks_parc);
numSes = numel(ranks_ses);
numSubjs = 426;

avg_mr_task_414 = zeros(numTasks, numParc, numSes);
avg_err_task_414 = zeros(numTasks, numParc, numSes);

mr_task_rt_hosvd = zeros(numTasks, numParc, numSes);
mr_task_tr_hosvd = zeros(numTasks, numParc, numSes);


% Within-Condition ST-HOSVD 
for task=1:numTasks
    task_t = load(sprintf('%s_%d_test.mat',task_labels{task},numSubjs));
    task_r = load(sprintf('%s_%d_retest.mat',task_labels{task},numSubjs));

    Y_task_test = tensor(task_t.FC3D);
    Y_task_test_m3 = tenmat(Y_task_test, 3);

    Y_task_retest = tensor(task_r.FC3D);
    Y_task_retest_m3 = tenmat(Y_task_retest, 3);
    tic
    count_parc = 0;
    for i = ranks_parc
        count_parc = count_parc + 1;
        count_ses = 0;
        for j = ranks_ses
            count_ses = count_ses + 1;

            % Using Test Estimate Retest
            % Tucker of Y1: Y1 ~= G x P1 x P1 x S1 
            decomp_test = hosvd(Y_task_test, norm(Y_task_test), "ranks", [i,i,j]);
            
            % Compute M1 = (G1 x P1 x P1)^{(3)}
            M_tr = ttm(decomp_test.core, {decomp_test.U{1}, decomp_test.U{1}}, [1 2]);
            M_m3_tr = tenmat(M_tr,3);

            % Estimate P2 = Y2^{(3)} * M_m3^{(+)}
            S_est_tr = Y_task_retest_m3 * pinv(double(M_m3_tr));

            ident_mat = corr(decomp_test.U{3}', double(S_est_tr)');
            [~, ~, mr_task_tr_hosvd(task, count_parc, count_ses)] = f_compute_id_metrics(ident_mat);

            % Using Retest Estimate Rest
            % Tucker of X1: X1 ~= G2 x P2 x P2 x S2  
            decomp_retest = hosvd(Y_task_retest, norm(Y_task_retest), "ranks", [i,i,j]);

            % Compute M2 = (G2 x P2 x P2)^{(3)}
            M_rt = ttm(decomp_retest.core, {decomp_retest.U{1}, decomp_retest.U{1}}, [1 2]);
            M_m3_rt = tenmat(M_rt,3);

            % Estimate C2 = Y^{(3)} x M_m3^(+)
            S_est_rt = Y_task_test_m3 * pinv(double(M_m3_rt));

            % Create Indentifiability Matrix by Cross Multiplying X
            ident_mat = corr(decomp_retest.U{3}', double(S_est_rt)');
            [~, ~, mr_task_rt_hosvd(task, count_parc,count_ses)] = f_compute_id_metrics(ident_mat);
        end
    end
    avg_mr_task_414(task,:,:) = (mr_task_tr_hosvd(task,:,:) + mr_task_rt_hosvd(task,:,:))./2;
end

% Plot EMOTION and REST Plots (Figure 3)
colors = distinguishable_colors(8);

for i = 1:length(task_labels)
    figure; 
    hold on;
    
    plot([50, 100, 150, 200, 250, 300, 350, 400, 426],reshape(avg_mr_task_414(i,1,:), 9,1), '-o', 'MarkerSize', 8, 'MarkerFaceColor',color_lines{1});
    plot([50, 100, 150, 200, 250, 300, 350, 400, 426],reshape(avg_mr_task_414(i,2,:), 9,1), '-o', 'MarkerSize', 8, 'MarkerFaceColor',color_lines{2});
    plot([50, 100, 150, 200, 250, 300, 350, 400, 426],reshape(avg_mr_task_414(i,3,:), 9,1), '-o', 'MarkerSize', 8, 'MarkerFaceColor',color_lines{3});
    plot([50, 100, 150, 200, 250, 300, 350, 400, 426],reshape(avg_mr_task_414(i,4,:), 9,1), '-o', 'MarkerSize', 8, 'MarkerFaceColor',color_lines{4});
    plot([50, 100, 150, 200, 250, 300, 350, 400, 426],reshape(avg_mr_task_414(i,5,:), 9,1), '-o', 'MarkerSize', 8, 'MarkerFaceColor',color_lines{5});
    [val, ind] = max(max(avg_mr_task_414(i,:,:)));
    plot(ranks_ses(ind),val, '-s', 'MarkerSize', 12, 'MarkerFaceColor','magenta')
    
    title(sprintf('%s (%s TPs)', task_labels{i}, task_length{i}));
    xlabel('Participant Rank');
    ylabel('Matching Rate');
    axis([0 426 0 1]);
    colororder(colors);
    if(i == length(task_labels))
        legend("Parcellation Rank 100/414", "Parcellation Rank 200/414", "Parcellation Rank 300/414", "Parcellation Rank 400/414", "Parcellation Rank 414/414", Location="southeast");
    end
    hold off; 
end


% Between-Condition Plot With Full Resting State Scanning Length
task_labels = {'EMOTION'}; 

rest_t = load("REST_426_test.mat");
rest_r = load("REST_426_retest.mat");

Y_rest_t = tensor(rest_t.FC3D);
Y_rest_r = tensor(rest_r.FC3D);

ranks_ses = [50,100,150,200,250,300,350,400,426];
ranks_parc = [100, 200, 300, 400, 414];
numTasks = numel(task_labels);
numParc = numel(ranks_parc);
numSes = numel(ranks_ses);
numSubjs = 426;

mr_rest_tr = zeros(numTasks, numParc, numSes);
mr_rest_rt = zeros(numTasks, numParc, numSes);
avg_mr_rest_task = zeros(numTasks, numParc, numSes);
for task=1:numTasks
    task_t = load(sprintf('%s_%d_test.mat',task_labels{task},numSubjs));
    task_r = load(sprintf('%s_%d_retest.mat',task_labels{task},numSubjs));
   
    Y_task_test = tensor(task_t.FC3D);
    Y_task_test_m3 = tenmat(Y_task_test, 3);

    Y_task_retest = tensor(task_r.FC3D);
    Y_task_retest_m3 = tenmat(Y_task_retest, 3);

    count_parc = 0;
    for i = ranks_parc
        count_parc = count_parc + 1;
        count_ses = 0;
        for j = ranks_ses
            count_ses = count_ses + 1;

            %% Using Rest Test Estimate Task Retest
            % Tucker of Y1: Y1 ~= G x P1 x P1 x S1 
            decomp_rest_test = hosvd(Y_rest_t, norm(Y_rest_t), "ranks", [i,i,j]);
        
            % Compute M = (G x P1 x P1)^{(3)}
            M_tr = ttm(decomp_rest_test.core, {decomp_rest_test.U{1}, decomp_rest_test.U{1}}, [1 2]);
            M_m3_tr = tenmat(M_tr,3);
    
            % Estimate P2 = Y2^{(3)} * M_m3^(+)
            S_est_tr = Y_task_retest_m3 * pinv(double(M_m3_tr));
    
            % Create Indentifiability Matrix by Cross Multiplying X
            ident_mat = corr(decomp_rest_test.U{3}', double(S_est_tr)');
            [~, ~, mr_rest_tr(task,count_parc,count_ses)] = f_compute_id_metrics(ident_mat);

            %% Using Rest Retest Estimate Task Test
            % Tucker of X: X ~= G x A x B x C 
            decomp_rest_retest = hosvd(Y_rest_r, norm(Y_rest_r), "ranks", [i,i,j]);
        
            % Compute (G x A x B) mode 3
            M_rt = ttm(decomp_rest_retest.core, {decomp_rest_retest.U{1}, decomp_rest_retest.U{1}}, [1 2]);
            M_m3_rt = tenmat(M_rt,3);
    
            % Estimate C2 = Y^{(3)} x M_m3^(+)
            S_est_rt = Y_task_test_m3 * pinv(double(M_m3_rt));
    
            % Create Indentifiability Matrix by Cross Multiplying X
            ident_mat = corr(decomp_rest_retest.U{3}', double(S_est_rt)');
            [~, ~, mr_rest_rt(task, count_parc,count_ses)] = f_compute_id_metrics(ident_mat);
            
        end
    end
    avg_mr_rest_task(task,:,:) = (mr_rest_tr(task,:,:) + mr_rest_rt(task,:,:))./2;
end

% Produce Rest-Emotion Plot (Figure 5)
color_lines = {'blue', 'red', 'green', 'black', 'magenta'};
for i = 1:length(task_labels)
    figure; 
    hold on;
    
    plot([50, 100, 150, 200, 250, 300, 350, 400, 426],reshape(avg_mr_rest_task(i,1,:), 9,1), '-o', 'MarkerSize', 8, 'MarkerFaceColor',color_lines{1});
    plot([50, 100, 150, 200, 250, 300, 350, 400, 426],reshape(avg_mr_rest_task(i,2,:), 9,1), '-o', 'MarkerSize', 8, 'MarkerFaceColor',color_lines{2});
    plot([50, 100, 150, 200, 250, 300, 350, 400, 426],reshape(avg_mr_rest_task(i,3,:), 9,1), '-o', 'MarkerSize', 8, 'MarkerFaceColor',color_lines{3});
    plot([50, 100, 150, 200, 250, 300, 350, 400, 426],reshape(avg_mr_rest_task(i,4,:), 9,1), '-o', 'MarkerSize', 8, 'MarkerFaceColor',color_lines{4});
    plot([50, 100, 150, 200, 250, 300, 350, 400, 426],reshape(avg_mr_rest_task(i,5,:), 9,1), '-o', 'MarkerSize', 8, 'MarkerFaceColor',color_lines{5});
    [val(1,i), ind] = max(max(avg_mr_rest_task(i,:,:)));
    
    plot(ranks_ses(ind), val(1,i), '-s', 'MarkerSize', 12, 'MarkerFaceColor','magenta')
    
    title(sprintf('REST - %s', task_labels{i}));
    xlabel('Participant Rank');
    ylabel('Matching Rate');
    axis([0 426 0 1]);
    colororder(colors);
    hold off; 
end




% Comparison between different time point sampling strategies
numSubjs = 426;
task_label = 'EMOTION';

task_length = 166;
rest_ts_t = load("ts_REST_426_test.mat");
rest_ts_r = load("ts_REST_426_retest.mat");

ranks_ses = [50 100 150 200 250 300 350 400 426];
ranks_parc = 414;
parc = 414;
sampling_strats = {'rand', 'consec'};
numReps = 100;
numSes = length(ranks_ses);
numSamps = length(sampling_strats);


d1 = ranks_parc; d2 = ranks_parc; d3=numSubjs;
mr_rest_rt_414_resamp = zeros(numSes, numSamps, numReps);

task_t = load(sprintf('%s_%d_test.mat', task_label,numSubjs));

Y_task_test = tensor(task_t.FC3D);
Y_task_test_data = tenmat(Y_task_test, 3);
clear Y_task_test;

mr_rest_task = zeros(numReps,numSes);
mr_task_rest = zeros(numReps,numSes);
count = 0;

for j=ranks_ses
    count = count+1;
    for i=1:numReps
        count2 = 0;
        for str=sampling_strats
            count2 = count2 + 1;
            
            %% Using Rest Retest estimate Task Test
            FCs_retest = compute_random_tps_FCs(rest_ts_r, numSubjs, parc, task_length, str);

            tens_rest_retest_FCs = tensor(FCs_retest);

            decomp_rt = hosvd(tens_rest_retest_FCs, norm(tens_rest_retest_FCs), "ranks", [ranks_parc,ranks_parc,j]);
            M_rt = ttm(decomp_rt.core, {decomp_rt.U{1}, decomp_rt.U{1}}, [1 2]);
            M_m3_rt = tenmat(M_rt,3);
    
            % Estimate P2 = Y2^{(3)} * M_m3^{(+)}
            S_est_rt = Y_task_test_data * pinv(double(M_m3_rt));
    
            % Create Indentifiability Matrix by Cross Multiplying X
            ident_mat = corr(decomp_rt.U{3}', double(S_est_rt)');
            [~, ~, mr_rest_rt_414_resamp(count, count2, i)] = f_compute_id_metrics(ident_mat);
        end
    end
end

% Produce the first plot (REST-EMOTION) from Figure 7A and 7B.  
session_ranks = {'50','100','150','200','250','300','350','400','426'};

for j = 1:numSamps
    figure
    axis square
    boxplot([reshape(mr_rest_rt_414_resamp(1,j,:), numReps,1),reshape(mr_rest_rt_414_resamp(2,j,:), ...
        numReps,1),reshape(mr_rest_rt_414_resamp(3,j,:), numReps,1),reshape(mr_rest_rt_414_resamp(4,j,:), ...
        numReps,1),reshape(mr_rest_rt_414_resamp(5,j,:), numReps,1),reshape(mr_rest_rt_414_resamp(6,j,:), ...
        numReps,1),reshape(mr_rest_rt_414_resamp(7,j,:), numReps,1),reshape(mr_rest_rt_414_resamp(8,j,:), ...
        numReps,1),reshape(mr_rest_rt_414_resamp(9,j,:), numReps,1)],'Notch','on','Labels', session_ranks);
    ylabel("Matching Rate"); xlabel("Participant Rank"); ylim([0, 1]);
    title(sprintf('REST - %s', task_label));
end
