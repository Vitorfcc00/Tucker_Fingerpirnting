function [Idiff,ID_rate,M_rate,mask_subj_1,mask_subj_2] = f_compute_id_metrics(I)
%  function developed by Dr. Kausar Abbas (CONNplexity Lab, Purdue University)

%% Computes three fingerprinting metrics
%  INPUT :
%             - I : Identifiability matrix (nSubj x nSubj)
%  OUTPUTS : 
%             - Idiff   : differential identifiaiblity score 
%                                The quest for identifiability in human functional connectomes (2018) Amico E, Goñi J. Scientific Reports 8.1: 8254
%                                https://doi.org/10.1038/s41598-018-25089-1
%
%             - ID_rate : identification rate 
%                                Functional connectome fingerprinting: identifying individuals using patterns of brain connectivity (2015).                               
%                                Finn, E. S., Shen, X., Scheinost, D., Rosenberg, M. D., Huang, J., Chun, M. M., ... & Constable, R. T. Nature neuroscience, 18(11), 1664-1671.  
%                                https://doi.org/10.1038/nn.4135
%
%             - M_rate  : matching rate
%                               Improving Functional Connectome Fingerprinting with Degree-Normalization (2021) 
%                               Chiem B, Abbas K, Amico E, Duong-Tran DA, Crevecoeur F, and Goñi J. Brain connectivity, 2021 (in press).
%                               https://doi.org/10.1089/brain.2020.0968

%% Idiff
numSubjs = size(I,1);
mask_diag = logical(eye(numSubjs));

Iself = mean(I(mask_diag));
Iothers = mean(I(~mask_diag));

Idiff = (Iself - Iothers)*100;

%% ID rate
[~,ind_1] = max(I,[],1);
[~,ind_2] = max(I,[],2);
    
ID_1 = nnz(ind_1==1:numSubjs)/numSubjs;
ID_2 = nnz(ind_2==[1:numSubjs]')/numSubjs;
ID_rate = (ID_1+ID_2)/2;

%% Matching Rate
% Test = database; Retest = target
II = I;
mask_subj_1 = false(numSubjs,1);
for i = 1:numSubjs
    [maxSim,ind_base] = nanmax(II,[],1);
    [~,ind_target] = nanmax(maxSim);
    
    if ind_target == ind_base(ind_target)
        mask_subj_1(ind_target) = true;
    end
    
    II(:,ind_target) = nan;
    II(ind_base(ind_target),:) = nan;
end
mR_1 = nnz(mask_subj_1)/numSubjs;

% Test = target; Retest = database
II = I';
mask_subj_2 = false(numSubjs,1);
for i = 1:numSubjs
    [maxSim,ind_base] = nanmax(II,[],1);
    [~,ind_target] = nanmax(maxSim);
    
    if ind_target == ind_base(ind_target)
        mask_subj_2(ind_target) = true;
    end
    
    II(:,ind_target) = nan;
    II(ind_base(ind_target),:) = nan;
end
mR_2 = nnz(mask_subj_2)/numSubjs;

M_rate = (mR_1 + mR_2)./2;

