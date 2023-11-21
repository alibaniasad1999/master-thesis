% load pretrained actor
% initial_condition = readmatrix('x0.csv');
% dot_X0 = initial_condition(4:5);
% X0 = initial_condition(1:2);
% all = readmatrix('all.csv');
% all_new = interp1(1:length(all), all ,1:20:length(all));
% all_new = all_new';
% trajectory = zeros(4, length(all_new));
% trajectory(1:2, :) = all_new(1:2, :);
% trajectory(3:4, :) = all_new(4:5, :);
trajectory = readmatrix("trajectory.csv");

observationInfo = rlNumericSpec([2 1]);
actionInfo = rlNumericSpec([2, 1], "UpperLimit", 10, "LowerLimit", -10);



mdl = 'AOM_RL';
agentBlk = [mdl '/RL Agent'];

% initial_condition = readmatrix('x0.csv');
env = rlSimulinkEnv(mdl,agentBlk,observationInfo,actionInfo);
env.ResetFcn = @(in)setVariable(in,['dot_X0', 'X0'] ,...
    [trajectory(1, 4:5), trajectory(1, 1:2)],'Workspace',mdl);
