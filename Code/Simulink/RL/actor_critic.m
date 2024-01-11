% load pretrained actor
all = readmatrix('trajectory.csv');
initial_condition = all(1, :);
X0 = initial_condition(1:2);
dot_X0 = initial_condition(4:5);
all_new = interp1(1:length(all), all ,1:20:length(all));
all_new = all_new';
trajectory = zeros(4, length(all_new));
trajectory(1:2, :) = all_new(1:2, :);
trajectory(3:4, :) = all_new(4:5, :);


observationInfo = rlNumericSpec([4 1]);
actionInfo = rlNumericSpec([2, 1], "UpperLimit", 1, "LowerLimit", -1);



mdl = 'RL';
agentBlk = [mdl '/RL Agent'];

env = rlSimulinkEnv(mdl,agentBlk,observationInfo,actionInfo);
env.ResetFcn = @(in)setVariable(in,['dot_X0', 'X0'] ,...
    [initial_condition(4:5), initial_condition(1:2)],'Workspace',mdl);
