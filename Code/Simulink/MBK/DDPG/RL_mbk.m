obsInfo = rlNumericSpec([1 1]);
% actrange = -1:0.1:1;
% actInfo = rlFiniteSetSpec(actrange);
actInfo = rlNumericSpec([1, 1], "UpperLimit", 10, "LowerLimit", -10);

mdl = 'MBK';
agentBlk = [mdl '/RL Agent'];


env = rlSimulinkEnv(mdl,agentBlk,obsInfo,actInfo);
env.ResetFcn = @(in)setVariable(in,'x0',(rand(1, 1)-0.5)*10,'Workspace',mdl);

