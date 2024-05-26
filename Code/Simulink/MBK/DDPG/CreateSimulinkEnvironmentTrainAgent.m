%% observstion and action info
% Observation info
obsInfo = rlNumericSpec([3 1],...
    LowerLimit=[-inf -inf -inf]',...
    UpperLimit=[ inf  inf  inf]');

% Name and description are optional and not used by the software
obsInfo.Name = "observations";
obsInfo.Description = "integrated error, error, and measured position";

% Action info
% actInfo = rlNumericSpec([1 1]);
actInfo = rlNumericSpec([1, 1], "UpperLimit", 100, "LowerLimit", -100);
actInfo.Name = "force";
%% environment
env = rlSimulinkEnv("MBK","MBK/RL Agent",...
    obsInfo,actInfo);
env.ResetFcn = @(in)localResetFcn(in);
%% simulation parameter
Ts = 0.10;
Tf = 10.0;
%% critic
% Observation path
statePath = [
    featureInputLayer( ...
        obsInfo.Dimension(1), ...
        Name="obsPathInputLayer")
    fullyConnectedLayer(400)
    reluLayer
    fullyConnectedLayer(300,Name="spOutLayer")
    ];

% Define action path.
actionPath = [
    featureInputLayer( ...
        actInfo.Dimension(1), ...
        Name="actPathInputLayer")
    fullyConnectedLayer(300, ...
        Name="apOutLayer", ...
        BiasLearnRateFactor=0)
    ];

% Define common path.
commonPath = [
    additionLayer(2,Name="add")
    reluLayer
    fullyConnectedLayer(1)
    ];

criticNetwork = dlnetwork();
criticNetwork = addLayers(criticNetwork,statePath);
criticNetwork = addLayers(criticNetwork,actionPath);
criticNetwork = addLayers(criticNetwork,commonPath);
criticNetwork = connectLayers(criticNetwork,"spOutLayer","add/in1");
criticNetwork = connectLayers(criticNetwork,"apOutLayer","add/in2");
criticNetwork = initialize(criticNetwork);

critic = rlQValueFunction(criticNetwork, ...
    obsInfo,actInfo, ...
    ObservationInputNames="obsPathInputLayer", ...
    ActionInputNames="actPathInputLayer");
%% actor
actorNetwork = [
    featureInputLayer(obsInfo.Dimension(1))
    fullyConnectedLayer(400)
    reluLayer
    fullyConnectedLayer(300)
    reluLayer
    fullyConnectedLayer(1)
    tanhLayer
    scalingLayer(Scale=max(actInfo.UpperLimit))
    ];
actorNet = dlnetwork(actorNetwork);
actor = rlContinuousDeterministicActor(actorNet,obsInfo,actInfo);
%% DDPG agent
agent = rlDDPGAgent(actor,critic);
%% hyperparameter
agent.SampleTime = Ts;

agent.AgentOptions.TargetSmoothFactor = 1e-2;
agent.AgentOptions.DiscountFactor = 1.0;
agent.AgentOptions.MiniBatchSize = 512;
agent.AgentOptions.ExperienceBufferLength = 1e6; 

agent.AgentOptions.NoiseOptions.Variance = 0.1;
agent.AgentOptions.NoiseOptions.VarianceDecayRate = 1e-4;

agent.AgentOptions.CriticOptimizerOptions.LearnRate = 1e-4;
agent.AgentOptions.CriticOptimizerOptions.GradientThreshold = 1;
agent.AgentOptions.ActorOptimizerOptions.LearnRate = 1e-4;
agent.AgentOptions.ActorOptimizerOptions.GradientThreshold = 1;
%% train agent
trainOpts = rlTrainingOptions(...
    MaxEpisodes=1000, ...
    MaxStepsPerEpisode=ceil(Tf/Ts), ...
    ScoreAveragingWindowLength=20, ...
    Verbose=false, ...
    Plots="training-progress",...
    StopTrainingCriteria="AverageReward",...
    StopTrainingValue=-5);
%% train
doTraining = true;

if doTraining
    % Train the agent.
    trainingStats = train(agent,env,trainOpts);
else
    % Load the pretrained agent for the example.
    load("MBK_DDPG.mat","agent")
end



















function in = localResetFcn(in)

% Randomize reference signal
blk = sprintf("MBK/ref position");
position = (rand(1, 1)-0.5)*10;

in = setBlockParameter(in,blk,Value=num2str(position));

% Randomize initial position
init = (rand(1, 1)-0.5)*10;
blk = "MBK/MBK System/Integrator1";
in = setBlockParameter(in,blk,InitialCondition=num2str(init));

end