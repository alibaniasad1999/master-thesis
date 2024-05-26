clear;
clc;
%% observstion and action info
% Observation info
m = 10;
b = 2;
k = 5;
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
mdl = 'MBK_DDPG_I';
agentBlk = [mdl '/RL Agent'];
env = rlSimulinkEnv(mdl,agentBlk,obsInfo,actInfo);
env.ResetFcn = @(in)setVariable(in,'x0',(rand(1, 1)-0.5)*10,'Workspace',mdl);
%% Time and step time
Ts = 0.05;
Tf = 20;
%% Create DDPG Agent
%% Critic
% Define state path.
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
critic = rlQValueFunction(criticNetwork, ...
    obsInfo,actInfo, ...
    ObservationInputNames="obsPathInputLayer", ...
    ActionInputNames="actPathInputLayer");
%% Actor
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
actorNetwork = dlnetwork(actorNetwork);
actorNetwork = initialize(actorNetwork);
actor = rlContinuousDeterministicActor(actorNetwork,obsInfo,actInfo);
%% Options
criticOpts = rlOptimizerOptions(LearnRate=1e-03,GradientThreshold=1);
actorOpts = rlOptimizerOptions(LearnRate=1e-03,GradientThreshold=1);
agentOpts = rlDDPGAgentOptions(...
    SampleTime=Ts,...
    CriticOptimizerOptions=criticOpts,...
    ActorOptimizerOptions=actorOpts,...
    ExperienceBufferLength=1e5,...
    DiscountFactor=0.99,...
    MiniBatchSize=128);
agentOpts.NoiseOptions.StandardDeviation = 0.6;
%% Agent
agent = rlDDPGAgent(actor,critic,agentOpts);
%% Train Option
maxepisodes = 5000;
maxsteps = ceil(Tf/Ts);
trainOpts = rlTrainingOptions(...
    MaxEpisodes=maxepisodes,...
    MaxStepsPerEpisode=maxsteps,...
    ScoreAveragingWindowLength=5,...
    Verbose=false,...
    Plots="training-progress",...
    StopTrainingCriteria="AverageReward",...
    StopTrainingValue=-50,...
    SaveAgentCriteria="EvaluationStatistic",...
    SaveAgentValue=-740);
%% Training
choice = menu('Do yo want to train?','Yes', 'No, just load Modle');
if choice == 1
    evaluator = rlEvaluator(...
    NumEpisodes=1,...
    EvaluationFrequency=10); 
    % Train the agent.
    trainingStats = train(agent,env,trainOpts,Evaluator=evaluator);
else
    msg = sprintf('I just  loaded Agent, have a nice day!');
    h = msgbox(msg);
    load('DDPG_MBK_I.mat')
    open_system(mdl)
end
