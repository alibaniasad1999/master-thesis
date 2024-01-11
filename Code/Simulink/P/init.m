%% plot the trajectory
trajectory = readmatrix("trajectory.csv");
X0 = trajectory(1, 1:2);
dot_X0 = trajectory(1, 4:5);
mu = 0.012277471;