%% plot the trajectory
trajectory = readmatrix("trajectory.csv");
x = trajectory(:, 1);
y = trajectory(:, 2);
z = trajectory(:, 3);
xdot = trajectory(:, 4);
ydot = trajectory(:, 5);
zdot = trajectory(:, 6);
plot(x, y, "Color", 'b', 'LineWidth',2)
