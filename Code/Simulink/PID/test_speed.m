tic
for i = 1:20
    sim('three_body_PID.slx');
end
toc
tic
for i = 1:20
    sim('three_body_PID_matlab_function.slx');
end
toc
