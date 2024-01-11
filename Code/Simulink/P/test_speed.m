init
tic
for i = 1:20
    sim('three_body_P.slx');
end
toc
tic
for i = 1:20
    sim('three_body_P_matlab_function.slx');
end
toc
