%% simulate the chunking task

% training properties
Ntimes = 2;
learning_rate = 0.3;
num_examples = 100;
batch_size = 50;
num_iterations = 40000;
init_scale = 0.5;

% simulation properties
NObjects = 4;
NPositions = 4;
NLoadObjects = 2;
NLoadPositions = 2;
NWorkingMemoryItems = (NObjects + NLoadObjects) * (NPositions + NLoadPositions);

% network properties
Ninput = NWorkingMemoryItems + NObjects + NLoadObjects;
Nhidden = 25;
Ncontrol = 25;
Noutput = NPositions + NLoadPositions;
%% train on permutations of 6 objects
% create the training data
inputs = zeros(num_examples, Ninput, Ntimes);
labels = zeros(num_examples, Noutput, Ntimes);
for i = 1:num_examples
    object_indices = 1:NPositions;
    % pick the location of A
    A_index = round(rand * (NPositions - 1)) + 1;
    
    % with 80% probability, B is after A
    if rand < 0.8
        B_index = mod(A_index + 1, NPositions);
        if B_index == 0
            B_index = NPositions;
        end
    else % otherwise, it is before A
        B_index = mod(A_index - 1, NPositions);
        if B_index == 0
            B_index = NPositions;
        end
    end
    
    % pick the locations of the remaining distractor objects
    distractor_indices = object_indices;
    distractor_indices([find(object_indices == A_index), find(object_indices == B_index)]) = [];
    distractor_indices = distractor_indices(randperm(length(distractor_indices))); % randomly permute the distractors
            
    object_indices = [A_index, B_index, distractor_indices];
    
    % pick the locations of the memory load objects
    load_object_indices = (NPositions + 1):(NPositions + NLoadPositions);
    load_object_indices = load_object_indices(randperm(length(load_object_indices)));
    for j = 1:length(load_object_indices)
        if rand < 0.5
            load_object_indices(j) = 0;
        end
    end

    object_indices = [object_indices, load_object_indices];
    
    % convert object indices to task layer representation
    for j = 1:length(object_indices)
        if object_indices(j) == 0
            continue
        end

        index = ((j - 1) * (NPositions + NLoadPositions)) + object_indices(j);
        inputs(i, index, 1) = 1;   
    end
    
    % pick an object to cue
    object_set = 1:(NObjects + NLoadObjects);
    object_set(find(object_indices == 0)) = []; % make sure that we can't cue null objects
    object_set(3:(NObjects - NLoadObjects)) = []; % make sure that we don't cue the distractors

    inputs(i, NWorkingMemoryItems + object_set(round(rand * (length(object_set) - 1)) + 1), Ntimes) = 1;
    
    % set the corresponding label
    labels(i, object_indices(find(inputs(i, (NWorkingMemoryItems + 1):end, Ntimes))), Ntimes) = 1;
end
%% train on permutations of 6 objects, new data every time
% create the training data
net = RNNmodel(Ninput, Nhidden, Noutput, Ncontrol, learning_rate, init_scale);
mse_log = zeros(1, num_iterations);
for k = 1:num_iterations
    inputs = zeros(num_examples, Ninput, Ntimes);
    labels = zeros(num_examples, Noutput, Ntimes);
    for i = 1:num_examples
        object_indices = 1:NPositions;
        % pick the location of A
        A_index = round(rand * (NPositions - 1)) + 1;

        % with 80% probability, B is after A
        if rand < 0.8
            B_index = mod(A_index + 1, NPositions);
            if B_index == 0
                B_index = NPositions;
            end
        else % otherwise, it is before A
            B_index = mod(A_index - 1, NPositions);
            if B_index == 0
                B_index = NPositions;
            end
        end

        % pick the locations of the remaining distractor objects
        distractor_indices = object_indices;
        distractor_indices([find(object_indices == A_index), find(object_indices == B_index)]) = [];
        distractor_indices = distractor_indices(randperm(length(distractor_indices))); % randomly permute the distractors

        object_indices = [A_index, B_index, distractor_indices];

        % pick the locations of the memory load objects
        load_object_indices = (NPositions + 1):(NPositions + NLoadPositions);
        load_object_indices = load_object_indices(randperm(length(load_object_indices)));
        for j = 1:length(load_object_indices)
            if rand < 0.5
                load_object_indices(j) = 0;
            end
        end

        object_indices = [object_indices, load_object_indices];

        % convert object indices to task layer representation
        for j = 1:length(object_indices)
            if object_indices(j) == 0
                continue
            end

            index = ((j - 1) * (NPositions + NLoadPositions)) + object_indices(j);
            inputs(i, index, 1) = 1;   
        end

        % pick an object to cue
        object_set = 1:(NObjects + NLoadObjects);
        object_set(find(object_indices == 0)) = []; % make sure that we can't cue null objects
        object_set(3:(NObjects - NLoadObjects)) = []; % make sure that we don't cue the distractors

        inputs(i, NWorkingMemoryItems + object_set(round(rand * (length(object_set) - 1)) + 1), Ntimes) = 1;

        % set the corresponding label
        labels(i, object_indices(find(inputs(i, (NWorkingMemoryItems + 1):end, Ntimes))), Ntimes) = 1;
    end
    mse_log(1, k) = net.trainOnline(inputs, labels, Ntimes, batch_size, 1, 1, 1);
    disp(k);
end
%% train on permutations of 2 objects
% create the training data
inputs = zeros(num_examples, Ninput, Ntimes);
labels = zeros(num_examples, Noutput, Ntimes);
for i = 1:num_examples
    object_indices = 1:NPositions;
    % pick the location of A
    A_index = round(rand * (NPositions - 1)) + 1;
    
    % with 80% probability, B is after A
    if rand < 0.8
        B_index = mod(A_index + 1, NPositions);
        if B_index == 0
            B_index = NPositions;
        end
    else % otherwise, it is before A
        B_index = mod(A_index - 1, NPositions);
        if B_index == 0
            B_index = NPositions;
        end
    end
    
    % for this training, we never have distractors
    object_indices = [A_index, B_index, 0, 0];
    
    % pick the locations of the memory load objects
    load_object_indices = (NPositions + 1):(NPositions + NLoadPositions);
    load_object_indices = load_object_indices(randperm(length(load_object_indices)));

    object_indices = [object_indices, load_object_indices];
    delete_indices = [1 2 (NPositions + 1):(NPositions + NLoadPositions)];
    delete_indices = delete_indices(randperm(length(delete_indices)));
    % convert object indices to task layer representation
    for j = 1:length(object_indices)
        if object_indices(j) == 0 || ismember(j, delete_indices(1:2))
            object_indices(j) = 0;
            continue
        end

        index = ((j - 1) * (NPositions + NLoadPositions)) + object_indices(j);
        inputs(i, index, 1) = 1;   
    end
    
    % pick an object to cue
    object_set = 1:(NObjects + NLoadObjects);
    object_set(find(object_indices == 0)) = []; % make sure that we can't cue null objects
    object_set(3:(NObjects - NLoadObjects)) = []; % make sure that we don't cue the distractors

    inputs(i, NWorkingMemoryItems + object_set(round(rand * (length(object_set) - 1)) + 1), Ntimes) = 1;
    
    % set the corresponding label
    labels(i, object_indices(find(inputs(i, (NWorkingMemoryItems + 1):end, Ntimes))), Ntimes) = 1;
end
%% train on the entire input space
input_space_size = 24 * 7 * 4; % upper bound: 4! arrangements of A,B,C,D, 7 arrangements of E, F, 4 objects to cue
% true size ends up being 528

main_object_perms = perms(1:NPositions);
load_object_perms = [0 0; 5 0; 6 0; 0 5; 0 6; 5 6; 6 5];
object_cues = [1 2 5 6];

object_indices = zeros(input_space_size, NObjects + NLoadObjects + 1);

counter = 1;
for i = 1:size(main_object_perms, 1)
    for j = 1:size(load_object_perms, 1)
        for k = 1:size(object_cues, 2)
            if j == 1 && k > 2
                continue
            elseif (j == 2 || j == 3) && k == 4
                continue
            elseif (j == 4 || j == 5) && k == 3
                continue
            end
            object_indices(counter, :) = [main_object_perms(i, :) load_object_perms(j, :) object_cues(k)];
            counter = counter + 1;
        end
    end
end

% trim rows of all zeros because i don't want to do the math
object_indices(all(object_indices == 0, 2), :) = [];
input_space_size = size(object_indices, 1);
batch_size = input_space_size / 2;

inputs = zeros(input_space_size, Ninput, Ntimes);
labels = zeros(input_space_size, Noutput, Ntimes);

for i = 1:input_space_size
    % convert object indices to task layer representation
    for j = 1:(length(object_indices(i, :)) - 1) % ignore the object cue
        if object_indices(i, j) == 0
            continue
        end

        index = ((j - 1) * (NPositions + NLoadPositions)) + object_indices(i, j);
        inputs(i, index, 1) = 1;   
    end
    
    inputs(i, NWorkingMemoryItems + object_indices(i, end), Ntimes) = 1;
    labels(i, object_indices(i, find(inputs(i, (NWorkingMemoryItems + 1):end, Ntimes))), Ntimes) = 1;
end

%% train
net = RNNmodel(Ninput, Nhidden, Noutput, Ncontrol, learning_rate, init_scale);
mse_log = net.trainOnline(inputs, labels, Ntimes, batch_size, num_iterations, 1, 0);
%% check performance on test data
test_set_size = 600;

% create the test data
test_inputs = zeros(test_set_size, Ninput, Ntimes);
test_labels = zeros(test_set_size, Noutput, Ntimes);

% first half have the correlation
for i = 1:(test_set_size/2)
    object_indices = 1:NPositions;
    % pick the location of A
    A_index = round(rand * (NPositions - 1)) + 1;
    B_index = mod(A_index + 1, NPositions);
    if B_index == 0
        B_index = NPositions;
    end
    
    % pick the locations of the remaining distractor objects
    distractor_indices = object_indices;
    distractor_indices([find(object_indices == A_index), find(object_indices == B_index)]) = [];
    distractor_indices = distractor_indices(randperm(length(distractor_indices))); % randomly permute the distractors
            
    object_indices = [A_index, B_index, distractor_indices];
    
    % pick the locations of the memory load objects
    load_object_indices = (NPositions + 1):(NPositions + NLoadPositions);
    load_object_indices = load_object_indices(randperm(length(load_object_indices)));
    
    if i <= (test_set_size / 6)
        load_object_indices = [0, 0];
    elseif (test_set_size / 6) < i && i < ((2 * test_set_size) / 6)
        load_object_indices(2) = 0;
    end

    object_indices = [object_indices, load_object_indices];
    
    % convert object indices to task layer representation
    for j = 1:length(object_indices)
        if object_indices(j) == 0
            continue
        end

        index = ((j - 1) * (NPositions + NLoadPositions)) + object_indices(j);
        test_inputs(i, index, 1) = 1;   
    end
    
    % pick an object to cue
    object_set = 1:(NObjects + NLoadObjects);
    object_set(find(object_indices == 0)) = []; % make sure that we can't cue null objects
    object_set(3:(NObjects - NLoadObjects)) = []; % make sure that we don't cue the distractors

    test_inputs(i, NWorkingMemoryItems + object_set(round(rand * (length(object_set) - 1)) + 1), Ntimes) = 1;
    
    % set the corresponding label
    test_labels(i, object_indices(find(test_inputs(i, (NWorkingMemoryItems + 1):end, Ntimes))), Ntimes) = 1;
end

for i = ((test_set_size/2) + 1):test_set_size
    object_indices = 1:NPositions;
    % pick the location of A
    A_index = round(rand * (NPositions - 1)) + 1;
    B_index = mod(A_index - 1, NPositions);
    if B_index == 0
        B_index = NPositions;
    end
    
    % pick the locations of the remaining distractor objects
    distractor_indices = object_indices;
    distractor_indices([find(object_indices == A_index), find(object_indices == B_index)]) = [];
    distractor_indices = distractor_indices(randperm(length(distractor_indices))); % randomly permute the distractors
            
    object_indices = [A_index, B_index, distractor_indices];
    
    % pick the locations of the memory load objects
    load_object_indices = (NPositions + 1):(NPositions + NLoadPositions);
    load_object_indices = load_object_indices(randperm(length(load_object_indices)));
    
    if i <= ((4 * test_set_size) / 6)
        load_object_indices = [0, 0];
    elseif ((4 * test_set_size) / 6) < i && i < ((5 * test_set_size) / 6)
        load_object_indices(2) = 0;
    end

    object_indices = [object_indices, load_object_indices];
    
    % convert object indices to task layer representation
    for j = 1:length(object_indices)
        if object_indices(j) == 0
            continue
        end

        index = ((j - 1) * (NPositions + NLoadPositions)) + object_indices(j);
        test_inputs(i, index, 1) = 1;   
    end
    
    % pick an object to cue
    object_set = 1:(NObjects + NLoadObjects);
    object_set(find(object_indices == 0)) = []; % make sure that we can't cue null objects
    object_set(3:(NObjects - NLoadObjects)) = []; % make sure that we don't cue the distractors

    test_inputs(i, NWorkingMemoryItems + object_set(round(rand * (length(object_set) - 1)) + 1), Ntimes) = 1;
    
    % set the corresponding label
    test_labels(i, object_indices(find(test_inputs(i, (NWorkingMemoryItems + 1):end, Ntimes))), Ntimes) = 1;
end
%% train with tests
net = RNNmodel(Ninput, Nhidden, Noutput, Ncontrol, learning_rate, init_scale);
small_set_errors = zeros(2, num_iterations); % row 1 is correlation, row 2 is anticorrelation
mid_set_errors = zeros(2, num_iterations);
big_set_errors = zeros(2, num_iterations);
mse_log = zeros(1, num_iterations);
for i = 1:num_iterations
    disp(i);
    mse_log(1, i) = net.trainOnline(inputs, labels, Ntimes, batch_size, 1, 1, 1);
    small_set_errors(1, i) = net.costSet(test_inputs(1:100, :, :), Ntimes, test_labels(1:100, :, :));
    small_set_errors(2, i) = net.costSet(test_inputs(301:400, :, :), Ntimes, test_labels(301:400, :, :));
    mid_set_errors(1, i) = net.costSet(test_inputs(101:200, :, :), Ntimes, test_labels(101:200, :, :));
    mid_set_errors(2, i) = net.costSet(test_inputs(401:500, :, :), Ntimes, test_labels(401:500, :, :));
    big_set_errors(1, i) = net.costSet(test_inputs(201:300, :, :), Ntimes, test_labels(201:300, :, :));  
    big_set_errors(2, i) = net.costSet(test_inputs(501:600, :, :), Ntimes, test_labels(501:600, :, :));
end
%%
plot(mse_log);
%%
hold on;
plot(small_set_errors(1, :));
plot(small_set_errors(2, :));
plot(mid_set_errors(1, :));
plot(mid_set_errors(2, :));
plot(big_set_errors(1, :));
plot(big_set_errors(2, :));
legend('small, correlated', 'small, anticorrelated', 'mid, correlated', 'mid, anticorrelated', 'big, correlated', 'big, anticorrelated');
hold off;
%% compare predictions with labels
outs = net.predictSet(inputs, Ntimes);

figure(1);
subplot(1, 2, 1);
imagesc(outs(:, :, end));
colorbar;
caxis([0 1]);
subplot(1, 2, 2);
imagesc(labels(:, :, end));
colorbar;
caxis([0 1]);
%% get activation correlations

corr_inputs_A = zeros(NPositions, Ninput, Ntimes);
corr_inputs_A(1, 1, 1) = 1;
corr_inputs_A(2, 2, 1) = 1;
corr_inputs_A(3, 3, 1) = 1;
corr_inputs_A(4, 4, 1) = 1;

corr_inputs_B = zeros(NPositions, Ninput, Ntimes);
corr_inputs_B(1, 7, 1) = 1;
corr_inputs_B(2, 8, 1) = 1;
corr_inputs_B(3, 9, 1) = 1;
corr_inputs_B(4, 10, 1) = 1;

corr_inputs_D = zeros(NPositions, Ninput, Ntimes);
corr_inputs_D(1, 19, 1) = 1;
corr_inputs_D(2, 20, 1) = 1;
corr_inputs_D(3, 21, 1) = 1;
corr_inputs_D(4, 22, 1) = 1;

[a_hids_A, a_cons_A, a_outs, z_hids, z_cons, z_outs] = net.predictSetVerbose(corr_inputs_A, Ntimes);
[a_hids_B, a_cons_B, a_outs, z_hids, z_cons, z_outs] = net.predictSetVerbose(corr_inputs_B, Ntimes);
[a_hids_D, a_cons_D, a_outs, z_hids, z_cons, z_outs] = net.predictSetVerbose(corr_inputs_D, Ntimes);


context_A_mean = mean(a_cons_A(:, :, end), 1);
hidden_A_mean = mean(a_hids_A(:, :, end), 1);

context_B_mean = mean(a_cons_B(:, :, end), 1);
hidden_B_mean = mean(a_hids_B(:, :, end), 1);

context_D_mean = mean(a_cons_D(:, :, end), 1);
hidden_D_mean = mean(a_hids_D(:, :, end), 1);

corr(transpose(context_A_mean), transpose(context_B_mean))
corr(transpose(hidden_A_mean), transpose(hidden_B_mean))
corr(transpose(context_A_mean), transpose(context_D_mean))
corr(transpose(hidden_A_mean), transpose(hidden_D_mean))
