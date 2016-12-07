classdef RNNmodel < handle
    properties(SetAccess = public)
        Ninput;          % input units
        Nhidden;         % hidden units
        Noutput;         % output units
        Ncontrol;        % control units
        
        l_rate;          % learning rate
        weights;         % network weights
        biases;          % network biases
    end
    
    methods
        function this = RNNmodel(varargin)
            % copy constructor
            if (isa(varargin{1}, 'RNNmodel'))
                copy = varargin{1};
                this.Ninput = copy.Ninput;
                this.Nhidden = copy.Nhidden;
                this.Noutput = copy.Noutput;
                this.Ncontrol = copy.Ncontrol;
                
                this.l_rate = copy.l_rate;
                this.weights = copy.weights;
                this.biases = copy.biases;
                
                return;
            end  
            % parse arguments
            % constructor shape: 
            % (Ninput, Nhidden, Noutput, Ncontrol, Ntimes, l_rate, init_scale)       
            
            % require layer sizes to always be specified at
            % construction
            this.Ninput = varargin{1};
            this.Nhidden = varargin{2};
            this.Noutput = varargin{3};
            this.Ncontrol = varargin{4};

            if (length(varargin) >= 5)
                this.l_rate = varargin{5};
            else
                this.l_rate = 0.3;
            end
            
            if (length(varargin) >= 6)
                init_scale = varargin{6};
            else
                init_scale = 1;
            end
            
            % init weights and biases
            this.weights.w_IH = (-1 + 2.*rand(this.Nhidden,this.Ninput)) * init_scale;
            this.weights.w_HO = (-1 + 2.*rand(this.Noutput,this.Nhidden)) * init_scale;
            this.weights.w_HC = (-1 + 2.*rand(this.Ncontrol,this.Nhidden)) * init_scale;
            this.weights.w_CH = (-1 + 2.*rand(this.Nhidden,this.Ncontrol)) * init_scale;
            this.weights.w_CO = (-1 + 2.*rand(this.Noutput,this.Ncontrol)) * init_scale;
            
            this.biases.b_H = -2 * ones(this.Nhidden, 1); % (-1 + 2.*rand(this.Nhidden, 1)) * init_scale;
            this.biases.b_O = -2 * ones(this.Noutput, 1); % (-1 + 2.*rand(this.Noutput, 1)) * init_scale;
            this.biases.b_C = -2 * ones(this.Ncontrol, 1); % (-1 + 2.*rand(this.Ncontrol, 1)) * init_scale;
        end
        
        % train the network using the given input and label set
        % time step is the third dimension of the data sets.
        % num_steps is the number of time steps that the network is
        % simulating.
        function [mse_log] = trainOnline(this, input_set, label_set, num_steps, batch_size, num_iterations, include_intermediates)
            assert(size(input_set, 3) == size(label_set, 3)); % they better have values for each time step
            assert(size(input_set, 1) == size(label_set, 1)); % they better have the same amount of examples
            assert(num_steps >= 0);
            assert(num_iterations >= 0);
            assert(batch_size >= 1);
            
            num_examples = size(input_set, 1);
            mse_log = zeros(1, num_iterations);
            for i = 1:num_iterations
                % randomly permute the sets before training
                 perm_indices = randperm(num_examples);
                 input_set = input_set(perm_indices, :, :);
                 label_set = label_set(perm_indices, :, :);
                
                % for each batch...
                for j = 1:ceil(num_examples / batch_size)
                    b_start = ((j - 1) * batch_size) + 1;
                    b_end = b_start + batch_size - 1;
                    [dweights, dbiases] = this.trainBatch(input_set(b_start:b_end, :, :), label_set(b_start:b_end, :, :), num_steps, include_intermediates);

                    this.weights.w_IH = this.weights.w_IH - (this.l_rate * dweights.w_IH);
                    this.weights.w_HO = this.weights.w_HO - (this.l_rate * dweights.w_HO);
                    this.weights.w_HC = this.weights.w_HC - (this.l_rate * dweights.w_HC);                   
                    this.weights.w_CH = this.weights.w_CH - (this.l_rate * dweights.w_CH);                   
                    this.weights.w_CO = this.weights.w_CO - (this.l_rate * dweights.w_CO);
                   
                    this.biases.b_H = this.biases.b_H - (this.l_rate * dbiases.b_H);
                    this.biases.b_O = this.biases.b_O - (this.l_rate * dbiases.b_O);
                    this.biases.b_C = this.biases.b_C - (this.l_rate * dbiases.b_C); 
                end
                set_cost = this.costSet(input_set, num_steps, label_set);
                disp([num2str(i) ' ' num2str(set_cost)]);
                mse_log(1, i) = set_cost;
            end
        end
        
        % train a batch of inputs 
        function [dweights, dbiases] = trainBatch(this, input_set, label_set, num_steps, include_intermediates)
            batch_size = size(input_set, 1);
            
            b_O_deltas = zeros(this.Noutput, 1, batch_size);
            b_H_deltas = zeros(this.Nhidden, 1, batch_size);
            b_C_deltas = zeros(this.Ncontrol, 1, batch_size);
            
            w_IH_deltas = zeros(this.Nhidden, this.Ninput, batch_size);
            w_HO_deltas = zeros(this.Noutput, this.Nhidden, batch_size);
            w_HC_deltas = zeros(this.Ncontrol, this.Nhidden, batch_size);
            w_CH_deltas = zeros(this.Nhidden, this.Ncontrol, batch_size);
            w_CO_deltas = zeros(this.Noutput, this.Ncontrol, batch_size);
            
            w = this.weights;
            
            % for every example...
            for i = 1:batch_size
                % feed forward
                [a_hid, a_con, a_out, z_hid, z_con, z_out] = runSample(this, input_set(i, :, :), num_steps);
                final_out = a_out(:, :, end);
                
                % calculate deltas
                
                % final layer stuff, using the output
                err = this.error(final_out, permute(label_set(i, :, end), [2 1 3]));
                dO = err .* this.sigmoid_prime(z_out(:, :, end));
                
                b_O_deltas(:, :, i) = dO;
                w_HO_deltas(:, :, i) = dO * permute(a_hid(:, :, end), [2 1 3]);
                w_CO_deltas(:, :, i) = dO * permute(a_con(:, :, end), [2 1 3]);
                
                dH = (transpose(w.w_HO) * dO) .* this.sigmoid_prime(z_hid(:, :, end));
                b_H_deltas(:, :, i) = dH;
                w_IH_deltas(:, :, i) = dH * input_set(i, :, end);
                w_CH_deltas(:, :, i) = dH * permute(a_con(:, :, end), [2 1 3]);
                
                dC = (transpose(w.w_CO) * dO) .* this.sigmoid_prime(z_con(:, :, end));
                dC = dC + (transpose(w.w_CH) * dH) .* this.sigmoid_prime(z_con(:, :, end));
                b_C_deltas(:, :, i) = dC;
                w_HC_deltas(:, :, i) = dC * permute(a_hid(:, :, end - 1), [2 1 3]);
                
                % the rest of the calculations don't need the final layer
                % stuff, so we can automate
                for j = (num_steps):-1:2
                    dOcurr = 0;
                    if include_intermediates == 1
                        layer_err = this.error(a_out(:, :, j), permute(label_set(i, :, j), [2 1 3]));
                        dOcurr = layer_err .* this.sigmoid_prime(z_out(:, :, j));

                        b_O_deltas(:, :, i) = b_O_deltas(:, :, i) + dOcurr;
                        w_HO_deltas(:, :, i) = w_HO_deltas(:, :, i) + (dOcurr * permute(a_hid(:, :, j), [2 1 3]));
                        w_CO_deltas(:, :, i) = w_CO_deltas(:, :, i) + (dOcurr * permute(a_con(:, :, j), [2 1 3]));
                    end
                    
                    dHcurr = (transpose(w.w_HC) * dC) .* this.sigmoid_prime(z_hid(:, :, j));
                    dCcurr = (transpose(w.w_CH) * dHcurr) .* this.sigmoid_prime(z_con(:, :, j));
                    
                    if include_intermediates == 1
                        dHcurr = dHcurr + (transpose(w.w_HO) * dOcurr) .* this.sigmoid_prime(z_hid(:, :, j));
                        dCcurr = dCcurr + (transpose(w.w_CO) * dOcurr) .* this.sigmoid_prime(z_con(:, :, j));
                    end
                    
                    b_H_deltas(:, :, i) = b_H_deltas(:, :, i) + dHcurr;
                    b_C_deltas(:, :, i) = b_C_deltas(:, :, i) + dCcurr;
                    
                    w_IH_deltas(:, :, i) = w_IH_deltas(:, :, i) + (dHcurr * input_set(i, :, j));
                    w_CH_deltas(:, :, i) = w_CH_deltas(:, :, i) + (dHcurr * permute(a_con(:, :, j), [2 1 3]));
                    w_HC_deltas(:, :, i) = w_HC_deltas(:, :, i) + (dCcurr * permute(a_hid(:, :, j - 1), [2 1 3]));
                    
                    dH = dHcurr;
                    dC = dCcurr;
                end
                
                b_H_deltas(:, :, i) = b_H_deltas(:, :, i) / num_steps;
                b_C_deltas(:, :, i) = b_C_deltas(:, :, i) / num_steps;
                w_IH_deltas(:, :, i) = w_IH_deltas(:, :, i) / num_steps;
                w_CH_deltas(:, :, i) = w_CH_deltas(:, :, i) / num_steps;
                w_HC_deltas(:, :, i) = w_HC_deltas(:, :, i) / num_steps;
                if include_intermediates == 1
                    w_CO_deltas(:, :, i) = w_CO_deltas(:, :, i) / num_steps;
                    w_HO_deltas(:, :, i) = w_HO_deltas(:, :, i) / num_steps;
                    b_O_deltas(:, :, i) = b_O_deltas(:, :, i) / num_steps;
                end
            end
            
            dweights.w_IH = mean(w_IH_deltas, 3);
            dweights.w_CH = mean(w_CH_deltas, 3);
            dweights.w_HC = mean(w_HC_deltas, 3);
            dweights.w_HO = mean(w_HO_deltas, 3);
            dweights.w_CO = mean(w_CO_deltas, 3);
            
            dbiases.b_H = mean(b_H_deltas, 3);
            dbiases.b_C = mean(b_C_deltas, 3);
            dbiases.b_O = mean(b_O_deltas, 3);
        end
        
        % feed an input through the layers
        function [a_hid, a_con, a_out, z_hid, z_con, z_out] = runSample(this, input_ex, num_steps)
            input = permute(input_ex, [2 1 3]);
            % activations for the h, c, o layers
            a_hid = zeros(this.Nhidden, 1, num_steps + 1); 
            a_con = zeros(this.Ncontrol, 1, num_steps + 1);
            a_out = zeros(this.Noutput, 1, num_steps + 1);
            
            z_hid = zeros(this.Nhidden, 1, num_steps + 1); 
            z_con = zeros(this.Ncontrol, 1, num_steps + 1);
            z_out = zeros(this.Noutput, 1, num_steps + 1);
            
            w = this.weights;
            b = this.biases;
            
            for i = 1:num_steps
                z_con(:, :, i + 1) = w.w_HC * a_hid(:, :, i) + b.b_C;
                a_con(:, :, i + 1) = this.sigmoid(z_con(:, :, i + 1));
                
                % input doesn't have virtual first layer, so input(i)
                % corresponds to z(i + 1)
                z_hid(:, :, i + 1) = (w.w_IH * input(:, :, i)) + (w.w_CH * a_con(:, :, i + 1)) + b.b_H;
                a_hid(:, :, i + 1) = this.sigmoid(z_hid(:, :, i + 1));
                
                z_out(:, :, i + 1) = w.w_HO * a_hid(:, :, i + 1) + w.w_CO * a_con(:, :, i + 1) + b.b_O;
                a_out(:, :, i + 1) = this.sigmoid(z_out(:, :, i + 1));
            end
        end
        
        function [out] = predictSample(this, input, num_steps)
            [a_hid, a_con, a_out, z_hid, z_con, z_out] = runSample(this, input, num_steps);
            out = a_out(:, :, end);
        end
        
        function [outs] = predictSet(this, input_set, num_steps)
            outs = zeros(size(input_set, 1), this.Noutput, num_steps + 1);
            for i = 1:size(input_set, 1)
                [a_hid, a_con, a_out, z_hid, z_con, z_out] = runSample(this, input_set(i, :, :), num_steps);
                outs(i, :, :) = permute(a_out(:, :, :), [2 1 3]);
            end
        end
        
        function [a_hids, a_cons, a_outs, z_hids, z_cons, z_outs] = predictSetVerbose(this, input_set, num_steps)
            a_outs = zeros(size(input_set, 1), this.Noutput, num_steps + 1);
            a_hids = zeros(size(input_set, 1), this.Nhidden, num_steps + 1);
            a_cons = zeros(size(input_set, 1), this.Ncontrol, num_steps + 1);
            z_outs = zeros(size(input_set, 1), this.Noutput, num_steps + 1);
            z_hids = zeros(size(input_set, 1), this.Nhidden, num_steps + 1);
            z_cons = zeros(size(input_set, 1), this.Ncontrol, num_steps + 1);
            for i = 1:size(input_set, 1)
                [a_hid, a_con, a_out, z_hid, z_con, z_out] = runSample(this, input_set(i, :, :), num_steps);
                a_outs(i, :, :) = permute(a_out(:, :, :), [2 1 3]);
                a_hids(i, :, :) = permute(a_hid(:, :, :), [2 1 3]);
                a_cons(i, :, :) = permute(a_con(:, :, :), [2 1 3]);
                z_outs(i, :, :) = permute(z_out(:, :, :), [2 1 3]);
                z_hids(i, :, :) = permute(z_hid(:, :, :), [2 1 3]);
                z_cons(i, :, :) = permute(z_con(:, :, :), [2 1 3]);
            end
        end
        
        function [cost] = costSample(this, input, num_steps, label)
            [a_hid, a_con, a_out, z_hid, z_con, z_out] = runSample(this, input, num_steps);
            diff = a_out(:, :, end) - transpose(label);
            cost = sum(diff .* diff);
        end
        
        function [cost] = costSet(this, input_sets, num_steps, labels)
            cost = 0;
            for i = 1:size(input_sets, 1)
                cost = cost + this.costSample(input_sets(i, :, :), num_steps, labels(i, :, end));
            end
            
            cost = cost / size(input_sets, 1);
            cost = cost / this.Noutput;
        end
        
        % compute the sigmoid of an input
        function [k] = sigmoid(this, z)
            %k = zeros(size(z, 1), 1);
            k = 1 ./ (1+exp(-z));
%             for i = 1:size(z, 1)
%                 k(i, 1) = 1/(1 + exp(-z(i, 1)));
%             end 
        end
        
        % compute the derivative sigmoid of an input
        function [k] = sigmoid_prime(this, z)
            k = (1 - this.sigmoid(z)) .* this.sigmoid(z); 
        end
        
        % computes the derivative of the error function on the output set,
        % given the correct labels
        function [k] = error(this, output, labels)
            k = output - labels;
        end
        
    end
end