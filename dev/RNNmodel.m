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
            
            this.biases.b_H = (-1 + 2.*rand(this.Nhidden, 1)) * init_scale;
            this.biases.b_O = (-1 + 2.*rand(this.Noutput, 1)) * init_scale;
            this.biases.b_C = (-1 + 2.*rand(this.Ncontrol, 1)) * init_scale;
        end
        
        % train the network using the given input and label set
        % time step is the third dimension of the data sets.
        % num_steps is the number of time steps that the network is
        % simulating.
        function [] = trainOnline(this, input_set, label_set, num_steps, batch_size, num_iterations)
            assert(size(input_set, 3) == size(label_set, 3)); % they better have values for each time step
            assert(size(input_set, 1) == size(label_set, 1)); % they better have the same amount of examples
            assert(num_steps >= 0);
            assert(num_iterations >= 0);
            assert(batch_size >= 1);
            
            num_examples = size(input_set, 1);
            
            for i = 1:num_iterations
                % randomly permute the sets before training
                perm_indices = randperm(num_examples, 1);
                input_set = input_set(perm_indices);
                label_set = label_set(perm_indices);
                
                for j = 1:ceil(num_examples / batch_size)
                    b_start = ((j - 1) * batch_size) + 1;
                    b_end = b_start + batch_size - 1;
                    [dweights, dbiases] = trainBatch(this, input_set(b_start:b_end, :, :), label_set(b_start:b_end, :, :), num_steps);
                    
                    % TODO update weights and biases
                    %
                    %
                    %
                    this.weights.w_IH = this.weights.w_IH - l_rate * dweights.w_IH;
                    this.weights.w_HO = this.weights.w_HO - l_rate * dweights.w_HO;
                    this.weights.w_HC = this.weights.w_HC - l_rate * dweights.w_HC;
                    this.weights.w_CH = this.weights.w_CH - l_rate * dweights.w_CH;
                    this.weights.w_CO = this.weights.w_CO - l_rate * dweights.w_CO;
                    
                    this.biases.b_H = this.biases.b_H - l_rate * dbiases.b_H;
                    this.biases.b_O = this.biases.b_O - l_rate * dbiases.b_O;
                    this.biases.b_C = this.biases.b_C - l_rate * dbiases.b_C; 
                    %
                    %
                    %
                end
            end
        end
        
        % train a batch of inputs 
        function [dweights, dbiases] = trainBatch(this, input_set, label_set, num_steps)
            batch_size = size(input_set, 1);
            for i = 1:batch_size
                % feed forward
                [a_hid, a_con, a_out] = runSample(this, input_set(i, :), num_steps);
                final_out = a_out(:, :, end);
                
                % calculate deltas
                err = error(final_out, label_set(i, :));
                
            end
        end
        
        function [a_hid, a_con, a_out] = runSample(this, input, num_steps)
            % activations for the h, c, o layers
            a_hid = zeros(this.Nhidden, 1, num_steps + 1); 
            a_con = zeros(this.Ncontrol, 1, num_steps + 1);
            a_out = zeros(this.Noutput, 1, num_steps + 1);
            
            sigm_handle = @sigmoid;
            w = this.weights;
            b = this.biases;
            
            % TODO: start-up might be weird...
            for i = 1:num_steps
                a_hid(:, :, i + 1) = arrayfun(sigm_handle, w.w_IH * input(:, :, i) + w.w_CH * a_con(:, :, i) - b.b_H);
                a_con(:, :, i + 1) = arrayfun(sigm_handle, w.w_HC * a_hid(:, :, i) - b.b_C);
                a_out(:, :, i + 1) = arrayfun(sigm_handle, w.w_HO * a_hid(:, :, i) + w.w_CO * a_con(:, :, i) - b.b_O);
            end
        end
        
        % compute the sigmoid of an input
        function k = sigmoid(z)
            k = 1/(1 + exp(-z));    
        end
        
        % compute the derivative sigmoid of an input
        function k = sigmoid_prime(z)
            k = (1 - sigmoid(z)) * sigmoid(z); 
        end
        
        % computes the derivative of the error function on the output set,
        % given the correct labels
        function [varargout] = error(output, labels)
            varargout = output - labels;
        end
        
    end
end