function run_simulations(model, ips_str)
    load_system(model);

    send_block = [model, '/TCPSend'];
    receive_block = [model, '/TCPReceive'];

    % Convert the input to a string if it is not already a string
    if ~ischar(ips_str) && ~isstring(ips_str)
        ips_str = char(ips_str);
    end

    % Split the IP string into a cell array of IP addresses
    ips = strsplit(ips_str, ',');
    num_ips = length(ips);
    num_workers = min(feature('numcores'), num_ips + 1);
    
    % Open a parallel pool with the number of workers equal to the number of IPs
    if isempty(gcp('nocreate'))
        parpool('local', num_workers);
    else
        pool = gcp('nocreate');
        if pool.NumWorkers ~= num_workers
            delete(pool);
            parpool('local', num_workers);
        end
    end

    simIn = repmat(Simulink.SimulationInput(model), 1, num_ips);

    for i = 1:num_ips
        simIn(i) = simIn(i).setBlockParameter(send_block, 'Host', ips{i});
        simIn(i) = simIn(i).setBlockParameter(receive_block, 'Host', ips{i});
    end

    % Run simulations and capture output
    simOut = parsim(simIn, 'ShowProgress', 'on');

    % Check for errors in simulation outputs
    errors = cell(1, length(simOut));
    for i = 1:length(simOut)
        if simOut(i).ErrorMessage
            errors{i} = simOut(i).ErrorMessage;
        else
            errors{i} = 'No error';
        end
    end

    % Save errors to a file or print them
    save('simulation_errors.mat', 'errors');
    disp(errors);
end

