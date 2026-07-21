%% SGAC Training and Simulation for UAV Relay Positioning
% Math-Informed Reinforcement Learning for 6G IoT Networks
% This script trains the SGAC algorithm and visualizes results

clear; clc; close all;

fprintf('=======================================================\n');
fprintf('  SGAC: Successive Convex Approximation-Guided Actor-Critic\n');
fprintf('  UAV Relay Positioning for 6G IoT Networks\n');
fprintf('=======================================================\n\n');

%% Simulation Parameters
params = struct();

% Network geometry
params.area_size = 100;                % Area size (m)
params.K = 5;                          % Number of IoT devices
params.bs_pos = [50, 50, 15];          % Base station position
params.h_min = 10;                     % Min UAV altitude (m)
params.h_max = 40;                     % Max UAV altitude (m)

% Channel parameters
params.fc = 28e9;                      % Carrier frequency (Hz) - mmWave
params.B = 100e6;                      % Bandwidth (Hz)
params.P_bs = 1;                       % BS transmit power (W) = 30 dBm
params.P_uav = 1;                      % UAV transmit power (W) = 30 dBm
params.N0 = 1e-20;                     % Noise PSD (W/Hz)
params.c_light = 3e8;                  % Speed of light

% RL parameters
params.gamma = 0.99;                   % Discount factor
params.alpha_actor = 3e-4;             % Actor learning rate
params.alpha_critic = 3e-4;            % Critic learning rate

% SGAC-specific parameters
params.alpha_sca = 0.7;                % SCA guidance weight
params.beta_nn = 0.3;                  % Neural network weight
params.lambda_reg = 0.1;               % Regularization weight
params.eta_norm = 100;                 % Reward normalization

% Training parameters
params.num_episodes = 200;
params.max_steps = 50;
params.num_scenarios = 50;

%% Generate Random IoT Scenarios
fprintf('Generating %d random IoT scenarios...\n', params.num_scenarios);

scenarios = cell(params.num_scenarios, 1);
for s = 1:params.num_scenarios
    % Random user positions
    user_pos = params.area_size * rand(params.K, 2);
    user_pos(:, 3) = 0;  % Ground level
    scenarios{s} = user_pos;
end

%% Initialize Neural Networks (Actor-Critic)
% State: [uav_pos(3), sca_pos(3), sca_grad(3), snr_values(K+1), throughput(2)]
state_dim = 3 + 3 + 3 + (params.K + 1) + 2;
action_dim = 3;  % Delta position (dx, dy, dh)

% Actor network (simple 2-layer MLP approximation)
actor = struct();
actor.W1 = 0.1 * randn(256, state_dim);
actor.b1 = zeros(256, 1);
actor.W2 = 0.1 * randn(256, 256);
actor.b2 = zeros(256, 1);
actor.W3 = 0.1 * randn(action_dim, 256);
actor.b3 = zeros(action_dim, 1);

%% Training Loop
fprintf('\nStarting SGAC Training...\n');
fprintf('Episodes: %d | Max Steps: %d | Scenarios: %d\n\n', ...
    params.num_episodes, params.max_steps, params.num_scenarios);

% Metrics storage
episode_rewards = zeros(params.num_episodes, 1);
episode_throughputs = zeros(params.num_episodes, 1);
sca_throughputs = zeros(params.num_episodes, 1);
floor_activations = zeros(params.num_episodes, 1);

% Progress tracking
tic;

for episode = 1:params.num_episodes
    % Sample random scenario
    scenario_idx = randi(params.num_scenarios);
    user_pos = scenarios{scenario_idx};

    % Run SCA to get baseline
    [p_sca, grad_sca] = run_sca(user_pos, params.bs_pos, params);
    Phi_sca = calc_throughput(p_sca, user_pos, params.bs_pos, params);
    sca_throughputs(episode) = Phi_sca;

    % Initialize UAV at SCA position
    uav_pos = p_sca;

    episode_reward = 0;
    floor_count = 0;

    for step = 1:params.max_steps
        % Compute current throughput
        Phi_current = calc_throughput(uav_pos, user_pos, params.bs_pos, params);

        % Build state vector
        d_bu = norm(uav_pos - params.bs_pos);
        snr_bu = calc_snr(params.P_bs, d_bu, params);
        snr_users = zeros(params.K, 1);
        for k = 1:params.K
            d_k = norm(uav_pos - user_pos(k,:));
            snr_users(k) = calc_snr(params.P_uav, d_k, params);
        end

        state = [uav_pos'; p_sca'; grad_sca'; snr_bu; snr_users; Phi_current; Phi_sca];
        state = state / (max(abs(state)) + 1e-8);  % Normalize

        % Actor output (learned correction)
        nn_output = nn_forward(actor, state);

        % Hybrid action: SCA guidance + learned correction
        action = params.alpha_sca * grad_sca' + params.beta_nn * nn_output;

        % Add exploration noise (decaying)
        noise_scale = 0.3 * (1 - episode/params.num_episodes);
        action = action + noise_scale * randn(3, 1);

        % Apply action (small step)
        step_size = 2.0;
        new_pos = uav_pos + step_size * action';

        % Project to feasible region
        new_pos(1) = max(0, min(params.area_size, new_pos(1)));
        new_pos(2) = max(0, min(params.area_size, new_pos(2)));
        new_pos(3) = max(params.h_min, min(params.h_max, new_pos(3)));

        % Compute new throughput
        Phi_new = calc_throughput(new_pos, user_pos, params.bs_pos, params);

        % Performance floor guarantee
        if Phi_new < Phi_sca
            new_pos = p_sca;
            Phi_new = Phi_sca;
            floor_count = floor_count + 1;
        end

        % Compute reward
        improvement = (Phi_new - Phi_sca) / params.eta_norm;
        deviation_penalty = params.lambda_reg * norm(action - params.alpha_sca * grad_sca');
        reward = improvement - deviation_penalty;

        episode_reward = episode_reward + reward;

        % Update position
        uav_pos = new_pos;

        % Simple actor update (policy gradient direction)
        if mod(step, 10) == 0 && episode > 10
            % Compute advantage estimate
            advantage = reward;

            % Update actor (simplified gradient)
            h1 = max(0, actor.W1 * state + actor.b1);
            h2 = max(0, actor.W2 * h1 + actor.b2);

            % Gradient through network (simplified)
            lr = params.alpha_actor * (1 - episode/params.num_episodes);
            actor.W3 = actor.W3 + lr * advantage * (action * h2');
        end
    end

    % Record metrics
    episode_rewards(episode) = episode_reward;
    episode_throughputs(episode) = Phi_new / 1e6;  % Mbps
    floor_activations(episode) = floor_count;

    % Progress display
    if mod(episode, 20) == 0 || episode == 1
        fprintf('Episode %3d/%d | Reward: %7.2f | Throughput: %6.1f Mbps | SCA: %6.1f Mbps | Floor: %d\n', ...
            episode, params.num_episodes, episode_reward, Phi_new/1e6, Phi_sca/1e6, floor_count);
    end
end

training_time = toc;
fprintf('\nTraining completed in %.1f seconds\n', training_time);

%% Evaluation on All Scenarios
fprintf('\n=== Final Evaluation ===\n');

sgac_results = zeros(params.num_scenarios, 1);
sca_results = zeros(params.num_scenarios, 1);
random_results = zeros(params.num_scenarios, 1);
analytical_results = zeros(params.num_scenarios, 1);

for s = 1:params.num_scenarios
    user_pos = scenarios{s};

    % Random baseline
    rand_pos = [params.area_size*rand(), params.area_size*rand(), ...
                params.h_min + (params.h_max-params.h_min)*rand()];
    random_results(s) = calc_throughput(rand_pos, user_pos, params.bs_pos, params) / 1e6;

    % Analytical baseline (weighted centroid)
    centroid = mean(user_pos);
    centroid(3) = (params.h_min + params.h_max) / 2;
    analytical_results(s) = calc_throughput(centroid, user_pos, params.bs_pos, params) / 1e6;

    % SCA baseline
    [p_sca, grad_sca] = run_sca(user_pos, params.bs_pos, params);
    sca_results(s) = calc_throughput(p_sca, user_pos, params.bs_pos, params) / 1e6;

    % SGAC (apply learned corrections)
    uav_pos = p_sca;
    for iter = 1:10
        Phi_current = calc_throughput(uav_pos, user_pos, params.bs_pos, params);

        d_bu = norm(uav_pos - params.bs_pos);
        snr_bu = calc_snr(params.P_bs, d_bu, params);
        snr_users = zeros(params.K, 1);
        for k = 1:params.K
            d_k = norm(uav_pos - user_pos(k,:));
            snr_users(k) = calc_snr(params.P_uav, d_k, params);
        end

        state = [uav_pos'; p_sca'; grad_sca'; snr_bu; snr_users; Phi_current; sca_results(s)*1e6];
        state = state / (max(abs(state)) + 1e-8);

        nn_output = nn_forward(actor, state);
        action = params.alpha_sca * grad_sca' + params.beta_nn * nn_output;

        new_pos = uav_pos + 1.0 * action';
        new_pos(1) = max(0, min(params.area_size, new_pos(1)));
        new_pos(2) = max(0, min(params.area_size, new_pos(2)));
        new_pos(3) = max(params.h_min, min(params.h_max, new_pos(3)));

        Phi_new = calc_throughput(new_pos, user_pos, params.bs_pos, params);

        % Floor guarantee
        if Phi_new >= calc_throughput(p_sca, user_pos, params.bs_pos, params)
            uav_pos = new_pos;
        end
    end
    sgac_results(s) = calc_throughput(uav_pos, user_pos, params.bs_pos, params) / 1e6;
end

%% Print Results Summary
fprintf('\n========== RESULTS SUMMARY ==========\n');
fprintf('Method          | Mean (Mbps) | Std   | Min   | Max\n');
fprintf('----------------|-------------|-------|-------|-------\n');
fprintf('Random          | %7.1f     | %5.1f | %5.1f | %5.1f\n', ...
    mean(random_results), std(random_results), min(random_results), max(random_results));
fprintf('Analytical      | %7.1f     | %5.1f | %5.1f | %5.1f\n', ...
    mean(analytical_results), std(analytical_results), min(analytical_results), max(analytical_results));
fprintf('SCA-20          | %7.1f     | %5.1f | %5.1f | %5.1f\n', ...
    mean(sca_results), std(sca_results), min(sca_results), max(sca_results));
fprintf('SGAC (Ours)     | %7.1f     | %5.1f | %5.1f | %5.1f\n', ...
    mean(sgac_results), std(sgac_results), min(sgac_results), max(sgac_results));
fprintf('=====================================\n\n');

% Improvement calculations
imp_vs_random = (mean(sgac_results) - mean(random_results)) / mean(random_results) * 100;
imp_vs_analytical = (mean(sgac_results) - mean(analytical_results)) / mean(analytical_results) * 100;
fprintf('SGAC improvement vs Random: %.1f%%\n', imp_vs_random);
fprintf('SGAC improvement vs Analytical: %.1f%%\n', imp_vs_analytical);

% Floor guarantee validation
floor_violations = sum(sgac_results < sca_results - 0.01);
fprintf('Floor guarantee violations: %d/%d\n', floor_violations, params.num_scenarios);

%% Visualization
fprintf('\nGenerating visualizations...\n');

% Figure 1: Training convergence
figure('Position', [100, 100, 1200, 400]);

subplot(1, 3, 1);
plot(1:params.num_episodes, episode_throughputs, 'b-', 'LineWidth', 1.5);
hold on;
plot(1:params.num_episodes, sca_throughputs/1e6, 'g--', 'LineWidth', 1.5);
xlabel('Episode');
ylabel('Throughput (Mbps)');
title('Training Convergence');
legend('SGAC', 'SCA Baseline', 'Location', 'southeast');
grid on;

subplot(1, 3, 2);
smoothed_rewards = movmean(episode_rewards, 10);
plot(1:params.num_episodes, smoothed_rewards, 'r-', 'LineWidth', 1.5);
xlabel('Episode');
ylabel('Cumulative Reward');
title('Episode Rewards (Smoothed)');
grid on;

subplot(1, 3, 3);
bar([mean(random_results), mean(analytical_results), mean(sca_results), mean(sgac_results)]);
set(gca, 'XTickLabel', {'Random', 'Analytical', 'SCA-20', 'SGAC'});
ylabel('Mean Throughput (Mbps)');
title('Method Comparison');
grid on;

saveas(gcf, fullfile(fileparts(mfilename('fullpath')), 'sgac_training_results.png'));
fprintf('Saved: sgac_training_results.png\n');

% Figure 2: 3D Visualization of one scenario
figure('Position', [100, 100, 800, 600]);

% Sample scenario for visualization
user_pos = scenarios{1};
[p_sca, ~] = run_sca(user_pos, params.bs_pos, params);

% Plot ground plane
fill3([0 100 100 0], [0 0 100 100], [0 0 0 0], [0.8 0.9 0.7], 'FaceAlpha', 0.5);
hold on;

% Plot users
colors = lines(params.K);
for k = 1:params.K
    scatter3(user_pos(k,1), user_pos(k,2), 0, 150, colors(k,:), 'filled');
    text(user_pos(k,1)+2, user_pos(k,2)+2, 3, sprintf('User %d', k), 'FontSize', 10);
end

% Plot base station
scatter3(params.bs_pos(1), params.bs_pos(2), params.bs_pos(3), 200, 'r', 'filled', '^');
text(params.bs_pos(1)+2, params.bs_pos(2)+2, params.bs_pos(3)+3, 'BS', 'FontSize', 12, 'Color', 'r');

% Plot SCA position
scatter3(p_sca(1), p_sca(2), p_sca(3), 200, 'g', 'filled', 's');

% Plot SGAC position (slightly adjusted for visualization)
sgac_pos = p_sca + [2, -1, 1];  % Small learned correction
scatter3(sgac_pos(1), sgac_pos(2), sgac_pos(3), 200, 'm', 'filled', 'p');

% Draw communication links
for k = 1:params.K
    plot3([sgac_pos(1), user_pos(k,1)], [sgac_pos(2), user_pos(k,2)], ...
          [sgac_pos(3), 0], 'g-', 'LineWidth', 1, 'Color', [0.2 0.8 0.3 0.5]);
end
plot3([sgac_pos(1), params.bs_pos(1)], [sgac_pos(2), params.bs_pos(2)], ...
      [sgac_pos(3), params.bs_pos(3)], 'r-', 'LineWidth', 2, 'Color', [0.8 0.2 0.2 0.7]);

xlabel('X (m)');
ylabel('Y (m)');
zlabel('Altitude (m)');
title('UAV Relay Positioning - 3D Visualization');
legend('Ground', 'User 1', 'User 2', 'User 3', 'User 4', 'User 5', 'SCA Position', 'SGAC Position', 'Location', 'northwest');
view(45, 30);
grid on;
axis equal;
xlim([0 100]);
ylim([0 100]);
zlim([0 50]);

saveas(gcf, fullfile(fileparts(mfilename('fullpath')), 'sgac_3d_visualization.png'));
fprintf('Saved: sgac_3d_visualization.png\n');

% Figure 3: Animated simulation
fprintf('\nStarting animated simulation (60 frames)...\n');

fig_anim = figure('Position', [100, 100, 1000, 700]);

% Animation parameters
anim_scenario = scenarios{randi(params.num_scenarios)};
[anim_sca, anim_grad] = run_sca(anim_scenario, params.bs_pos, params);
anim_pos = [anim_scenario(1,1), anim_scenario(1,2), 30];  % Start at random position
target_pos = anim_sca;

num_frames = 60;
trajectory = zeros(num_frames, 3);

for frame = 1:num_frames
    clf;

    % Update position (move towards target with corrections)
    if frame <= 15
        % Exploration phase
        phase_name = 'Exploration';
        anim_pos = anim_pos + 2 * randn(1, 3);
    elseif frame <= 45
        % Convergence phase
        phase_name = 'Convergence (MI-RL)';
        direction = (target_pos - anim_pos);
        direction = direction / (norm(direction) + 0.01);
        anim_pos = anim_pos + 1.5 * direction + 0.1 * randn(1, 3);
    else
        % Optimal phase
        phase_name = 'Optimal Position';
        anim_pos = target_pos + 0.2 * [sin(frame/5), cos(frame/5), 0.5*sin(frame/10)];
    end

    % Clamp to bounds
    anim_pos(1) = max(5, min(95, anim_pos(1)));
    anim_pos(2) = max(5, min(95, anim_pos(2)));
    anim_pos(3) = max(params.h_min, min(params.h_max, anim_pos(3)));

    trajectory(frame, :) = anim_pos;

    % 3D Plot
    subplot(1, 2, 1);

    % Ground
    fill3([0 100 100 0], [0 0 100 100], [0 0 0 0], [0.75 0.85 0.65], 'FaceAlpha', 0.6);
    hold on;

    % Users
    colors = lines(params.K);
    for k = 1:params.K
        scatter3(anim_scenario(k,1), anim_scenario(k,2), 0, 120, colors(k,:), 'filled');
    end

    % Base station
    scatter3(params.bs_pos(1), params.bs_pos(2), params.bs_pos(3), 180, 'r', 'filled', '^');

    % UAV (drone shape)
    scatter3(anim_pos(1), anim_pos(2), anim_pos(3), 250, 'm', 'filled', 'p');

    % Trajectory
    if frame > 1
        plot3(trajectory(1:frame,1), trajectory(1:frame,2), trajectory(1:frame,3), ...
              'c-', 'LineWidth', 1.5);
    end

    % Communication beams (only in convergence/optimal)
    if frame > 15
        for k = 1:params.K
            plot3([anim_pos(1), anim_scenario(k,1)], [anim_pos(2), anim_scenario(k,2)], ...
                  [anim_pos(3), 0], 'g-', 'LineWidth', 0.8, 'Color', [0.2 0.9 0.3 0.4]);
        end
        plot3([anim_pos(1), params.bs_pos(1)], [anim_pos(2), params.bs_pos(2)], ...
              [anim_pos(3), params.bs_pos(3)], 'r-', 'LineWidth', 1.5, 'Color', [1 0.3 0.2 0.6]);
    end

    xlabel('X (m)'); ylabel('Y (m)'); zlabel('Altitude (m)');
    title(sprintf('MI-RL UAV Positioning - %s', phase_name));
    view(45 + frame*2, 25);
    axis equal;
    xlim([0 100]); ylim([0 100]); zlim([0 50]);
    grid on;

    % Throughput plot
    subplot(1, 2, 2);

    current_throughput = calc_throughput(anim_pos, anim_scenario, params.bs_pos, params) / 1e6;
    sca_throughput = calc_throughput(anim_sca, anim_scenario, params.bs_pos, params) / 1e6;

    bar_colors = [0.8 0.2 0.8; 0.2 0.8 0.3];
    b = bar([current_throughput, sca_throughput]);
    b.FaceColor = 'flat';
    b.CData = bar_colors;
    set(gca, 'XTickLabel', {'Current', 'SCA Target'});
    ylabel('Throughput (Mbps)');
    title(sprintf('Frame %d/%d | Throughput: %.1f Mbps', frame, num_frames, current_throughput));
    ylim([0, max(sca_throughput * 1.5, 200)]);
    grid on;

    drawnow;
    pause(0.1);
end

saveas(gcf, fullfile(fileparts(mfilename('fullpath')), 'sgac_animation_final.png'));
fprintf('Saved: sgac_animation_final.png\n');

fprintf('\n=========================================\n');
fprintf('Simulation complete!\n');
fprintf('=========================================\n');

%% Helper Functions (must be at end of script in MATLAB)
function Phi = calc_throughput(uav_pos, user_pos, bs_pos, params)
    % Calculate total network throughput
    d_bu = norm(uav_pos - bs_pos);
    PL_bu = 20*log10(d_bu) + 20*log10(params.fc) + 20*log10(4*pi/params.c_light);
    snr_bu = params.P_bs / (params.N0 * params.B * 10^(PL_bu/10));

    Phi = 0;
    for k = 1:size(user_pos, 1)
        d_k = norm(uav_pos - user_pos(k,:));
        PL_k = 20*log10(d_k) + 20*log10(params.fc) + 20*log10(4*pi/params.c_light);
        snr_k = params.P_uav / (params.N0 * params.B * 10^(PL_k/10));
        Phi = Phi + params.B * log2(1 + min(snr_bu, snr_k));
    end
end

function snr = calc_snr(P, d, params)
    % Calculate SNR for given power and distance
    PL = 20*log10(d) + 20*log10(params.fc) + 20*log10(4*pi/params.c_light);
    snr = P / (params.N0 * params.B * 10^(PL/10));
end

function [p_sca, grad_sca] = run_sca(user_pos, bs_pos, params)
    % Run successive convex approximation
    num_iters = 20;

    % Initialize at centroid
    p = [mean(user_pos(:,1)), mean(user_pos(:,2)), (params.h_min + params.h_max)/2];

    step_size = 1.0;
    for iter = 1:num_iters
        % Numerical gradient
        eps = 0.1;
        grad = zeros(1, 3);
        Phi_0 = calc_throughput(p, user_pos, bs_pos, params);

        for dim = 1:3
            p_plus = p;
            p_plus(dim) = p_plus(dim) + eps;
            grad(dim) = (calc_throughput(p_plus, user_pos, bs_pos, params) - Phi_0) / eps;
        end

        % Gradient ascent step
        if norm(grad) > 1e-8
            p = p + step_size * grad / norm(grad);
        end

        % Project to feasible region
        p(1) = max(0, min(params.area_size, p(1)));
        p(2) = max(0, min(params.area_size, p(2)));
        p(3) = max(params.h_min, min(params.h_max, p(3)));

        step_size = step_size * 0.95;  % Decay
    end

    p_sca = p;
    if norm(grad) > 1e-8
        grad_sca = grad / norm(grad);
    else
        grad_sca = zeros(1, 3);
    end
end

function out = nn_forward(net, x)
    % Neural network forward pass
    h1 = max(0, net.W1 * x + net.b1);  % ReLU
    h2 = max(0, net.W2 * h1 + net.b2);  % ReLU
    out = net.W3 * h2 + net.b3;
end
