function x_filtered = first_order_filter(x, ts, tau)
%   Apply a first-order low-pass filter to input signal x
%   x   : input signal (vector)
%   ts   : sampling time (scalar)
%   tau : time constant (scalar)

    alpha = ts / (ts + tau);   % Filter coefficient
    x_filtered = zeros(size(x));      % Pre-allocate output

    % Apply the difference equation
    %x_filtered(1) = alpha * x(1);     % Initial condition
    x_filtered(1) = 0;     % Initial condition for zero
    for n = 2:length(x)
        x_filtered(n) = alpha * x(n) + (1 - alpha) * x_filtered(n-1);
    end
end