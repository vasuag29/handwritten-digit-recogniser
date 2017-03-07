function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));


m = size(X, 1);
         
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

a1 = [ones(m,1) X];
z2 = (a1*Theta1');
a2 = [ones(size(z2,1),1) sigmoid(z2)];
hx = sigmoid(a2 * Theta2');
a3 = hx;
y1=eye(num_labels);
y = y1(y,:);
    
J = (1/m) .* sum(sum( -y.*log(hx)  - (1-y).* log(1-hx)));

J = J + (lambda/(2*m)) * ( sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)) );


delta3 = hx - y;
z2 = [ones(m,1) z2];
delta2 = delta3 * Theta2.*sigmoidGradient(z2) ;

D1 = delta2(:, 2:end)'*a1;
D2 = delta3'*a2;

Theta1_grad = (1/m) .* D1;
Theta2_grad = (1/m) .* D2;


Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + (lambda/m) * Theta1(:,2:end);
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + (lambda/m) * Theta2(:,2:end);


grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
