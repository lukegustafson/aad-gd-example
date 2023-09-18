/*
Begin a new calculation for adjoint (reverse) algorithmic differentiation
*/
function aad_begin()
{
    aad_stack = [];
}

/*
Create an AAD node from a number
*/
function aad_const(c)
{
    var node = {
        value: c,
        parents: [],     //no inputs to a constant
        ddparents: []
    };
    aad_stack.push(node);
    return node;
}

/*
Perform addition on two AAD nodes
*/
function aad_add(node1, node2)
{
    var node = {
        value: node1.value + node2.value, 
        parents: [node1, node2], //inputs to our addition
        ddparents: [1, 1] //derivative is 1 for addition
    };
    aad_stack.push(node);
    return node;
}

/*
Perform subtraction on two AAD nodes: node1 - node2
*/
function aad_sub(node1, node2)
{
    var node = {
        value: node1.value - node2.value, 
        parents: [node1, node2],
        ddparents: [1, -1]
    };
    aad_stack.push(node);
    return node;
}

/*
Perform multiplication on two AAD nodes
*/
function aad_mul(node1, node2)
{
    var node = {
        value: node1.value * node2.value, 
        parents: [node1, node2], //inputs to our multiplication
        //d/dx(xy) = y, d/dy(xy) = x
        ddparents: [node2.value, node1.value ]
    };
    aad_stack.push(node);
    return node;
}

/*
Perform division on two AAD nodes: node1 / node2
*/
function aad_div(node1, node2)
{
	if(node2.value == 0)
		throw "Division by 0";
	
	var node = { 
		value: node1.value / node2.value, 
		parents: [node1, node2], 
		//d/dx(x/y) = 1/y, d/dy(x/y) = -x/y^2
		ddparents: [ 1 / node2.value, -node1.value / (node2.value * node2.value) ] 
	};
	aad_stack.push(node);
	return node;
}

/*
Apply the exponential function on one AAD node
*/
function aad_exp(node)
{
	var x = Math.exp(node.value);

	//Catch errors to help debugging
	if(isNaN(x))
		throw "NAN from exp(" + node.value + ")";

	var ret = { 
		value: x, 
		parents: [node],
		//d/dx(exp(x)) = exp(x)
		ddparents: [x]
	};
	aad_stack.push(ret);
	return ret;
}

/*
Perform exponentiation on two AAD nodes: node1 ** node2
*/
function aad_pow(node1, node2)
{
	if(node1.value == 0)
		return aad_const(0);
	else
		return aad_exp(aad_mul(node2, aad_log(node1)));
}

/*
Maximum of two AAD nodes
*/
function aad_max(node1, node2)
{
	ret = {
		value: Math.max(node1.value, node2.value),
		parents: [node1, node2],
		ddparents: [node1.value >= node2.value ? 1 : 0, node1.value >= node2.value ? 0 : 1]
	}
	aad_stack.push(ret);	
	return ret;
}

/*
Apply the natural logarithm to an AAD node
*/
function aad_log(node)
{
    var x = Math.log(node.value);

    //Catch errors to help debugging
    if(isNaN(x))
        throw "NAN from log(" + node.value + ")";

    var ret = { 
        value: x, 
        parents: [node], 
        //d/dx(log(x)) = 1/x
        ddparents: [1 / node.value]
    };
    aad_stack.push(ret);
    return ret;
}

/*
Calculates the partial derivative of the last AAD node with respect to every 
AAD node created since aad_begin() was called. These partial derivatives
are saved into the "derivative" field of each AAD node.
*/
function aad_calc_derivs()
{
    //Start with the final node. Derivative of the node wrt itself is 1
    aad_stack[aad_stack.length - 1].derivative = 1;

    //Walk up the calculation graph
    for(var n = aad_stack.length - 1; n >= 0; n--)
    {
        var node = aad_stack[n];

        //Apply chain rule to each parent of the node
        for(var i = 0; i < node.parents.length; i++)
        {
            var parent = node.parents[i];
            parent.derivative = (parent.derivative || 0) + 
                node.ddparents[i] * node.derivative;
        }
    }
}

/*
Multiply a vector by a number
*/
function vec_scale(vec, x)
{
	return vec.map(y => x * y);
}

/*
Add two vectors
*/
function vec_add(vec1, vec2)
{
	if(vec1.length != vec2.length)
		throw "unequal vector sizes";
	return vec1.map((x,i) => x + vec2[i]);
}

/*
Euclidean norm of a vector
*/
function vec_norm(vec)
{
	var t = 0;
	for(var i = 0; i < vec.length; i++)
		t += vec[i] * vec[i];
	return Math.sqrt(t);
}

/*
Return true if vectors are equal
*/
function vec_equal(vec1, vec2)
{
	if(vec1.length != vec2.length)
		return false;
	for(var i = 0; i < vec1.length; i++)
		if(vec1[i] != vec2[i])
			return false;
	return true;
}

/*
Performs a very basic gradient descent minimization.

f: function from a vector (array of numbers) to [value, derivative] where "value" is a number and "derivative" is a vector
guess: vector used as the initial starting point
max_iter: maximum iterations to use if tolerance is never reached
*/
function gradient_descent_minimize(f, guess, max_iter)
{
	var iter = 0;
	var step = 1;
	var x = guess;
	
	while(true)
	{
		var [y, dy] = f(x);
		var direction = vec_scale(dy, -1);
		var norm = vec_norm(direction);

		//Check termination condition
		iter++;
		if(step * norm <= 1e-15 * Math.abs(y) || iter >= max_iter)
			return x;
		
		//Shrink step size until we satisfy Armijo rule
		while(true)
		{
			var new_x = vec_add(x, vec_scale(direction, step / norm));
			var new_y = f(new_x)[0];
	
			if(y - new_y > .1 * step * norm || vec_equal(x, new_x))
				break;
			step *= .5;
		}
		
		//Update for next iteration
		step *= 1.1;
		x = new_x;
	}
}

/*
Example using the AAD functions
*/
aad_begin();
var xs = [1, 2, 3, 4].map(aad_const);
var ys = [2, 3, 4, 5].map(aad_const);
var sum = aad_const(0);
for(var i = 0; i < 4; i++)
	sum = aad_add(sum, aad_mul(xs[i], ys[i]));
var result = aad_log(sum);
aad_calc_derivs();
console.log("Value of function: " + result.value);
console.log("Derivatives wrt xs: " + xs.map(x => x.derivative));
console.log("Derivatives wrt ys: " + ys.map(y => y.derivative));

/*
Example using AAD with gradient descent

We optimize the y_i such that the total length of the path:
  (0,0),
  (.01, y_1),
  (.02, y_2),
  ...
  (.99, y_99),
  (1,1)
is minimized.
*/

function objective_function(heights)
{
	//Set up AAD
	aad_begin();
	heights_aad = heights.map(aad_const);
	
	//Sum up distances of 100 segments
	var total_length = aad_const(0);
	for(var i = 0; i < 100; i++)
	{
		var left_height  = i == 0  ? aad_const(0) : heights_aad[i-1];
		var right_height = i == 99 ? aad_const(1) : heights_aad[i];
		var height_diff = aad_sub(left_height, right_height);
		var length = aad_pow(aad_add(aad_const(.0001), aad_mul(height_diff, height_diff)), aad_const(.5));
		total_length = aad_add(total_length, length);
	}
	
	//Run AAD
	aad_calc_derivs();
	
	//Return the total error and the gradient
	return [total_length.value, heights_aad.map(c => c.derivative)];
}

var result = gradient_descent_minimize(objective_function, Array(99).fill(0), 10000);

console.log("Gradient descent result = " + result);
console.log("Objective function = " + objective_function(result)[0]);
