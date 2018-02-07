// Library Module Implementing Layers of Perceptrons for CY2D7 .. SE2NN11
// Dr Richard Mitchell 15/12/06 ... 18/09/13
// Adapted by 

#include "mlplayer.h"
#include <math.h>
#include <iomanip>
#include <valarray>

// functions students write which return vectors need to be safe
// before student writes the code ... so by default they call this function
vector<double> dummyVector (int num) { 
	vector<double> dummy(num, 0);	
		return dummy;
};		// dummy vector : 

double myrand (void) {			
	// return a random number in the range -1..1
	// do so calling the rand function in math library
   return -1.0 + (2.0 * rand() / RAND_MAX);
}

// Implementation of LinearLayerNetwork *****************************

LinearLayerNetwork::LinearLayerNetwork (int numIns, int numOuts) { 
		// constructor for Layer of linearly activated neurons
		// it is passed the numbers of inputs and of outputs
		// there are numOuts neurons in the layer
		// each neuron has an output, a delta and an error - 
		//     so have an array of outputs, deltas and errors
		// each neuron has numIns+1 weights (first being the bias weight)
		//	   so have large array of weights, and of weight changes
	int ct;   
	numInputs = numIns;							// store number inputs
	numNeurons = numOuts;						// and of outputs in object
	numWeights = (numInputs + 1) * numNeurons;	// for convenience calculate number of weights
												// each neuron has bias + weight for each input
	outputs.resize(numNeurons);					// get space for array for outputs
	deltas.resize(numNeurons);					// and deltas
    weights.resize(numWeights);					// get space for weights
    changeInWeights.resize(numWeights);			// and change in weights

	for (ct=0; ct<numWeights; ct++)  {
		weights[ct] = myrand();					// initialise weights randomly
		changeInWeights[ct] = 0;				// initialise changeInWeights to 0
    }
	for (ct=0; ct < numNeurons; ct++) {			// initialise outputs and deltas to 0
		outputs[ct] = 0;
		deltas[ct] = 0;
	}
}

LinearLayerNetwork::~LinearLayerNetwork() {
	// destructor ... returns all 'new' memory to heap ... do  nowt as all are vectors
}

void LinearLayerNetwork::CalcOutputs(vector<double> ins) {
	// calculate the sum of each input in (ins) * associated weight, for each neuron in the layer
	// store the results in the array of outputs
	// as weights are stored in array, and accessed in order, use a counter auto indexed through array
int wtindex = 0;			// counter/index set to refer to first weight of first neuron
	for (int neuronct=0; neuronct < numNeurons; neuronct++) {			// process each neuron in the layer, in order
		outputs[neuronct] = weights[wtindex++];							// output = bias weight (move index to next weight)
		for (int inputct=0; inputct < numInputs; inputct++)				// for each input
			outputs[neuronct] += ins[inputct] * weights[wtindex++];		// add input * next weight
	}
}

void LinearLayerNetwork::ComputeNetwork (dataset &data) {
		// pass each item in dataset to network and calculate the outputs

	for (int ct=0; ct<data.numData(); ct++) {							// for each item in data set
	    CalcOutputs (data.GetNthInputs(ct));							// calculate the weighted sum of inputs
		StoreOutputs(ct, data);											// copy outputs back into data set
	}
}

void LinearLayerNetwork::StoreOutputs (int n, dataset &data) {
	// copy calculated network outputs into n'th outputs in data
	data.SetNthOutputs(n, outputs);

										// Pass the outputs from the layer back to the data set, data
}

void LinearLayerNetwork::FindDeltas (vector<double> errors) {
	// find the deltas of each neuron in layer, from the errors which are in the array passed here
	deltas = errors; //the deltas for a linear neuron are the same as the errors, this simply applys this rule by assiging the erros variable to the deltas variable 
}

void LinearLayerNetwork::ChangeAllWeights (vector<double> ins, double learnRate, double momentum) {
	// Change all weights in layer - using inputs ins and [learning rate, momentum]
	double numIns;
	int nnwud = 0;

	for (int countNeurons = 0; countNeurons < numNeurons; countNeurons++)
	{
		for (int wct = 0; wct < numInputs + 1; wct++) {
			if (wct == 0) numIns = 1.0; else numIns = ins[wct - 1];
			changeInWeights[nnwud] = numIns * deltas[countNeurons] * learnRate + changeInWeights[nnwud] * momentum;
			weights[nnwud] += changeInWeights[nnwud];
			nnwud++;
			//takes the value of numIns and loops over the wct variable telling it to +1 to numInputs and then uses the function changeInWeights[wct] = numIns * deltas[wct] * learnRate + changeInWeights[wct] * momentum; to take the previouse weight(changeInWeights) and multiply by the deltas, momentum and learn rate and then pass it back to the changeInWeights[wct] variable before assiging these values to weights 
		}

	}
}

void LinearLayerNetwork::AdaptNetwork (dataset &data, double learnRate, double momentum) {
		// pass whole dataset to network : for each item
		//   calculate outputs, copying them back to data
		//   adjust weights using the delta rule : targets are in data
		//     where learnparas[0] is learning rate; learnparas[1] is momentum

	for (int ct=0; ct<data.numData(); ct++) {
				// for each item in data set
		CalcOutputs(data.GetNthInputs(ct));
				// get inputs from data and pass to network to calculate the outputs
		StoreOutputs (ct, data);
				// return calculated outputs from network back to dataset 
		FindDeltas(data.GetNthErrors(ct));
				// get errors from data and so get neuron to calculate the deltas
		ChangeAllWeights(data.GetNthInputs(ct), learnRate, momentum);
				// and then change all the weights, passing inputs and learning constants
	}
}

void LinearLayerNetwork::SetTheWeights (vector<double> initWt) {
	// set the weights of the layer to the values in initWt

	// do so by copying from initWt into object's weights
   weights = initWt;	
   // copy all weights (numweights says how many) from array initWt to layer's array Weights
}

int LinearLayerNetwork::HowManyWeights (void) {
	// return the number of weights in layer
	return numWeights;
}

vector<double> LinearLayerNetwork::ReturnTheWeights () {
	// return in theWts the current value of all the weights in the layer
	return weights;	//this function is asking the compiler to return a value of double (weights) and the return weights; function tells the system to return the vector double weights to the system 		
} 

vector<double> LinearLayerNetwork::PrevLayersErrors () {
	// find weighted sum of deltas in this layer, being errors for prev layer : return result
	vector<double> errorcount(numInputs, 0); //temp vector  for storing outputs

	for (int ctr = 0; ctr < numInputs; ctr++) // loops through the below function 
	{
		for (int ctr2 = 0; ctr2 < numNeurons; ctr2++)
		{
			errorcount[ctr] += deltas[ctr2] * weights[ctr2*(numInputs +1) + ctr + 1]; //tells the network take errorcount vectore at possistion of the counter  and initilise it to the deltas at possistion 0 and * by the weights at possistion of counter plus 1
		}
	}
	return errorcount;	// returns my vector of errorcount thats storing the output of the prev layer errors
}

// Implementation of SigmoidalLayerNetwork *****************************

SigmoidalLayerNetwork::SigmoidalLayerNetwork (int numIns, int numOuts) 
:LinearLayerNetwork (numIns, numOuts) {
	// just use inherited constructor - no extra variables to initialise
}

SigmoidalLayerNetwork::~SigmoidalLayerNetwork() {
	// destructor - does not need to do anything other than call inherited destructor
}

void SigmoidalLayerNetwork::CalcOutputs(vector<double> ins) {		
	// Calculate outputs being Sigmoid (WeightedSum of ins)
	LinearLayerNetwork::CalcOutputs(ins); // takes the function of LinearLayerNetwork calcoutputs and uses inheritance to take the member variables.
	for (int i = 0; i < outputs.size(); i++) //loops through the function for each output of the vector rather than the entire vector at once
	{
		outputs[i] = 1.0 / (1.0 + exp(-outputs[i])); // states that outputs is equal to 1/1 + the expinential function(-outputs)
	}
}

void SigmoidalLayerNetwork::FindDeltas (vector<double> errors) {		
	// Calculate the Deltas for the layer - 
	// For this class these are Outputs * (1 - Outputs) * Errors

	for (int i = 0; i < deltas.size(); i++) //used to loop through the vector for the below function
	{
		deltas[i] = outputs[i] * (1.0 - outputs[i]) *errors[i]; //calculates the deltas by taking the outputs and multiplying the the (1-outputs) and multiplying by errors (sets a value of 1 -outputs *errors *outputs as delta)
	}

}


// Implementation of MultiLayerNetwork *****************************

MultiLayerNetwork::MultiLayerNetwork (int numIns, int numOuts, LinearLayerNetwork *tonextlayer) 
  :SigmoidalLayerNetwork (numIns, numOuts) {
		// construct a hidden layer with numIns inputs and numOuts outputs
		// where (a pointer to) its next layer is in tonextlayer
		// use inherited constructor for hidden layer
		// and attach the pointer to the next layer that is passed
   nextlayer = tonextlayer;
}

MultiLayerNetwork::~MultiLayerNetwork() {
	delete nextlayer;		// remove output layer, then auto-call inherited destructor
}

void MultiLayerNetwork::CalcOutputs(vector<double> ins) {
		// calculate the outputs of network given the inputs ins
	SigmoidalLayerNetwork::CalcOutputs(ins); //calls the inherited function of calcOutputs from the sigmoidlayer 
	nextlayer->CalcOutputs(outputs); // tells the network to point to the next layer and tells it that (this) uses calcOutputs with the outputs veriable as a condidition
}

void MultiLayerNetwork::StoreOutputs(int n, dataset &data) {
	nextlayer->StoreOutputs(n, data);// calls the StoreOutputs function from the previous layer and points (this) to the nextlayer and returns the nth dataset

}

void MultiLayerNetwork::FindDeltas (vector<double> errors) {
	// find all deltas in the network
	nextlayer->FindDeltas(errors); //points to the next layer finddeltas method passing errors as a peramiter
	SigmoidalLayerNetwork::FindDeltas(nextlayer->PrevLayersErrors()); //calls the finddeltas function from the sigmoid layer and points it tothr next layer and tells it (this) is calculated using prev layer errors 
}

void MultiLayerNetwork::ChangeAllWeights(vector<double> ins, double learnRate, double momentum) {
	// Change all weights in network 
	SigmoidalLayerNetwork::ChangeAllWeights(ins, learnRate, momentum); // calls the sigmoid layer change all weights and pases them the peramiters of the method
	nextlayer->ChangeAllWeights(outputs, learnRate, momentum); // this tells the network take the change in weights and move it to the next layer and by taking outputs from previous layer and passing it as a peramiter 

}

void MultiLayerNetwork::SetTheWeights(vector<double> initWt) {
	// load all weights in network using values in initWt
	// initWt has the weights for this layer, followed by those for next layer

		// first numWeights weights are for this layer
		// set vector wthis so has weights from start of initWt til one before numWeights
	vector<double> wthis(initWt.begin(), initWt.begin() + numWeights);

		// set these into this layer 
	SigmoidalLayerNetwork::SetTheWeights(wthis);

		// next copy rest of weights from initWt[numWeights..]  to vector wrest
	vector<double> wrest(initWt.begin() + numWeights, initWt.end());
		// and then send them to the next layer
	nextlayer->SetTheWeights(wrest);
}

int MultiLayerNetwork::HowManyWeights (void) {
	// return the number of weights in network 

	return nextlayer->HowManyWeights() + numWeights; // takes a pointer to the nextlayer and tells the network nextlayer(this) = howmanyweights then add the number of weights (this layer) and return them both 
}

vector<double> MultiLayerNetwork::ReturnTheWeights () {
	// return the weights of the network into theWts
	// the weights in this layer are put at the start, followed by those of next layer

 
	vector<double> tmpWts = weights; //stores the weights on this layer into a tempry vector to prevent overwrighting of data
	vector<double> strWts = nextlayer->ReturnTheWeights(); //takes a tempry vector and assigns it to a pointer of nextlayer(this)  + return the weights inherited from sigmoid in the next layer
	tmpWts.insert(tmpWts.end(), strWts.begin(), strWts.end()); // tells the network, take my vector of tmpWts and add on to the end of tmpWts the vector from nextlayer the start of strWts until the end of strWts
	return tmpWts; //returns the tmpWts vector with the weights of this and the next layer

	}


